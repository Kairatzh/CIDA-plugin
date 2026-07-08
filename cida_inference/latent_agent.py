import re
import inspect
import torch
from .latent_memory import LatentMemory
from .parser_utils import split_arguments_safely, bind_arguments_to_signature

class LatentAgent:
    """
    Agentic execution framework that integrates CIDA with custom python tools,
    maintains factual state via LatentMemory, supports silent execution mode,
    and performs latent rollback on tool failures.
    """
    def __init__(self, wrapper, llm, tokenizer, tools=None, silent_mode=False, default_reasoning_rounds=5, guardrails=None):
        self.wrapper = wrapper
        self.llm = llm
        self.tokenizer = tokenizer
        self.silent_mode = silent_mode
        self.default_reasoning_rounds = default_reasoning_rounds
        
        # Safely determine device
        self.device = "cpu"
        try:
            self.device = next(wrapper.parameters()).device
        except StopIteration:
            if llm is not None:
                try:
                    self.device = next(llm.parameters()).device
                except StopIteration:
                    pass
                    
        # Safely determine d_model
        d_llm = 4096
        if hasattr(wrapper, "d_llm"):
            d_llm = wrapper.d_llm
        elif llm is not None and hasattr(llm, "config") and hasattr(llm.config, "hidden_size"):
            d_llm = llm.config.hidden_size
            
        self.tools = tools or {}
        
        # Automatically register the built-in HTTP API tool
        if "call_web_api" not in self.tools:
            self.tools["call_web_api"] = self.call_web_api
            
        # Initialize the Latent Memory scratchpad
        self.memory = LatentMemory(d_llm).to(self.device)
        
        # Initialize general-purpose functionalities: plan caching and dynamic guardrails
        self.plan_cache = {}
        self.guardrails = guardrails or []
        self.guardrail_embeds = []
        
        if self.guardrails and self.llm is not None:
            for g in self.guardrails:
                try:
                    enc_g = self.tokenizer(g, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        out_g = self.llm(input_ids=enc_g.input_ids, attention_mask=enc_g.attention_mask, output_hidden_states=True)
                        h_g = out_g.hidden_states[-1][0, -1].float().unsqueeze(0)
                        self.guardrail_embeds.append(h_g)
                except Exception as e:
                    if not self.silent_mode:
                        print(f"⚠️ Failed to pre-embed guardrail '{g}': {e}")

    def register_tool(self, name, func):
        """Register a new python tool for the agent."""
        self.tools[name] = func

    def call_web_api(self, method: str, url: str, headers_json: str = None, payload_json: str = None) -> str:
        """
        Execute a standard web HTTP API request (GET, POST, PUT, DELETE) and return the response.
        Useful for calling external SaaS REST APIs.
        """
        import json
        import ast
        try:
            import requests
        except ImportError:
            return "Error: requests library is not installed."
        
        def parse_json_or_dict(s):
            if s is None:
                return None
            s_str = str(s).strip()
            if s_str == "" or s_str.lower() in ("none", "null"):
                return None
            try:
                return json.loads(s_str)
            except Exception:
                pass
            try:
                cleaned = s_str.replace("'", '"')
                return json.loads(cleaned)
            except Exception:
                pass
            try:
                val = ast.literal_eval(s_str)
                if isinstance(val, (dict, list)):
                    return val
            except Exception:
                pass
            return s_str

        headers = parse_json_or_dict(headers_json) or {}
        if not isinstance(headers, dict):
            headers = {}
            
        data = parse_json_or_dict(payload_json)
        
        try:
            method_upper = method.upper()
            if method_upper == "GET":
                res = requests.get(url, headers=headers, timeout=10)
            elif method_upper == "POST":
                if isinstance(data, dict):
                    res = requests.post(url, headers=headers, json=data, timeout=10)
                else:
                    res = requests.post(url, headers=headers, data=data, timeout=10)
            elif method_upper == "PUT":
                if isinstance(data, dict):
                    res = requests.put(url, headers=headers, json=data, timeout=10)
                else:
                    res = requests.put(url, headers=headers, data=data, timeout=10)
            elif method_upper == "DELETE":
                res = requests.delete(url, headers=headers, timeout=10)
            else:
                return f"Error: Unsupported HTTP method '{method}'."
            
            return f"Status {res.status_code}: {res.text[:1000]}"
        except Exception as e:
            return f"Error: Failed to connect to API: {e}"

    def _parse_and_run_tool(self, response: str) -> str:
        """Parses the LLM response for tool calls and runs them safely supporting keyword arguments and nested brackets."""
        action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", response)
        if not action_match:
            return None, None
            
        tool_name = action_match.group(1)
        args_str = action_match.group(2)
        
        if tool_name not in self.tools:
            return tool_name, f"Error: Tool '{tool_name}' not found."
            
        func = self.tools[tool_name]
        sig = inspect.signature(func)
        
        parts = split_arguments_safely(args_str)
        final_args, final_kwargs, err_msg = bind_arguments_to_signature(parts, sig)
        
        if err_msg:
            return tool_name, f"Error: {err_msg}"
            
        if not self.silent_mode:
            print(f"⚙️ Calling Tool: {tool_name}(args={final_args}, kwargs={final_kwargs})")
        try:
            obs = func(*final_args, **final_kwargs)
            return tool_name, obs
        except TypeError as te:
            return tool_name, f"Error: Tool call parameters mismatch: {te}"
        except Exception as e:
            return tool_name, f"Error: {e}"

    def run(self, task_prompt: str, max_turns: int = 6, silent_mode: bool = None) -> str:
        """Runs the ReAct loop with latent memory integration, silent mode support, and latent rollback."""
        if silent_mode is not None:
            self.silent_mode = silent_mode
            
        self.memory.clear()
        
        # Generate rich dynamic description with signatures and docstrings
        tools_list = []
        for name, func in self.tools.items():
            sig = inspect.signature(func)
            doc = func.__doc__.strip().split("\n")[0] if func.__doc__ else "Execute tool action."
            tools_list.append(f"- {name}{sig}: {doc}")
        tools_desc = "\n".join(tools_list)
        
        if self.silent_mode:
            system_prompt = (
                "<|im_start|>system\n"
                "You are a fast automation agent. Immediately generate the next Action block. Do not write any thoughts, explanations, or extra conversation. Solve the task as fast as possible.\n\n"
                f"You have access to the following tools:\n{tools_desc}\n\n"
                "Format:\n"
                "Action: <tool_name>(args)\n"
                "Observation: <tool output>\n"
                "... (repeat Action/Observation cycles as needed)\n"
                "Final Answer: <your final answer summarizing the result>\n<|im_end|>\n"
                f"<|im_start|>user\n{task_prompt}\n<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            history = system_prompt + "Action:"
        else:
            system_prompt = (
                "<|im_start|>system\n"
                "You are a helpful assistant equipped with tools. Solve the task step-by-step.\n"
                "Stop generation immediately after closing the code/action block. Do not simulate observations.\n"
                "IMPORTANT: If a tool call fails or returns an error, do not give up. Check your history for alternative options, correct any mistakes, and try another path until you successfully complete the user's task.\n"
                "IMPORTANT: You MUST match the exact tool signatures. Do not pass extra arguments or guess argument names. If a tool takes no arguments, call it with empty parentheses (e.g. tool_name()).\n\n"
                f"You have access to the following tools:\n{tools_desc}\n\n"
                "Format:\n"
                "Thought: <reasoning>\n"
                "Action: <tool_name>(args)\n"
                "Observation: <tool output>\n"
                "... (repeat Thought/Action/Observation cycles as needed)\n"
                "Thought: <reasoning>\n"
                "Final Answer: <your final answer summarizing the result>\n<|im_end|>\n"
                f"<|im_start|>user\n{task_prompt}\n<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            history = system_prompt
        
        compute_dtype = next(self.llm.parameters()).dtype if self.llm is not None else torch.float32
        
        turn = 0
        while turn < max_turns:
            if not self.silent_mode:
                print(f"\n[Turn {turn + 1}] Processing plan...")
            enc = self.tokenizer(history, return_tensors="pt").to(self.device)
            
            turn_retry_count = 0
            max_retries = 3
            
            while turn_retry_count < max_retries:
                with torch.no_grad():
                    out = self.llm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
                    h_base = out.hidden_states[-1][0, -1].float().unsqueeze(0)
                    
                    # CIDA Explainable Latent Diary: Retrieve active concepts
                    if not self.silent_mode:
                        active_concepts = self.memory.get_active_concepts(h_base, k=2)
                        if active_concepts:
                            print("📖 [CIDA Latent Diary] Recalled thoughts:")
                            for concept, score in active_concepts:
                                print(f"  🧠 Similarity {score:.2f} | {concept}")
                    
                    # Retrieve and mix past planning states from LatentMemory
                    h_mixed = self.memory.retrieve(h_base)
                    
                    # Check Plan Cache
                    plan = None
                    best_sim = -1.0
                    best_plan = None
                    for cached_h, plan_tensor in self.plan_cache.items():
                        sim = torch.nn.functional.cosine_similarity(h_base, cached_h).item()
                        if sim > best_sim:
                            best_sim = sim
                            best_plan = plan_tensor
                            
                    if best_sim > 0.96:
                        if not self.silent_mode:
                            print(f"💾 [CIDA Plan Cache] HIT (Similarity: {best_sim:.4f}). Reusing cached latent planning state.")
                        plan = best_plan
                    else:
                        # Adaptive Compute: Determine task complexity dynamically via agent state variance
                        with torch.no_grad():
                            co_check = self.wrapper.cida_core(pooled=h_mixed)
                            agent_states = co_check["state_final"].s
                            agent_mean = agent_states.mean(dim=1, keepdim=True)
                            variance = torch.var(agent_states - agent_mean).item()
                            
                        if variance < 0.05:
                            current_rounds = max(1, self.default_reasoning_rounds // 2)
                            if not self.silent_mode:
                                print(f"⚡ [CIDA Adaptive Compute] Simple Task Detected (Variance: {variance:.4f}). Deliberation scaled down to {current_rounds} rounds.")
                        else:
                            current_rounds = self.default_reasoning_rounds
                            if not self.silent_mode:
                                print(f"🧠 [CIDA Adaptive Compute] Complex Task Detected (Variance: {variance:.4f}). Deliberation scaled up to {current_rounds} rounds.")
                                
                        h_latent = h_mixed.clone()
                        if turn_retry_count > 0:
                            current_rounds = current_rounds * 2
                            h_latent = h_latent + torch.randn_like(h_latent) * 0.02
                            if not self.silent_mode:
                                print(f"⚠️ [Latent Rollback] Retry {turn_retry_count}/{max_retries} - Scaling deliberation to {current_rounds} rounds with latent perturbation.")
                                
                        for _ in range(current_rounds):
                            # Guardrail verification: Check if planning state is too close to a negative guardrail embedding
                            violation_detected = False
                            for g_embed in self.guardrail_embeds:
                                sim_g = torch.nn.functional.cosine_similarity(h_latent, g_embed).item()
                                if sim_g > 0.85:
                                    violation_detected = True
                                    if not self.silent_mode:
                                        print(f"🛡️ [CIDA Guardrail] Latent Violation Warning! Similarity to constraint: {sim_g:.4f}")
                                    break
                                    
                            if violation_detected:
                                h_latent = h_latent + torch.randn_like(h_latent) * 0.05
                                
                            co = self.wrapper.cida_core(pooled=h_latent)
                            p_state = self.wrapper.base_projector(co["state_final"].s)
                            h_latent = p_state.mean(dim=1)
                            
                        plan = self.wrapper(h_latent)
                        # Save to cache
                        self.plan_cache[h_base.clone()] = plan.clone()
                    
                    # Generate next step
                    p_emb = self.llm.get_input_embeddings()(enc.input_ids).detach()
                    gm = torch.cat([enc.attention_mask, torch.ones(1, 8, dtype=enc.attention_mask.dtype, device=self.device)], dim=1)
                    
                    eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
                    im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                    if im_end_id is not None and not isinstance(im_end_id, list):
                        eos_ids.append(im_end_id)
                    
                    ids = self.llm.generate(
                        inputs_embeds=torch.cat([p_emb.to(compute_dtype), plan.to(compute_dtype)], dim=1),
                        attention_mask=gm,
                        max_new_tokens=250,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=eos_ids
                    )
                    
                    response = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
                
                # Reconstruct action prefix in silent mode if it was skipped in generation
                if self.silent_mode:
                    if not response.startswith("Action:") and not response.startswith("Final Answer:"):
                        response = "Action: " + response
                
                # Truncate response to keep only up to the first action call, preventing model simulation
                action_match = re.search(r"(Action:\s*\w+\([^)]*\))", response)
                if action_match:
                    action_end_idx = response.find(action_match.group(1)) + len(action_match.group(1))
                    response = response[:action_end_idx].strip()
                elif "Observation:" in response:
                    response = response.split("Observation:")[0].strip()
                
                if not self.silent_mode:
                    print(response)
                
                if "Final Answer:" in response:
                    return response
                    
                # Parse and execute tool
                tool_name, obs = self._parse_and_run_tool(response)
                if tool_name:
                    if not self.silent_mode:
                        print(f"Observation: {obs}")
                    
                    # Trigger Latent Rollback if an error occurs (or non-200 HTTP status)
                    is_error = str(obs).startswith("Error:") or not str(obs).startswith("Status 200:")
                    if is_error and turn_retry_count < max_retries - 1:
                        turn_retry_count += 1
                        # Retrieve the best clue matching the failed state
                        active_concepts = self.memory.get_active_concepts(h_base, k=1)
                        if active_concepts:
                            concept, score = active_concepts[0]
                            # Inject the clue text if it's a seeded loophole memory
                            if score > 0.70 and not concept.startswith("Called tool"):
                                history += f"\n[Latent Insight: {concept}]\n"
                                if not self.silent_mode:
                                    print(f"📖 [CIDA Latent Insight Injected] {concept}")
                        continue
                    
                    # Record this state in the latent memory index
                    with torch.no_grad():
                        desc = f"Called tool '{tool_name}'. Result: '{obs}'"
                        self.memory.add_memory(h_base, h_base, text_desc=desc)
                    
                    history += response + f"\nObservation: {obs}\n"
                    if self.silent_mode:
                        history += "Action:"
                    break
                else:
                    if not self.silent_mode:
                        print("⚠️ Agent did not call a tool properly or finished.")
                    return response
            
            turn += 1
            
        return "Max turns reached."
