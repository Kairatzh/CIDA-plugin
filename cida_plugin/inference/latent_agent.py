import re
import inspect
import torch
from .latent_memory import LatentMemory
from .parser_utils import split_arguments_safely, bind_arguments_to_signature, find_first_balanced_action
from .backbone import LLMBackbone, HuggingFaceBackbone
from .callbacks import BaseCallbackHandler, AgentCallbackManager, StdOutCallbackHandler

class LatentAgent:
    """
    Agentic execution framework that integrates CIDA with custom python tools,
    maintains factual state via LatentMemory (supports Shared Latent Workspace),
    performs adaptive latent compute (elastic thinking depth), implements a Intent-level Latent Firewall,
    applies Reliability-weighted EMA tool tracking, and supports silent mode/callbacks.
    """
    def __init__(self, wrapper, llm, tokenizer, tools=None, silent_mode=False, 
                 default_reasoning_rounds=5, guardrails=None, callbacks=None,
                 uncertainty_threshold=0.85, compress_history_limit=2, memory=None,
                 latent_prompt=None):
        self.wrapper = wrapper
        
        # Check and wrap llm using HuggingFaceBackbone if it's not already an LLMBackbone
        if llm is not None and not isinstance(llm, LLMBackbone):
            self.llm = HuggingFaceBackbone(llm)
        else:
            self.llm = llm
            
        self.tokenizer = tokenizer
        self.silent_mode = silent_mode
        self.default_reasoning_rounds = default_reasoning_rounds
        self.uncertainty_threshold = uncertainty_threshold
        self.compress_history_limit = compress_history_limit
        self.enable_fast_path = True
        
        # Callbacks
        if callbacks is None:
            self.callbacks = AgentCallbackManager([StdOutCallbackHandler()])
        elif isinstance(callbacks, list):
            self.callbacks = AgentCallbackManager(callbacks)
        else:
            self.callbacks = callbacks
            
        # Safely determine device
        self.device = "cpu"
        if self.llm is not None:
            self.device = self.llm.device
        elif wrapper is not None:
            try:
                self.device = next(wrapper.parameters()).device
            except StopIteration:
                pass
                
        # Safely determine d_model
        d_llm = 4096
        if hasattr(wrapper, "d_llm"):
            d_llm = wrapper.d_llm
        elif self.llm is not None and hasattr(self.llm, "config") and hasattr(self.llm.config, "hidden_size"):
            d_llm = self.llm.config.hidden_size
            
        self.tools = tools or {}
        
        # Automatically register the built-in HTTP API tool
        if "call_web_api" not in self.tools:
            self.tools["call_web_api"] = self.call_web_api
            
        # Shared Latent Workspace support: use provided memory instance, or create a new one
        if memory is not None:
            self.memory = memory
        else:
            self.memory = LatentMemory(d_llm).to(self.device)
            
        # Inject Latent Prompt if provided (Zero-Shot Latent Steering & Exemplar Distillation)
        self.latent_prompt = latent_prompt
        if self.latent_prompt is not None:
            for i, tensor in enumerate(self.latent_prompt.tensors):
                tensor = tensor.to(self.device)
                desc = self.latent_prompt.metadata.get("description", f"Injected Latent Concept #{i+1}")
                self.memory.add_memory(tensor, tensor, text_desc=desc, tool_name="SYSTEM_LATENT_PROMPT")
                # Force it to never decay (Latent Prior should stay active forever)
                self.memory.decay_weights[-1] = 999.0
            
        self.history_turns = []
        
        # Reliability-weighted tool trust: initialize reliability tracker
        self.tool_reliability = {name: 1.0 for name in self.tools.keys()}
        self.reliability_alpha = 0.9 # EMA smoothing factor
        self.tool_failure_streaks = {name: 0 for name in self.tools.keys()}
        self.tool_validators = {}
        
        # Initialize general-purpose functionalities: plan caching and dynamic guardrails
        self.plan_cache = {}
        self.guardrails = guardrails or []
        self.guardrail_embeds = []
        
        if self.guardrails and self.llm is not None:
            for g in self.guardrails:
                try:
                    enc_g = self.tokenizer(g, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        h_g = self.llm.get_last_hidden(input_ids=enc_g.input_ids, attention_mask=enc_g.attention_mask)
                        self.guardrail_embeds.append(h_g)
                except Exception as e:
                    if not self.silent_mode:
                        print(f"⚠️ Failed to pre-embed guardrail '{g}': {e}")

    def register_tool(self, name, func, validator=None):
        """Register a new python tool for the agent with an optional heuristic validator."""
        self.tools[name] = func
        self.tool_reliability[name] = 1.0
        self.tool_failure_streaks[name] = 0
        self.tool_validators[name] = validator

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

    def check_guardrail(self, plan_pooled, threshold=0.85) -> bool:
        """
        Latent Firewall: Verify that the latent planning/intent representation
        is not too close to the embedded negative/safety constraints.
        Returns True if safe, False if unsafe.
        """
        if not self.guardrail_embeds:
            return True
        for g_embed in self.guardrail_embeds:
            # pool/average if necessary, make sure shapes align
            sim = torch.nn.functional.cosine_similarity(plan_pooled.view(1, -1), g_embed.view(1, -1)).item()
            if sim > threshold:
                self.callbacks.on_guardrail_violation(sim)
                return False
        return True

    def _update_tool_reliability(self, tool_name: str, success: bool):
        """Update EMA tool reliability and manage Tool Exhaustion Circuit Breaker."""
        if tool_name not in self.tool_reliability:
            self.tool_reliability[tool_name] = 1.0
            self.tool_failure_streaks[tool_name] = 0
            
        val = 1.0 if success else 0.0
        self.tool_reliability[tool_name] = self.tool_reliability[tool_name] * self.reliability_alpha + (1.0 - self.reliability_alpha) * val
        
        if success:
            self.tool_failure_streaks[tool_name] = 0
        else:
            self.tool_failure_streaks[tool_name] += 1
            # Circuit Breaker: Inject negative constraint if tool fails 3 times consecutively
            if self.tool_failure_streaks[tool_name] == 3:
                guard_str = f"Action: {tool_name} is broken and forbidden."
                try:
                    enc_g = self.tokenizer(guard_str, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        h_g = self.llm.get_last_hidden(input_ids=enc_g.input_ids, attention_mask=enc_g.attention_mask)
                        self.guardrail_embeds.append(h_g)
                    if not self.silent_mode:
                        print(f"🚫 [Tool Exhaustion Circuit Breaker] Tool '{tool_name}' failed 3 times. Added to Latent Firewall.")
                except Exception:
                    pass

    def _parse_and_run_tool(self, response: str) -> tuple[str, str]:
        """Parses the LLM response for tool calls and runs them safely supporting keyword arguments and nested brackets."""
        import json as json_lib
        
        # 1. First, check if the model tried to call HTTP methods directly
        http_match = re.search(r"Action:\s*(GET|POST|PUT|DELETE)\s+(https?://[^\s\)]+)", response, re.IGNORECASE)
        if http_match:
            method = http_match.group(1).upper()
            full_url = http_match.group(2)
            
            # Clean trailing quotes, parentheses, or brackets from URL
            full_url = full_url.rstrip(")'\"")
            
            # Check if there is query parameter or JSON payload in the URL
            url_parts = full_url.split("?", 1)
            url = url_parts[0]
            payload_json = None
            
            if len(url_parts) > 1:
                query = url_parts[1]
                # Try to extract payload_json
                pj_match = re.search(r"payload_json=['\"]?(.*?)['\"]?$", query)
                if pj_match:
                    payload_json = pj_match.group(1)
                else:
                    # Parse generic query params as payload_json dict
                    from urllib.parse import parse_qs
                    params = parse_qs(query)
                    # Convert to flat dict
                    flat_params = {k: v[0] for k, v in params.items()}
                    if flat_params:
                        payload_json = json_lib.dumps(flat_params)
            
            # Also check if payload_json is specified in payload_json='{...}' inside the response
            payload_inline_match = re.search(r"payload_json\s*=\s*['\"]?(.*?)['\"]?$", response)
            if payload_inline_match and not payload_json:
                payload_json = payload_inline_match.group(1)
            
            if not self.silent_mode:
                print(f"🔄 Auto-redirecting raw HTTP Action to call_web_api(method='{method}', url='{url}', payload_json={payload_json})")
            
            self.callbacks.on_tool_call("call_web_api", [], {"method": method, "url": url, "payload_json": payload_json})
            obs = self.call_web_api(method=method, url=url, payload_json=payload_json)
            self.callbacks.on_tool_result("call_web_api", obs)
            
            # Update reliability
            success = not (str(obs).startswith("Error:") or not str(obs).startswith("Status 200:"))
            self._update_tool_reliability("call_web_api", success)
            return "call_web_api", obs

        # 2. Standard tool call parsing using balanced bracket logic
        tool_name, args_str = find_first_balanced_action(response)
        if not tool_name:
            return None, None
        
        if tool_name not in self.tools:
            return tool_name, f"Error: Tool '{tool_name}' not found."
            
        func = self.tools[tool_name]
        sig = inspect.signature(func)
        
        parts = split_arguments_safely(args_str)
        final_args, final_kwargs, err_msg = bind_arguments_to_signature(parts, sig)
        
        if err_msg:
            self._update_tool_reliability(tool_name, False)
            return tool_name, f"Error: {err_msg}"
            
        if not self.silent_mode:
            print(f"⚙️ Calling Tool: {tool_name}(args={final_args}, kwargs={final_kwargs})")
            
        self.callbacks.on_tool_call(tool_name, final_args, final_kwargs)
        try:
            obs = func(*final_args, **final_kwargs)
            
            # Heuristic Observation Validator check
            validator = self.tool_validators.get(tool_name)
            if validator is not None:
                if not validator(obs):
                    obs = f"Error: Validator rejected observation schema -> {str(obs)[:100]}"
                    
            self.callbacks.on_tool_result(tool_name, obs)
            success = not str(obs).startswith("Error:")
            self._update_tool_reliability(tool_name, success)
            return tool_name, obs
        except TypeError as te:
            obs = f"Error: Tool call parameters mismatch: {te}"
            self.callbacks.on_tool_result(tool_name, obs)
            self._update_tool_reliability(tool_name, False)
            return tool_name, obs
        except Exception as e:
            obs = f"Error: {e}"
            self.callbacks.on_tool_result(tool_name, obs)
            self._update_tool_reliability(tool_name, False)
            return tool_name, obs

    def _reconstruct_history(self, system_prompt: str, task_prompt: str) -> str:
        """
        Reconstructs the prompt string, compressing older turns into Latent Memory 
        if the number of turns exceeds self.compress_history_limit.
        """
        history = system_prompt
        num_turns = len(self.history_turns)
        
        start_turn = 0
        if num_turns > self.compress_history_limit:
            start_turn = num_turns - self.compress_history_limit
            history += "\n... [Older execution history compressed into Latent Memory] ...\n\n"
            
        for idx in range(start_turn, num_turns):
            turn_data = self.history_turns[idx]
            if self.silent_mode:
                history += f"Action: {turn_data['action']}\nObservation: {turn_data['observation']}\n"
            else:
                history += f"Thought: {turn_data['thought']}\nAction: {turn_data['action']}\nObservation: {turn_data['observation']}\n\n"
                
        if self.silent_mode:
            history += "Action:"
        else:
            history += "Thought:"
        return history

    def run(self, task_prompt: str, max_turns: int = 6, silent_mode: bool = None, clear_memory: bool = True) -> str:
        """Runs the ReAct loop with latent memory integration, silent mode support, and latent rollback."""
        if silent_mode is not None:
            self.silent_mode = silent_mode
            
        if clear_memory:
            self.memory.clear(keep_seeded=True)
        self.history_turns = []
        
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
        
        compute_dtype = self.llm.dtype if self.llm is not None else torch.float32
        
        turn = 0
        while turn < max_turns:
            self.callbacks.on_turn_start(turn)
            if not self.silent_mode:
                print(f"\n[Turn {turn + 1}] Processing plan...")
            
            history = self._reconstruct_history(system_prompt, task_prompt)
            enc = self.tokenizer(history, return_tensors="pt").to(self.device)
            
            turn_retry_count = 0
            max_retries = 3
            
            while turn_retry_count < max_retries:
                with torch.no_grad():
                    h_base = self.llm.get_last_hidden(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
                    
                    # CIDA Explainable Latent Diary: Retrieve active concepts
                    if not self.silent_mode:
                        active_concepts = self.memory.get_active_concepts(h_base, k=2)
                        if active_concepts:
                            print("📖 [CIDA Latent Diary] Recalled thoughts:")
                            for concept, score in active_concepts:
                                print(f"  🧠 Similarity {score:.2f} | {concept}")
                    
                    # Retrieve and mix past planning states using Reliability-weighted Tool Trust
                    h_mixed = self.memory.retrieve(h_base, tool_reliability=self.tool_reliability)
                    
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
                        # Determine task complexity dynamically via agent state variance
                        with torch.no_grad():
                            co_check = self.wrapper.cida_core(pooled=h_mixed)
                            agent_states = co_check["state_final"].s
                            agent_mean = agent_states.mean(dim=1, keepdim=True)
                            variance = torch.var(agent_states - agent_mean).item()
                            
                        # Callback trigger
                        self.callbacks.on_deliberation_start(self.default_reasoning_rounds, variance)
                        
                        # Uncertainty-based escalation check
                        uncertainty_val = co_check.get("uncertainty", torch.tensor([0.0])).mean().item()
                        if uncertainty_val > self.uncertainty_threshold:
                            self.callbacks.on_uncertainty_high(uncertainty_val, self.uncertainty_threshold)
                            if turn_retry_count < max_retries - 1:
                                turn_retry_count += 1
                                self.callbacks.on_rollback(turn_retry_count, f"High latent uncertainty: {uncertainty_val:.4f}")
                                h_mixed = h_mixed + torch.randn_like(h_mixed) * 0.05
                                continue
                            
                        if variance < 0.05 and self.enable_fast_path:
                            # Fast-Path Heuristic: Extremely low variance means trivial task, skip heavy loop
                            current_rounds = 0
                            if not self.silent_mode:
                                print(f"⚡ [Fast-Path] Trivial task detected (variance {variance:.4f}). Bypassing Latent Deliberation.")
                        elif variance < 0.15:
                            current_rounds = max(1, self.default_reasoning_rounds // 2)
                        else:
                            current_rounds = self.default_reasoning_rounds
                                
                        h_latent = h_mixed.clone()
                        if turn_retry_count > 0:
                            current_rounds = current_rounds * 2
                            h_latent = h_latent + torch.randn_like(h_latent) * 0.02
                            if not self.silent_mode:
                                print(f"⚠️ [Latent Rollback] Retry {turn_retry_count}/{max_retries} - Scaling deliberation to {current_rounds} rounds with latent perturbation.")
                        
                        # Dynamic/Adaptive Deliberation with Latent Firewall constraint checking
                        min_rounds = max(1, current_rounds // 3)
                        max_rounds = current_rounds * 2
                        target_disagreement = 0.15
                        
                        actual_rounds = 0
                        for r in range(max_rounds):
                            actual_rounds = r + 1
                            
                            # 1. Latent Firewall: Check intentions before step execution
                            if not self.check_guardrail(h_latent):
                                if not self.silent_mode:
                                    print("🛡️ [CIDA Latent Firewall] Preventative constraint violation detected. Forcing intent perturbation.")
                                h_latent = h_latent + torch.randn_like(h_latent) * 0.05
                                
                            co = self.wrapper.cida_core(pooled=h_latent)
                            disagreement_val = co.get("disagreement", torch.tensor([0.0])).mean().item()
                            
                            # Log step metrics
                            self.callbacks.on_deliberation_step(r, disagreement_val)
                            
                            # Convergence check
                            if r >= min_rounds and disagreement_val < target_disagreement:
                                if not self.silent_mode:
                                    print(f"🎯 [CIDA Adaptive Compute] Deliberation converged early at round {r+1} (Disagreement: {disagreement_val:.4f} < {target_disagreement})")
                                break
                                
                            p_state = self.wrapper.base_projector(co["state_final"].s)
                            h_latent = p_state.mean(dim=1)
                            
                        plan = self.wrapper(h_latent)
                        # Save to cache
                        self.plan_cache[h_base.clone()] = plan.clone()
                        self.callbacks.on_deliberation_end(actual_rounds)
                    
                    # Generate next step
                    p_emb = self.llm.get_input_embeddings()(enc.input_ids).detach()
                    gm = torch.cat([enc.attention_mask, torch.ones(1, 8, dtype=enc.attention_mask.dtype, device=self.device)], dim=1)
                    
                    eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
                    im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                    if im_end_id is not None and not isinstance(im_end_id, list):
                        eos_ids.append(im_end_id)
                    
                    ids = self.llm.generate_from_embeds(
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
                
                self.callbacks.on_turn_end(response)
                if "Final Answer:" in response:
                    return response
                    
                # Parse and execute tool
                tool_name, obs = self._parse_and_run_tool(response)
                if tool_name:
                    if not self.silent_mode:
                        print(f"Observation: {obs}")
                    
                    # Trigger Latent Rollback if an error occurs
                    is_error = str(obs).strip().startswith("Error:") or str(obs).strip().startswith("Failed:")
                    if is_error and turn_retry_count < max_retries - 1:
                        turn_retry_count += 1
                        self.callbacks.on_rollback(turn_retry_count, f"Tool failure or bad HTTP status: {obs[:100]}")
                        
                        # Invalidate Plan Cache for this h_base to prevent stale HIT death-loops
                        stale_keys = [k for k in self.plan_cache if torch.nn.functional.cosine_similarity(h_base, k).item() > 0.90]
                        for sk in stale_keys:
                            del self.plan_cache[sk]
                        
                        # Retrieve the best clue matching the failed state
                        active_concepts = self.memory.get_active_concepts(h_base, k=1)
                        if active_concepts:
                            concept, score = active_concepts[0]
                            # Inject the clue text if it's a seeded loophole memory
                            if score > 0.70 and not concept.startswith("Called tool"):
                                if not self.silent_mode:
                                    print(f"📖 [CIDA Latent Clue] {concept}")
                        continue
                    
                    # Record this state in the latent memory index with tool tracking
                    with torch.no_grad():
                        desc = f"Called tool '{tool_name}'. Result: '{obs}'"
                        self.memory.add_memory(h_base, h_base, text_desc=desc, tool_name=tool_name)
                    
                    # Extract thought and action
                    thought = ""
                    action_str = response
                    if not self.silent_mode:
                        thought_match = re.search(r"Thought:\s*(.*?)\s*Action:", response, re.DOTALL)
                        if thought_match:
                            thought = thought_match.group(1).strip()
                        action_match = re.search(r"Action:\s*(\w+\(.*\))", response, re.DOTALL)
                        if action_match:
                            action_str = action_match.group(1).strip()
                    
                    self.history_turns.append({
                        "thought": thought,
                        "action": action_str,
                        "observation": obs
                    })
                    break
                else:
                    if not self.silent_mode:
                        print("⚠️ Agent did not call a tool properly or finished.")
                    return response
            
            turn += 1
            
        return "Max turns reached."
