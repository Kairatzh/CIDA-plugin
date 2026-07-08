import re
import inspect
import torch
from .latent_memory import LatentMemory

class LatentAgent:
    """
    Agentic execution framework that integrates CIDA with custom python tools
    and maintains factual state via LatentMemory.
    """
    def __init__(self, wrapper, llm, tokenizer, tools=None):
        self.wrapper = wrapper
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = next(wrapper.parameters()).device
        self.tools = tools or {}
        
        # Initialize the Latent Memory scratchpad
        self.memory = LatentMemory(wrapper.d_llm).to(self.device)

    def register_tool(self, name, func):
        """Register a new python tool for the agent."""
        self.tools[name] = func

    def _parse_and_run_tool(self, response: str) -> str:
        """Parses the LLM response for tool calls and runs them safely."""
        action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", response)
        if not action_match:
            return None, None
            
        tool_name = action_match.group(1)
        args_str = action_match.group(2)
        
        # Clean arguments: remove variable names (e.g. flight_id='FL-303' -> 'FL-303')
        args = []
        for a in args_str.split(","):
            a = a.strip()
            if "=" in a:
                a = a.split("=")[1].strip()
            a = a.strip('"').strip("'")
            if a:
                args.append(a)
                
        if tool_name not in self.tools:
            return tool_name, f"Error: Tool '{tool_name}' not found."
            
        print(f"⚙️ Calling Tool: {tool_name}({args})")
        try:
            # Execute python tool
            obs = self.tools[tool_name](*args)
            return tool_name, obs
        except TypeError as te:
            return tool_name, f"Error: Tool call parameters mismatch: {te}"
        except Exception as e:
            return tool_name, f"Error: {e}"

    def run(self, task_prompt: str, max_turns: int = 6) -> str:
        """Runs the ReAct loop with latent memory integration."""
        self.memory.clear()
        
        # Generate rich dynamic description with signatures and docstrings
        tools_list = []
        for name, func in self.tools.items():
            sig = inspect.signature(func)
            doc = func.__doc__.strip().split("\n")[0] if func.__doc__ else "Execute tool action."
            tools_list.append(f"- {name}{sig}: {doc}")
        tools_desc = "\n".join(tools_list)
        
        system_prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant equipped with tools. Solve the task step-by-step.\n"
            "Stop generation immediately after closing the code/action block. Do not simulate observations.\n\n"
            f"You have access to the following tools:\n{tools_desc}\n\n"
            "Format:\n"
            "Thought: <reasoning>\n"
            "Action: <tool_name>(args)\n<|im_end|>\n"
            f"<|im_start|>user\n{task_prompt}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        history = system_prompt
        compute_dtype = next(self.llm.parameters()).dtype
        
        for turn in range(max_turns):
            print(f"\n[Turn {turn + 1}] Processing plan...")
            enc = self.tokenizer(history, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.llm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
                h_base = out.hidden_states[-1][0, -1].float().unsqueeze(0)
                
                # CIDA Explainable Latent Diary: Retrieve active concepts
                active_concepts = self.memory.get_active_concepts(h_base, k=2)
                if active_concepts:
                    print("📖 [CIDA Latent Diary] Recalled thoughts:")
                    for concept, score in active_concepts:
                        print(f"  🧠 Similarity {score:.2f} | {concept}")
                
                # Retrieve and mix past planning states from LatentMemory
                h_mixed = self.memory.retrieve(h_base)
                
                # Project the CIDA plan based on mixed representation
                plan = self.wrapper(h_mixed)
                
                # Generate next step
                p_emb = self.llm.get_input_embeddings()(enc.input_ids).detach()
                gm = torch.cat([enc.attention_mask, torch.ones(1, 8, dtype=enc.attention_mask.dtype, device=self.device)], dim=1)
                
                ids = self.llm.generate(
                    inputs_embeds=torch.cat([p_emb.to(compute_dtype), plan.to(compute_dtype)], dim=1),
                    attention_mask=gm,
                    max_new_tokens=250,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                response = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            
            # Truncate response to keep only up to the first action call, preventing model simulation
            action_match = re.search(r"(Action:\s*\w+\([^)]*\))", response)
            if action_match:
                action_end_idx = response.find(action_match.group(1)) + len(action_match.group(1))
                response = response[:action_end_idx].strip()
            elif "Observation:" in response:
                response = response.split("Observation:")[0].strip()
                
            print(response)
            
            if "Final Answer:" in response:
                return response
                
            # Parse and execute tool
            tool_name, obs = self._parse_and_run_tool(response)
            if tool_name:
                print(f"Observation: {obs}")
                
                # Record this state in the latent memory index along with the text description
                with torch.no_grad():
                    desc = f"Called tool '{tool_name}'. Result: '{obs}'"
                    self.memory.add_memory(h_base, h_base, text_desc=desc)
                
                history += response + f"\nObservation: {obs}\n"
            else:
                print("⚠️ Agent did not call a tool properly or finished.")
                return response
                
        return "Max turns reached."
