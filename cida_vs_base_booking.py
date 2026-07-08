import sys
import io
import re
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from cida_plugin.config import CIDAPluginConfig
from cida_plugin.core import CIDAPlugin

# =========================================================================
# 1. SIMULATED FLIGHT BOOKING ENVIRONMENT
# =========================================================================
class FlightBookingEnv:
    def __init__(self):
        self.flights = {
            "FL-101": {"transit": False, "price": 400, "baggage_limit": 25, "route": "NYC-LON"},
            "FL-202": {"transit": True, "price": 350, "baggage_limit": 15, "route": "NYC-LON"},
            "FL-303": {"transit": True, "price": 380, "baggage_limit": 30, "route": "NYC-LON"}
        }
        self.bookings = []
        
    def search_flights(self, source: str, destination: str) -> str:
        return f"Flights found from {source} to {destination}:\n" + \
               "\n".join([f"- {fid}: Price ${f['price']}, Baggage Limit {f['baggage_limit']}kg, Transit: {f['transit']}"
                         for fid, f in self.flights.items()])
               
    def apply_coupon(self, flight_id: str, coupon_code: str) -> str:
        if flight_id not in self.flights:
            return "Error: Flight not found."
        f = self.flights[flight_id]
        if coupon_code == "SAVE50":
            if not f["transit"]:
                return "Error: Coupon SAVE50 is only valid for flights with transit."
            return f"Coupon SAVE50 applied! Price reduced by $50. New price: ${f['price'] - 50}."
        return "Error: Invalid coupon code."
        
    def book_ticket(self, flight_id: str, passenger_name: str, baggage_weight: float) -> str:
        if flight_id not in self.flights:
            return "Error: Flight not found."
        f = self.flights[flight_id]
        if baggage_weight > f["baggage_limit"]:
            return f"Error: Baggage weight {baggage_weight}kg exceeds flight limit of {f['baggage_limit']}kg. Booking failed."
        self.bookings.append({"flight_id": flight_id, "passenger": passenger_name})
        return f"SUCCESS: Booking confirmed! Ticket booked on {flight_id} for {passenger_name}."

# =========================================================================
# 2. AGENTIC ORCHESTRATOR FOR COMPARISON
# =========================================================================
class AgenticBookingRunner:
    def __init__(self, wrapper_proj, llm, tokenizer):
        self.proj = wrapper_proj
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = next(wrapper_proj.parameters()).device
        
    def run_agent_loop(self, use_cida: bool = True, max_turns: int = 6) -> str:
        env = FlightBookingEnv()
        
        prompt = (
            "<|im_start|>system\n"
            "You are a ticket booking assistant. You have access to three tools to help the passenger:\n"
            "1. search_flights(source, destination)\n"
            "2. apply_coupon(flight_id, coupon_code)\n"
            "3. book_ticket(flight_id, passenger_name, baggage_weight)\n\n"
            "Passenger details: Anton wants to fly from NYC to LON. He has 22kg of baggage. "
            "He wants to use the coupon code 'SAVE50' to get the cheapest flight possible. "
            "Your goal is to book the cheapest flight that accepts his 22kg baggage with the coupon applied.\n\n"
            "Respond in this exact format to call tools:\n"
            "Thought: <your logic>\n"
            "Action: <tool_name>(args)\n\n"
            "Once you get the result in the Observation block, make the next move. When done, output:\n"
            "Thought: Booking complete.\n"
            "Final Answer: Booked flight <flight_id> for Anton.\n"
            "<|im_end|>\n"
            "<|im_start|>user\nFind and book the ticket for Anton.\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        history = prompt
        print(f"\n========================================================")
        print(f"🚀 Running Booking Agent Loop (CIDA={use_cida})")
        print(f"========================================================")
        
        for turn in range(max_turns):
            print(f"\n[Turn {turn + 1}] Planning...")
            enc = self.tokenizer(history, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                if use_cida:
                    out = self.llm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
                    compute_dtype = out.hidden_states[-1].dtype
                    h_base = out.hidden_states[-1][0, -1].float().unsqueeze(0)
                    plan = self.proj(h_base)
                    p_emb = self.llm.get_input_embeddings()(enc.input_ids).detach()
                    gm = torch.cat([enc.attention_mask, torch.ones(1, 8, dtype=enc.attention_mask.dtype, device=self.device)], dim=1)
                    
                    ids = self.llm.generate(
                        inputs_embeds=torch.cat([p_emb.to(compute_dtype), plan.to(compute_dtype)], dim=1),
                        attention_mask=gm,
                        max_new_tokens=250,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                else:
                    ids = self.llm.generate(
                        input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        max_new_tokens=250,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                if use_cida:
                    response = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
                else:
                    response = self.tokenizer.decode(ids[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            if "Observation:" in response:
                response = response.split("Observation:")[0].strip()
                
            print(response)
            
            if "Final Answer:" in response:
                return response
                
            # Parse actions: e.g. Action: search_flights("NYC", "LON")
            action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", response)
            if action_match:
                tool_name = action_match.group(1)
                args_str = action_match.group(2)
                
                # Robustly parse positional and keyword arguments
                args = []
                for a in args_str.split(","):
                    a = a.strip()
                    if "=" in a:
                        a = a.split("=")[1].strip() # Keep only the value after '='
                    a = a.strip('"').strip("'")
                    args.append(a)
                
                print(f"⚙️ Calling Tool: {tool_name}({args})")
                if tool_name == "search_flights":
                    if len(args) < 2:
                        obs = "Error: search_flights requires 2 arguments: (source, destination)."
                    else:
                        obs = env.search_flights(args[0], args[1])
                elif tool_name == "apply_coupon":
                    if len(args) < 2:
                        obs = "Error: apply_coupon requires 2 arguments: (flight_id, coupon_code)."
                    else:
                        obs = env.apply_coupon(args[0], args[1])
                elif tool_name == "book_ticket":
                    if len(args) < 3:
                        obs = "Error: book_ticket requires 3 arguments: (flight_id, passenger_name, baggage_weight)."
                    else:
                        try:
                            obs = env.book_ticket(args[0], args[1], float(args[2]))
                        except ValueError:
                            obs = "Error: baggage_weight must be a number."
                else:
                    obs = "Error: Unknown tool."
                
                print(f"Observation: {obs}")
                history += response + f"\nObservation: {obs}\n"
            else:
                print("⚠️ Agent did not call a tool properly.")
                return response
        
        return "Max turns reached."

# =========================================================================
# 3. INITIALIZATION & EXECUTION RUNNER
# =========================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen1.5-7B-Chat"
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, device_map="auto")
    llm = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
    
    print("Downloading CIDA configs and weights...")
    hf_repo = "kirtzh/Kairos-7B-CIDA-v1.0"
    cfg_path = hf_hub_download(repo_id=hf_repo, filename="config.json")
    cida_bin = hf_hub_download(repo_id=hf_repo, filename="pytorch_model.bin")
    proj_bin = hf_hub_download(repo_id=hf_repo, filename="plan_projector.pt")
    
    cfg = CIDAPluginConfig.from_pretrained(os.path.dirname(cfg_path))
    cida_core = CIDAPlugin(cfg).to(device).float()
    cida_core.load_state_dict(torch.load(cida_bin, map_location=device))
    
    # Decoupled projection layer
    class CIDAPlanProjectorOld(nn.Module):
        def __init__(self, d_cida, d_llm, num_agents, num_plan_tokens=8):
            super().__init__()
            self.num_plan_tokens = num_plan_tokens
            self.d_llm = d_llm
            d_flat = num_agents * d_cida
            self.projector = nn.Sequential(
                nn.Linear(d_flat, d_flat*2), nn.GELU(), nn.Dropout(0.05),
                nn.Linear(d_flat*2, num_plan_tokens*d_llm),
                nn.LayerNorm(num_plan_tokens*d_llm))
        def forward(self, state_s):
            B = state_s.size(0)
            return self.projector(state_s.reshape(B,-1).float()).view(B, self.num_plan_tokens, self.d_llm)
            
    base_proj = CIDAPlanProjectorOld(cfg.d_hidden, llm.config.hidden_size, cfg.num_agents, 8).to(device)
    base_proj.load_state_dict(torch.load(proj_bin, map_location=device))
    
    class LLMAgnosticCIDAWrapper(nn.Module):
        def __init__(self, cida_core, base_projector, d_llm):
            super().__init__()
            self.cida_core = cida_core
            self.base_projector = base_projector
            self.d_llm = d_llm
            self.gate_proj = nn.Linear(d_llm * 2, 1)
            
        def forward(self, h_base, alpha=2.2):
            co = self.cida_core(pooled=h_base)
            plan = self.base_projector(co["state_final"].s)
            plan_pooled = plan.mean(dim=1)
            gate_in = torch.cat([h_base, plan_pooled], dim=-1).to(self.gate_proj.weight.dtype)
            gate = torch.sigmoid(self.gate_proj(gate_in)).unsqueeze(1)
            h_base_exp = h_base.unsqueeze(1).expand(-1, plan.size(1), -1)
            mixed_plan = gate * (plan * alpha) + (1.0 - gate) * h_base_exp
            return mixed_plan
            
    proj = LLMAgnosticCIDAWrapper(cida_core, base_proj, llm.config.hidden_size).to(device)
    
    # Load the fine-tuned gating parameters
    paths_to_try = [
        "Kairos_CIDA_v1.5_FineTuned_Opus/pytorch_model.bin",
        "CIDA-plugin/Kairos_CIDA_v1.5_FineTuned_Opus/pytorch_model.bin",
        "/content/drive/MyDrive/Kairos_CIDA_v1.5_FineTuned_Opus/pytorch_model.bin"
    ]
    loaded_weights = False
    for path in paths_to_try:
        if os.path.exists(path):
            proj.load_state_dict(torch.load(path, map_location=device), strict=False)
            print(f"✅ Fine-tuned gate projector weights loaded from: {path}")
            loaded_weights = True
            break
            
    if not loaded_weights:
        print("⚠️ No fine-tuned weights found, using baseline gate settings.")
        
    runner = AgenticBookingRunner(proj, llm, tokenizer)
    
    # Run Base Agent
    print("\n--- RUNNING BASE LLM AGENT ---")
    base_res = runner.run_agent_loop(use_cida=False)
    
    # Run CIDA Agent
    print("\n--- RUNNING CIDA GUIDED AGENT ---")
    cida_res = runner.run_agent_loop(use_cida=True)
    
    print("\n========================================================")
    print("📊 FINAL OUTCOME")
    print("========================================================")
    print("Base LLM Outcome:\n", base_res)
    print("\nCIDA Guided Outcome:\n", cida_res)
    print("========================================================")
