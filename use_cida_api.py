import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cida_plugin.config import CIDAPluginConfig
from cida_plugin.core import CIDAPlugin
from cida_plugin.projector import LatentAEProjector
from cida_plugin.hf import wrap_hf_causal_lm

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print("1. Loading Tokenizer and Base LLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    base_model.eval()
    
    print("\n2. Initializing CIDA Plugin and Plan Projector...")
    # Initialize configuration (matching the dimensions of the base model)
    cfg = CIDAPluginConfig(
        d_input=base_model.config.hidden_size,
        d_hidden=1024,
        d_message=1024,
        num_agents=4,
        num_classes=2,
        max_rounds=4,
        num_attn_heads=8,
        use_perspective_projector=True,
        anonymous_messages=True,
        sparse_comm_k=3,
        use_dynamic_trust=True
    )
    
    # Initialize CIDA and Projector
    cida = CIDAPlugin(cfg).to(DEVICE).float()
    proj = LatentAEProjector(cfg.d_hidden, base_model.config.hidden_size, cfg.num_agents, num_plan=8).to(DEVICE)
    
    # Load your trained weights here if available:
    # cida.load_state_dict(torch.load("cida_v5_rl_weights/pytorch_model.bin", map_location=DEVICE))
    # proj.load_state_dict(torch.load("cida_v5_rl_weights/projector.pt", map_location=DEVICE))
    
    print("\n3. Wrapping Causal LLM with CIDA (Plug-and-Play)...")
    # This single call wraps the model with CIDA's latent deliberation loop!
    cida_model = wrap_hf_causal_lm(base_model, cida, proj)
    
    # Define a complex logical reasoning task
    prompt = "A is older than B, B is older than C. Is C older than A?"
    inputs = tokenizer(f"<|im_start|>system\nSolve logically.\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(DEVICE)
    
    print(f"\n4. Prompt: '{prompt}'")
    print("Running inference with CIDA...")
    
    # Run the wrapped generate method
    # Inside, CIDA intercepts the hidden states, runs 4 agents debate,
    # projects the consensus to 8 virtual plan tokens, and prepends them.
    with torch.no_grad():
        outputs = cida_model.generate_with_cida(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            do_sample=False
        )
        
    # Decode the answer (excluding the prompt tokens)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print("\n🟢 CIDA Guided Answer:")
    print(generated_text)
    
    print("\n5. Peeking inside the CIDA Deliberation Box:")
    # We can also call the forward pass to inspect the internal agent metrics
    with torch.no_grad():
        # Get the hidden states from the LLM prompt forward pass
        llm_out = base_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
        h_base = llm_out.hidden_states[-1][0, -1].float().unsqueeze(0)
        
        # Run CIDA agents
        cida_out = cida(pooled=h_base)
        
    print(f"   - Number of Agents: {cfg.num_agents}")
    print(f"   - Deliberation Rounds: {cfg.max_rounds}")
    print(f"   - Agent Disagreement: {cida_out['disagreement'].mean().item():.4f} (low is better)")
    print(f"   - Uncertainty: {cida_out['uncertainty'].mean().item():.4f}")
    print(f"   - Injected Plan Dimension: {proj(cida_out['state_final'].s).shape} (8 plan tokens)")

if __name__ == "__main__":
    main()
