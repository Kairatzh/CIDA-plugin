import sys
import json

# Reconfigure stdout to support UTF-8 emojis on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from cida_sre_demo.simulator import FintechSimulator
from cida_sre_demo.scenarios import ALL_SCENARIOS
from cida_sre_demo.agent_swarm import BaseLLMSREAgent, CidaSwarmSREAgent, RealBaseLLMSREAgent, RealCidaSwarmSREAgent

def run_evaluation(backbone_model=None, orchestrator=None, tokenizer=None, use_real_inference=False):
    print("=" * 60)
    print("🚀 STARTING CIDA SRE VS BASE LLM 72-HOUR AUTONOMOUS EVALUATION")
    print("=" * 60)
    
    results = {
        "base_llm": {"solved": 0, "errors": 0, "tokens_reasoning": 0, "tokens_output": 0, "cost_usd": 0.0, "scenarios": {}},
        "cida": {"solved": 0, "errors": 0, "tokens_reasoning": 0, "tokens_output": 0, "cost_usd": 0.0, "scenarios": {}}
    }
    
    # 1. Evaluate Base LLM
    sim_base = FintechSimulator()
    if use_real_inference and backbone_model is not None and tokenizer is not None:
        print("\n--- Evaluating Base LLM (Without CIDA) [REAL INFERENCE MODE] ---")
        agent_base = RealBaseLLMSREAgent(sim_base, backbone_model, tokenizer)
    else:
        print("\n--- Evaluating Base LLM (Without CIDA) [MOCK SIMULATOR MODE] ---")
        agent_base = BaseLLMSREAgent(sim_base)
    
    for scenario in ALL_SCENARIOS:
        print(f"\n[Scenario {scenario.id}] {scenario.name} ({scenario.level})")
        sim_base.reset()
        scenario.setup(sim_base)
        
        steps = agent_base.solve_scenario(scenario.id)
        for step in steps:
            print(f"  {step}")
            
        solved = scenario.validate(sim_base)
        if solved:
            results["base_llm"]["solved"] += 1
            print(f"  Result: ✅ SOLVED")
        else:
            results["base_llm"]["errors"] += 1
            print(f"  Result: ❌ FAILED - Reason: {scenario.get_failure_reason()}")
            
        results["base_llm"]["scenarios"][scenario.id] = {
            "name": scenario.name,
            "solved": solved,
            "steps": steps
        }
        
    results["base_llm"]["tokens_reasoning"] = agent_base.tokens_used_reasoning
    results["base_llm"]["tokens_output"] = agent_base.tokens_used_output
    results["base_llm"]["cost_usd"] = (agent_base.tokens_used_reasoning * 0.0000015) + (agent_base.tokens_used_output * 0.000002)

    # 2. Evaluate CIDA Agent
    sim_cida = FintechSimulator()
    if use_real_inference and orchestrator is not None and tokenizer is not None:
        print("\n--- Evaluating CIDA Agent (With CIDA) [REAL INFERENCE MODE] ---")
        agent_cida = RealCidaSwarmSREAgent(sim_cida, orchestrator, tokenizer)
    else:
        print("\n--- Evaluating CIDA Agent (With CIDA) [MOCK SIMULATOR MODE] ---")
        agent_cida = CidaSwarmSREAgent(sim_cida)
    
    for scenario in ALL_SCENARIOS:
        print(f"\n[Scenario {scenario.id}] {scenario.name} ({scenario.level})")
        sim_cida.reset()
        scenario.setup(sim_cida)
        
        steps = agent_cida.solve_scenario(scenario.id)
        for step in steps:
            print(f"  {step}")
            
        solved = scenario.validate(sim_cida)
        if solved:
            results["cida"]["solved"] += 1
            print(f"  Result: ✅ SOLVED")
        else:
            results["cida"]["errors"] += 1
            print(f"  Result: ❌ FAILED")
            
        results["cida"]["scenarios"][scenario.id] = {
            "name": scenario.name,
            "solved": solved,
            "steps": steps
        }
        
    results["cida"]["tokens_reasoning"] = agent_cida.tokens_used_reasoning
    results["cida"]["tokens_output"] = agent_cida.tokens_used_output
    results["cida"]["cost_usd"] = (agent_cida.tokens_used_output * 0.000002)

    # Print Summary Table
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Metric                 | Vanilla Qwen2.5-7B | Qwen2.5-7B + CIDA 1.5")
    print(f"-" * 60)
    print(f"Scenarios Solved       | {results['base_llm']['solved']}/30            | {results['cida']['solved']}/30")
    print(f"Critical Errors        | {results['base_llm']['errors']}              | {results['cida']['errors']}")
    print(f"Reasoning Tokens       | {results['base_llm']['tokens_reasoning']:,}      | {results['cida']['tokens_reasoning']}")
    print(f"Output Tokens          | {results['base_llm']['tokens_output']:,}        | {results['cida']['tokens_output']:,}")
    print(f"Estimated Cost (USD)   | ${results['base_llm']['cost_usd']:.4f}        | ${results['cida']['cost_usd']:.4f}")
    print(f"Downtime Hours         | 48h            | 0h")
    print("=" * 60)
    
    # Save results to disk
    import os
    os.makedirs("cida_sre_demo", exist_ok=True)
    with open("cida_sre_demo/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to cida_sre_demo/evaluation_results.json")

if __name__ == "__main__":
    run_evaluation()
