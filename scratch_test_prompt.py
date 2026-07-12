q = "q"
correct_ans = "ans"
ans_base = "base"
ans_v5 = "v5"

prompt = (
    f"<|im_start|>system\n"
    f"You are a precise grading assistant. Evaluate if the answers of Model A and Model B are correct according to the Reference Correct Answer.\n"
    f"Respond ONLY with a JSON object in this format: {{\"base_correct\": 0 or 1, \"cida_correct\": 0 or 1}}\n"
    f"<|im_end|>\n"
)
print("Success!")
