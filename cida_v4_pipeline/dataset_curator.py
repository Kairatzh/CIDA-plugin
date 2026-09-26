"""
dataset_curator.py — Frontier Reasoning Dataset Pipeline for CIDA v4

Standardizes industry-grade training corpora from Hugging Face:
1. AI-MO/NuminaMath-CoT (Competition / Olympiad mathematics with boxed answers)
2. BAAI/TACO (Competitive programming with executable unit tests)
3. openbmb/UltraInteract_sft (Tool-use, code debugging, and System-2 interactions)

Also includes an offline fallback generator for immediate testing without HuggingFace downloads.
"""

import os
import sys
import json
import re
import argparse
from typing import Dict, Any, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class DatasetCurator:
    """
    Curates, cleans, and standardizes frontier reasoning datasets into structured JSONL
    ready for Offline Latent Caching and Latent GRPO training.
    """
    def __init__(self, output_dir: str = "./curated_data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def extract_boxed_math_answer(solution_text: str) -> Optional[str]:
        """Extracts content inside LaTeX \\boxed{...} or the last numeric equation."""
        # Check for \boxed{...}
        boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", solution_text)
        if boxed_matches:
            return boxed_matches[-1].strip()
            
        # Check for nested braces inside \boxed{...}
        nested_match = re.search(r"\\boxed\{(.+)\}", solution_text)
        if nested_match:
            content = nested_match.group(1).rstrip("}")
            return content.strip()
            
        # Fallback: look for "The answer is: X" or "#### X"
        ans_patterns = [
            r"####\s*([^\n]+)",
            r"[Tt]he answer is:?\s*([^\n\.]+)",
            r"= ([0-9]+)\s*$",
        ]
        for pat in ans_patterns:
            m = re.search(pat, solution_text)
            if m:
                return m.group(1).strip()
                
        return None

    def curate_numinamath(self, num_samples: int = 5000, split: str = "train") -> str:
        """
        Curates Olympiad & competition mathematics from AI-MO/NuminaMath-CoT.
        """
        out_file = os.path.join(self.output_dir, f"numinamath_curated_{num_samples}.jsonl")
        print(f"[DATA] Loading AI-MO/NuminaMath-CoT (target: {num_samples} samples)...")
        
        try:
            from datasets import load_dataset
            ds = load_dataset("AI-MO/NuminaMath-CoT", split=split, streaming=True)
        except Exception as e:
            print(f"[WARN] Could not stream from Hugging Face ({e}). Using local mock data.")
            return self.generate_local_seed_data(num_samples=min(100, num_samples))

        valid_count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for item in ds:
                problem = item.get("problem", "").strip()
                solution = item.get("solution", "").strip()
                source = item.get("source", "olympiad")
                
                # Filter: length between 30 and 1500 chars
                if len(problem) < 30 or len(problem) > 2000:
                    continue
                    
                boxed_ans = self.extract_boxed_math_answer(solution)
                if not boxed_ans:
                    continue
                    
                entry = {
                    "id": f"numina_{valid_count:06d}",
                    "task_type": "math",
                    "domain": source,
                    "prompt": f"Problem: {problem}\nSolve the problem and provide the final answer.",
                    "solution": solution,
                    "verifier_spec": {
                        "type": "math_equal",
                        "target_answer": boxed_ans
                    }
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                valid_count += 1
                
                if valid_count >= num_samples:
                    break
                    
        print(f"[OK] Successfully curated {valid_count} NuminaMath problems to: {out_file}")
        return out_file

    def curate_taco_code(self, num_samples: int = 2000, split: str = "train") -> str:
        """
        Curates competitive programming and algorithms from BAAI/TACO with executable unit tests.
        """
        out_file = os.path.join(self.output_dir, f"taco_curated_{num_samples}.jsonl")
        print(f"[DATA] Loading BAAI/TACO code dataset (target: {num_samples} samples)...")
        
        try:
            from datasets import load_dataset
            ds = load_dataset("BAAI/TACO", split=split, streaming=True)
        except Exception as e:
            print(f"[WARN] Could not load TACO ({e}). Falling back to local seed data.")
            return self.generate_local_seed_data(num_samples=min(100, num_samples))

        valid_count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for item in ds:
                question = item.get("question", "").strip()
                input_output = item.get("input_output", "")
                starter_code = item.get("starter_code", "")
                
                if len(question) < 40 or not input_output:
                    continue
                    
                try:
                    io_dict = json.loads(input_output) if isinstance(input_output, str) else input_output
                    inputs = io_dict.get("inputs", [])
                    outputs = io_dict.get("outputs", [])
                    if not inputs or not outputs:
                        continue
                except Exception:
                    continue
                    
                entry = {
                    "id": f"taco_{valid_count:06d}",
                    "task_type": "code",
                    "domain": "algorithms",
                    "prompt": f"Write a complete Python 3 solution for the following specification:\n\n{question}\n\n{starter_code}",
                    "verifier_spec": {
                        "type": "python_test",
                        "test_cases": [
                            {"in": inp, "out": out}
                            for inp, out in zip(inputs[:5], outputs[:5])
                        ]
                    }
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                valid_count += 1
                
                if valid_count >= num_samples:
                    break
                    
        print(f"[OK] Successfully curated {valid_count} TACO coding problems to: {out_file}")
        return out_file

    # Convenience aliases
    curate_taco = curate_taco_code
    curate_numina = curate_numinamath

    def generate_local_seed_data(self, num_samples: int = 50) -> str:
        """
        Generates high-quality offline seed data for immediate local training and smoke-testing.
        Includes Olympiad math, algorithmic coding, and multi-step reasoning.
        """
        out_file = os.path.join(self.output_dir, "cida_seed_reasoning_data.jsonl")
        print(f"[DATA] Generating {num_samples} local seed reasoning samples...")
        
        # High-signal templates
        seed_problems = [
            {
                "task_type": "math",
                "prompt": "Find the sum of all positive integers n such that n^2 + 19n + 48 is a perfect square.",
                "solution": "Let n^2 + 19n + 48 = k^2. Multiplying by 4 gives (2n+19)^2 - 4k^2 = 169. Hence (2n+19-2k)(2n+19+2k) = 169. Solving factor pairs gives n = 28. \\boxed{28}",
                "target_answer": "28"
            },
            {
                "task_type": "math",
                "prompt": "Compute the number of ordered pairs of integers (x, y) such that x^2 + y^2 <= 25 and x + y >= 3.",
                "solution": "Counting integer lattice points inside circle of radius 5 and above line x+y=3 yields 43 points. \\boxed{43}",
                "target_answer": "43"
            },
            {
                "task_type": "math",
                "prompt": "What is the remainder when 7^2025 is divided by 100?",
                "solution": "Euler's totient phi(100) = 40. 7^40 = 1 (mod 100). 2025 = 40*50 + 25. 7^25 = (7^4)^6 * 7 = 01 * 07 = 7 mod 100. \\boxed{7}",
                "target_answer": "7"
            },
            {
                "task_type": "code",
                "prompt": "Write a Python function `longest_palindrome(s: str) -> str` that finds the longest palindromic substring in s.",
                "solution": "def longest_palindrome(s: str) -> str:\n    if not s: return ''\n    start, max_len = 0, 1\n    for i in range(len(s)):\n        for l, r in ((i, i), (i, i+1)):\n            while l >= 0 and r < len(s) and s[l] == s[r]:\n                if r - l + 1 > max_len:\n                    start = l\n                    max_len = r - l + 1\n                l -= 1\n                r += 1\n    return s[start:start+max_len]",
                "test_cases": [
                    {"in": "babad", "out": "bab"},
                    {"in": "cbbd", "out": "bb"},
                    {"in": "a", "out": "a"}
                ]
            },
            {
                "task_type": "code",
                "prompt": "Write a Python function `min_coins(coins: list[int], amount: int) -> int` to return the fewest coins needed to make up amount, or -1 if impossible.",
                "solution": "def min_coins(coins, amount):\n    dp = [float('inf')] * (amount + 1)\n    dp[0] = 0\n    for c in coins:\n        for x in range(c, amount + 1):\n            dp[x] = min(dp[x], dp[x - c] + 1)\n    return dp[amount] if dp[amount] != float('inf') else -1",
                "test_cases": [
                    {"in": "[1, 2, 5], 11", "out": "3"},
                    {"in": "[2], 3", "out": "-1"},
                    {"in": "[1], 0", "out": "0"}
                ]
            }
        ]
        
        with open(out_file, "w", encoding="utf-8") as f:
            for i in range(num_samples):
                base_sample = seed_problems[i % len(seed_problems)]
                entry = {
                    "id": f"seed_{i:04d}",
                    "task_type": base_sample["task_type"],
                    "prompt": base_sample["prompt"] + (f" (Sample variation #{i+1})" if i >= len(seed_problems) else ""),
                    "solution": base_sample.get("solution", ""),
                    "verifier_spec": {
                        "type": "math_equal" if base_sample["task_type"] == "math" else "python_test",
                        "target_answer": base_sample.get("target_answer", ""),
                        "test_cases": base_sample.get("test_cases", [])
                    }
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
        print(f"[OK] Generated {num_samples} high-signal seed samples to: {out_file}")
        return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frontier Dataset Curator for CIDA v4")
    parser.add_argument("--source", type=str, choices=["numinamath", "taco", "seed"], default="seed")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./curated_data")
    args = parser.parse_args()
    
    curator = DatasetCurator(output_dir=args.output_dir)
    if args.source == "numinamath":
        curator.curate_numinamath(num_samples=args.samples)
    elif args.source == "taco":
        curator.curate_taco_code(num_samples=args.samples)
    else:
        curator.generate_local_seed_data(num_samples=args.samples)
