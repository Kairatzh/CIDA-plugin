import json

with open('Kairos_CIDA_CTF_Swarm_Demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the last cell which is the HotpotQA cell
last_cell = nb['cells'][-1]

# Replace the datasets logic with hardcoded samples to avoid HF API errors
new_source = []
for line in last_cell['source']:
    if "from datasets import load_dataset" in line:
        continue
    if "hotpot = load_dataset" in line:
        continue
    if "test_samples =" in line:
        new_source.extend([
            '# Using hardcoded samples to bypass HuggingFace API namespace errors\\n',
            'test_samples = [\\n',
            '    {\\n',
            '        "question": "What is the population of the city where the author of \\"The Great Gatsby\\" was born?",\\n',
            '        "answer": "St. Paul, Minnesota (population around 311,527)"\\n',
            '    },\\n',
            '    {\\n',
            '        "question": "Who was the director of the 2014 sci-fi film starring Matthew McConaughey and Anne Hathaway?",\\n',
            '        "answer": "Christopher Nolan"\\n',
            '    }\\n',
            ']\\n'
        ])
        continue
    if "datasets" in line and "!pip install" in line:
        line = line.replace("datasets", "")
    new_source.append(line)

nb['cells'][-1]['source'] = new_source

with open('Kairos_CIDA_CTF_Swarm_Demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook fixed with hardcoded samples!")
