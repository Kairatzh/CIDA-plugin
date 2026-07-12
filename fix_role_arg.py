import json
import re

with open('Kairos_CIDA_CTF_Swarm_Demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Use regex to remove role="..." including multiline ones
            # For simplicity, if we see role=, we can just skip the line if it's in a single line
            if line.strip().startswith('role="'):
                continue
            if line.strip().startswith('role="""') or 'role=' in line:
                # Naive replace for simple lines
                new_line = re.sub(r'\s*role=.*?,', '', line)
                if new_line != line:
                    line = new_line
            # If the role is multiline like role="""...
            if 'role="""' in line:
                line = ''
            new_source.append(line)
        
        # We need a robust way to delete the multiline role argument
        # Let's just do a string replace on the whole cell source
        full_source = "".join(cell['source'])
        
        # Regex to remove role="...", role='...', role="""..."""
        # It handles multiline by using DOTALL
        full_source = re.sub(r'role\s*=\s*(["\']{3}[\s\S]*?["\']{3}|["\'].*?["\'])\s*,?\s*', '', full_source)
        
        cell['source'] = [line + "\n" if i < len(full_source.split('\n'))-1 else line for i, line in enumerate(full_source.split('\n'))]

with open('Kairos_CIDA_CTF_Swarm_Demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Removed 'role' arguments from LatentAgent instantiations!")
