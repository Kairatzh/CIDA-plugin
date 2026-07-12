import json

with open('Kairos_CIDA_CTF_Swarm_Demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Replace tools=[execute_bash] with tools={"execute_bash": execute_bash}
        if "tools=[execute_bash]" in source:
            source = source.replace("tools=[execute_bash]", 'tools={"execute_bash": execute_bash}')
            
        # Replace tools=[search_wikipedia] with tools={"search_wikipedia": search_wikipedia}
        if "tools=[search_wikipedia]" in source:
            source = source.replace("tools=[search_wikipedia]", 'tools={"search_wikipedia": search_wikipedia}')
            
        cell['source'] = [line + "\\n" if i < len(source.split('\\n'))-1 else line for i, line in enumerate(source.split('\\n'))]

with open('Kairos_CIDA_CTF_Swarm_Demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed tools arguments to be dictionaries instead of lists!")
