import json
with open("Kairos_CIDA_v4_RL_Training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for c in nb["cells"]:
    if c["id"] == "c6":
        with open("c6_source.txt", "w", encoding="utf-8") as out:
            out.write("".join(c["source"]))
        print("Written to c6_source.txt")
