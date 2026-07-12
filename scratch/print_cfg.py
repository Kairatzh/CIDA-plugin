import os
from huggingface_hub import hf_hub_download
from cida_plugin.config import CIDAPluginConfig

hf_repo = "kirtzh/Kairos-7B-CIDA-v1.0"
cfg_path = hf_hub_download(repo_id=hf_repo, filename="config.json")
cfg = CIDAPluginConfig.from_pretrained(os.path.dirname(cfg_path))
print("Config dict:")
for k, v in cfg.to_dict().items():
    print(f"  {k}: {v}")
