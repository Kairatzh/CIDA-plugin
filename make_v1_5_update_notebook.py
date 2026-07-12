import nbformat as nbf

def create_update_notebook():
    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# 🚀 Upgrade Kairos CIDA to v1.5 on Hugging Face\n\n"
        "This notebook downloads the baseline `Kairos-7B-CIDA-v1.0` adapter from Hugging Face, "
        "merges it with the fine-tuned v1.5 agentic weights from your Google Drive, "
        "and pushes the fully upgraded `v1.5` model back to the Hugging Face Hub."
    ))
    
    nb.cells.append(nbf.v4.new_code_cell(
        "!pip install -q transformers bitsandbytes accelerate huggingface_hub\n"
        "!pip install -q git+https://github.com/Kairatzh/CIDA-plugin.git"
    ))
    
    nb.cells.append(nbf.v4.new_markdown_cell(
        "### 1. Mount Google Drive\n"
        "We need to access the locally fine-tuned v1.5 Opus weights."
    ))
    
    nb.cells.append(nbf.v4.new_code_cell(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')"
    ))
    
    nb.cells.append(nbf.v4.new_markdown_cell(
        "### 2. Login to Hugging Face\n"
        "You need a write-access token to push the updated model."
    ))
    
    nb.cells.append(nbf.v4.new_code_cell(
        "from huggingface_hub import notebook_login\n"
        "notebook_login()"
    ))
    
    nb.cells.append(nbf.v4.new_markdown_cell(
        "### 3. Load Base Model and CIDA v1.0\n"
        "We initialize the wrapper using the same pattern tested in `run_cloud_operations_benchmark.py`."
    ))
    
    nb.cells.append(nbf.v4.new_code_cell(
        "import torch\n"
        "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n"
        "from cida_plugin.inference.loader import CIDAWrapper\n"
        "import os\n\n"
        "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n\n"
        "# Load base Qwen model\n"
        "bnb_config = BitsAndBytesConfig(\n"
        "    load_in_4bit=True,\n"
        "    bnb_4bit_compute_dtype=torch.float16,\n"
        ")\n"
        "print('Loading Qwen 1.5 7B...')\n"
        "llm = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-7B-Chat', quantization_config=bnb_config, device_map='auto')\n\n"
        "print('Loading baseline CIDA v1.0 weights from Hub...')\n"
        "wrapper = CIDAWrapper.from_pretrained(\n"
        "    base_model=llm,\n"
        "    cida_repo='kirtzh/Kairos-7B-CIDA-v1.0'\n"
        ")\n"
        "print('CIDA v1.0 successfully loaded.')"
    ))
    
    nb.cells.append(nbf.v4.new_markdown_cell(
        "### 4. Merge Fine-Tuned v1.5 Weights\n"
        "Now we apply the fine-tuned gating and latent parameters."
    ))
    
    nb.cells.append(nbf.v4.new_code_cell(
        "ft_path = '/content/drive/MyDrive/Kairos_CIDA_v1.5_FineTuned_Opus/pytorch_model.bin'\n"
        "if os.path.exists(ft_path):\n"
        "    print(f'Loading fine-tuned v1.5 weights from {ft_path}...')\n"
        "    state_dict = torch.load(ft_path, map_location='cpu')\n"
        "    # strict=False allows loading partial gating updates on top of v1.0\n"
        "    wrapper.load_state_dict(state_dict, strict=False)\n"
        "    print('✅ v1.5 weights successfully merged into the CIDA Wrapper.')\n"
        "else:\n"
        "    print('❌ Error: Fine-tuned weights not found in Google Drive.')"
    ))
    
    nb.cells.append(nbf.v4.new_markdown_cell(
        "### 5. Push v1.5 to Hugging Face Hub\n"
        "We push the updated `cida_core` weights, the `base_projector`, and the config back to the Hub."
    ))
    
    nb.cells.append(nbf.v4.new_code_cell(
        "from huggingface_hub import HfApi\n"
        "api = HfApi()\n\n"
        "REPO_ID = 'kirtzh/Kairos-7B-CIDA-v1.5' # Or your target repo\n\n"
        "print(f'Creating/Updating repository: {REPO_ID}')\n"
        "api.create_repo(repo_id=REPO_ID, exist_ok=True)\n\n"
        "# Save updated components locally first\n"
        "os.makedirs('v1_5_export', exist_ok=True)\n\n"
        "print('Saving updated CIDA core...')\n"
        "torch.save(wrapper.cida_core.state_dict(), 'v1_5_export/pytorch_model.bin')\n\n"
        "print('Saving updated Plan Projector...')\n"
        "torch.save(wrapper.base_projector.state_dict(), 'v1_5_export/plan_projector.pt')\n\n"
        "print('Saving configuration...')\n"
        "wrapper.cida_core.config.save_pretrained('v1_5_export')\n\n"
        "# Upload to Hub\n"
        "print('Uploading to Hugging Face...')\n"
        "api.upload_folder(\n"
        "    folder_path='v1_5_export',\n"
        "    repo_id=REPO_ID,\n"
        "    commit_message='🚀 Update CIDA adapters to v1.5 (Latent Agentic Fine-tuning)'\n"
        ")\n"
        "print('✅ Upgrade complete! Your v1.5 agentic model is now live on Hugging Face.')"
    ))
    
    with open("Update_CIDA_to_v1_5.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print("Created Update_CIDA_to_v1_5.ipynb successfully.")

if __name__ == '__main__':
    create_update_notebook()
