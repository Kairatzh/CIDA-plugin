"""
CIDA Role Pre-Training v2 — Foundation-Grade Script.
- Encoder: microsoft/deberta-v3-base (768 dimensions) with LoRA PEFT
- Datasets: Direct downloads of official SNLI (Stanford) & MultiNLI (NYU)
- Target: 90%+ accuracy on logic & NLI benchmarks
"""
import sys, os, time, math, warnings, zipfile, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
warnings.filterwarnings('ignore')

# ── Setup ────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IS_AMP = DEVICE == 'cuda'
DTYPE = torch.bfloat16 if (DEVICE=='cuda' and torch.cuda.get_device_capability()[0]>=8) else torch.float16
torch.manual_seed(42); np.random.seed(42)
os.makedirs('ckpt_role_foundation', exist_ok=True)
os.makedirs('results_role_foundation', exist_ok=True)
D_EMB, D_HIDDEN = 768, 512 # Doubled hidden dimension for higher capacity

# ── 1. Robust Downloaders (Bypassing HF datasets API) ────────────────────────
def download_and_extract_nli(data_dir="nli_data"):
    os.makedirs(data_dir, exist_ok=True)
    snli_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    mnli_url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
    
    snli_zip = os.path.join(data_dir, "snli.zip")
    mnli_zip = os.path.join(data_dir, "mnli.zip")
    
    # Download SNLI if not exists
    if not os.path.exists(os.path.join(data_dir, "snli_1.0", "snli_1.0_train.txt")):
        print("Downloading SNLI from Stanford University (90MB)...")
        urllib.request.urlretrieve(snli_url, snli_zip)
        with zipfile.ZipFile(snli_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(snli_zip)
        
    # Download MultiNLI if not exists
    if not os.path.exists(os.path.join(data_dir, "multinli_1.0", "multinli_1.0_train.txt")):
        print("Downloading MultiNLI from NYU (220MB)...")
        urllib.request.urlretrieve(mnli_url, mnli_zip)
        with zipfile.ZipFile(mnli_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(mnli_zip)

def parse_nli_file(file_path, max_samples=150000):
    texts, labels = [], []
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        s1_idx = header.index("sentence1")
        s2_idx = header.index("sentence2")
        lbl_idx = header.index("gold_label")
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) <= max(s1_idx, s2_idx, lbl_idx):
                continue
            lbl = parts[lbl_idx]
            if lbl in label_map:
                s1, s2 = parts[s1_idx], parts[s2_idx]
                if len(s1.strip()) > 5 and len(s2.strip()) > 5:
                    texts.append(f"{s1} | {s2}")
                    labels.append(label_map[lbl])
            if len(texts) >= max_samples:
                break
    return texts, labels

# ── 2. DeBERTa-v3-base Encoder with LoRA PEFT ────────────────────────────────
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

print("Initializing DeBERTa-v3-base model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
raw_encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")

# Apply LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_proj", "value_proj"],
    lora_dropout=0.05,
    bias="none"
)
encoder = get_peft_model(raw_encoder, peft_config)
print("LoRA PEFT applied to DeBERTa model:")
encoder.print_trainable_parameters()
encoder.to(DEVICE)

# ── 3. Import CIDA + v2 modules ─────────────────────────────────────────────
if '__file__' in globals():
    base_dir = os.path.dirname(os.path.abspath(__file__))
else:
    base_dir = os.path.abspath('.')
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, '..'))
sys.path.insert(0, os.path.join(base_dir, 'cida_role_pretrain_v2'))

from cida_plugin import CIDAPlugin, CIDAPluginConfig
from cida_plugin.agent import RoleNames
from role_identity import RoleIdentityLayer, ContrastiveMessageLoss, RoleTimeDynamics
from model_and_losses import RoleCIDA, RolePreTrainLossV2, UncertaintyCalibrationLossV2, EMAModel

# Override RoleCIDA's forward pass to accept tokenized inputs and run encoder dynamically
original_init = RoleCIDA.__init__
def new_init(self, d_emb, d_hidden, num_classes, plugin_cls, plugin_cfg, encoder_model=None, embed_fn=None):
    original_init(self, d_emb, d_hidden, num_classes, plugin_cls, plugin_cfg, embed_fn)
    self.encoder = encoder_model

def new_forward(self, input_ids, attention_mask=None):
    if attention_mask is None:
        # Fallback if raw embeddings are passed (e.g. during gradient verification code)
        x = input_ids
    else:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        mask_exp = attention_mask.unsqueeze(-1)
        x = outputs.last_hidden_state * mask_exp
        x = x.sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        
    s_init = self.role_layer(x)
    out = self._forward_with_role_init(x, s_init)
    
    out['anchor_div_loss'] = self.role_layer.anchor_diversity_loss()
    out['msg_contrast_loss'] = self.msg_contrast(out.get('m_all', []))
    out['tau'] = self.time_dynamics(x[:, :self.d_hidden]
                                     if x.size(-1) >= self.d_hidden
                                     else self.role_layer.input_proj(x))
    out['anchor_sim'] = self.role_layer.anchor_similarity_matrix()
    return out

RoleCIDA.__init__ = new_init
RoleCIDA.forward = new_forward

# ── 4. Main script execution ─────────────────────────────────────────────────
def main():
    download_and_extract_nli()
    print("Parsing NLI files...")
    snli_texts, snli_labels = parse_nli_file("nli_data/snli_1.0/snli_1.0_train.txt", max_samples=150000)
    mnli_texts, mnli_labels = parse_nli_file("nli_data/multinli_1.0/multinli_1.0_train.txt", max_samples=150000)
    
    all_texts = snli_texts + mnli_texts
    all_labels = snli_labels + mnli_labels
    print(f"Total texts parsed: {len(all_texts):,}")
    
    # Pre-tokenize all texts for extreme speed
    print("Pre-tokenizing texts...")
    t0 = time.time()
    encodings = tokenizer(all_texts, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    roles = torch.tensor(all_labels, dtype=torch.long)
    print(f"Tokenization complete in {time.time()-t0:.1f}s.")
    
    # Make Datasets and Loaders
    class TokenizedDataset(Dataset):
        def __init__(self, ids, masks, roles):
            self.ids = ids
            self.masks = masks
            self.roles = roles
        def __len__(self): return len(self.roles)
        def __getitem__(self, idx):
            return {
                'input_ids': self.ids[idx],
                'attention_mask': self.masks[idx],
                'role': self.roles[idx]
            }

    def make_loaders(ids, masks, roles, val_ratio=0.05, bs=128):
        n = len(roles); nv = max(1000, int(n*val_ratio))
        idx = torch.randperm(n)
        tr = TokenizedDataset(ids[idx[nv:]], masks[idx[nv:]], roles[idx[nv:]])
        vl = TokenizedDataset(ids[idx[:nv]], masks[idx[:nv]], roles[idx[:nv]])
        kw = dict(batch_size=bs, num_workers=4, pin_memory=True, persistent_workers=True)
        return DataLoader(tr, shuffle=True, **kw), DataLoader(vl, shuffle=False, **kw)
        
    s1_trl, s1_vll = make_loaders(input_ids, attention_masks, roles, bs=128)
    
    # Configure CIDA with DeBERTa input size (768) and increased rounds (4)
    cfg = CIDAPluginConfig(
        d_input=D_EMB, d_hidden=D_HIDDEN, d_message=D_HIDDEN,
        num_agents=4, num_classes=3, max_rounds=4, multi_label=False,
        num_attn_heads=8, role_prior_blend=0.40, lambda_cal=0.3, lambda_ac=0.1,
        min_disagreement=0.08, comm_dropout=0.15, use_perspective_projector=True,
        anonymous_messages=True, sparse_comm_k=2, use_dynamic_trust=True,
        trust_ema_gamma=0.9,
    )
    
    # Load model with LoRA encoder embedded inside
    model = RoleCIDA(D_EMB, D_HIDDEN, 3, CIDAPlugin, cfg, encoder_model=encoder).to(DEVICE)
    role_loss_fn = RolePreTrainLossV2(n_classes=3)
    calib_loss_fn = UncertaintyCalibrationLossV2()
    
    # Setup Optimizer to train both CIDA and LoRA weights
    lora_params = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = optim.AdamW([
        {'params': model.role_layer.parameters(), 'lr': 5e-4, 'weight_decay': 0.0},
        {'params': model.plugin.parameters(), 'lr': 2e-4, 'weight_decay': 1e-2},
        {'params': model.role_msg.parameters(), 'lr': 3e-4, 'weight_decay': 1e-3},
        {'params': model.msg_contrast.parameters(), 'lr': 3e-4, 'weight_decay': 0.0},
        {'params': model.time_dynamics.parameters(), 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': lora_params, 'lr': 8e-5, 'weight_decay': 1e-2}, # LoRA learning rate
    ], fused=(DEVICE=='cuda'))
    
    STAGE_EPOCHS = {1: 4, 2: 3, 3: 3}
    total_steps = sum(STAGE_EPOCHS[s]*len(s1_trl) for s in [1, 2, 3])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=[8e-4, 3e-4, 5e-4, 5e-4, 2e-4, 1.5e-4],
        total_steps=total_steps, pct_start=0.05, anneal_strategy='cos')
    ema = EMAModel(model, decay=0.999)
    
    def compute_asi(b_all):
        if not b_all: return 0.0
        b = b_all[-1]
        ents = [-(b[:,ag,:].clamp(1e-7,1)*b[:,ag,:].clamp(1e-7,1).log()).sum(1).mean().item()
                for ag in range(b.shape[1])]
        return float(np.std(ents))

    @torch.no_grad()
    def evaluate(mdl, loader, max_batches=80):
        mdl.eval()
        all_acc, all_dis, all_asi = [], [], []
        agent_conf, agent_ent = defaultdict(list), defaultdict(list)
        for i, batch in enumerate(loader):
            if i >= max_batches: break
            ids, masks = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE)
            y = batch['role'].to(DEVICE).clamp(0, 2)
            with torch.autocast(device_type='cuda' if DEVICE=='cuda' else 'cpu', dtype=DTYPE, enabled=IS_AMP):
                out = mdl(ids, masks)
            preds = out['p_final'][:, :3].argmax(1)
            all_acc.append((preds == y).float().mean().item())
            all_dis.append(out['disagreement'].mean().item())
            all_asi.append(compute_asi(out['b_all']))
            if out['b_all']:
                b = out['b_all'][-1]
                for ag in range(b.shape[1]):
                    bi = b[:, ag, :].clamp(1e-7, 1)
                    agent_conf[ag].append(bi.max(1).values.mean().item())
                    agent_ent[ag].append(-(bi*bi.log()).sum(1).mean().item())
        agents = {ag: {'conf': np.mean(agent_conf[ag]), 'ent': np.mean(agent_ent[ag])}
                  for ag in sorted(agent_conf)}
        return {'acc': np.mean(all_acc), 'disagreement': np.mean(all_dis),
                'asi': np.mean(all_asi), 'agents': agents}

    def train_epoch(mdl, loader, opt, sched, stage, anneal):
        mdl.train()
        total_loss, n = 0.0, 0
        scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE==torch.float16))
        for batch in loader:
            ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            masks = batch['attention_mask'].to(DEVICE, non_blocking=True)
            y = batch['role'].to(DEVICE, non_blocking=True).clamp(0, 2)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda' if DEVICE=='cuda' else 'cpu', dtype=DTYPE, enabled=IS_AMP):
                out = mdl(ids, masks)
                if stage in [1, 3]:
                    loss, info = role_loss_fn(out['b_all'], y,
                        out['anchor_div_loss'], out['msg_contrast_loss'], stage, anneal)
                else:
                    loss_r, _ = role_loss_fn(out['b_all'], y,
                        out['anchor_div_loss'], out['msg_contrast_loss'], 1, anneal)
                    loss_c, _ = calib_loss_fn(out['b_all'], y)
                    loss = 0.5 * loss_r + 0.5 * loss_c
                if out.get('L_orth') is not None:
                    loss = loss + 0.02 * out['L_orth']
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update(mdl)
            total_loss += loss.item(); n += 1
        return total_loss / max(n, 1)
        
    best_asi = 0.0
    global_ep = 0
    history = []
    
    for stage in [1, 2, 3]:
        names = {1:'Role Identity', 2:'Uncertainty Calibration', 3:'Cross-Domain'}
        print(f'\n{"="*60}\n  STAGE {stage}: {names[stage]}\n{"="*60}')
        for ep in range(STAGE_EPOCHS[stage]):
            global_ep += 1
            anneal = ep / max(STAGE_EPOCHS[stage]-1, 1)
            t0 = time.time()
            avg_loss = train_epoch(model, s1_trl, optimizer, scheduler, stage, anneal)
            
            # Evaluate
            ema.apply(model)
            ev = evaluate(model, s1_vll)
            ema.restore(model)
            
            dt = time.time() - t0
            history.append({'stage':stage,'epoch':global_ep,'loss':avg_loss,'time':dt,
                            **{k:v for k,v in ev.items() if k!='agents'}})
            
            print(f'\nEp {global_ep}/{sum(STAGE_EPOCHS.values())} [{dt:.0f}s] '
                  f'loss={avg_loss:.4f} acc={ev["acc"]:.4f} '
                  f'dis={ev["disagreement"]:.4f} ASI={ev["asi"]:.4f}')
            
            for ag, s in ev['agents'].items():
                print(f'  {RoleNames.get(ag):<12}: conf={s["conf"]:.3f} ent={s["ent"]:.3f}')
            
            sim = model.role_layer.anchor_similarity_matrix()
            off = sim[~torch.eye(4, dtype=bool)].abs().mean().item()
            print(f'  Anchor off-diag: {off:.4f}')
            
            if ev['asi'] > best_asi:
                best_asi = ev['asi']
                torch.save({
                    'plugin': model.plugin.state_dict(),
                    'role_layer': model.role_layer.state_dict(),
                    'role_msg': model.role_msg.state_dict(),
                    'msg_contrast': model.msg_contrast.state_dict(),
                    'time_dynamics': model.time_dynamics.state_dict(),
                    'ema_shadow': ema.shadow,
                    'config': model.plugin.config.to_dict(),
                }, 'ckpt_role_foundation/best_foundation.pt')
                print(f'  ✓ Best Saved (ASI={best_asi:.4f})')
                
    print("\nPre-training Complete! Checkpoint saved to: ckpt_role_foundation/best_foundation.pt")

# ── 5. Production Inference Engine ──────────────────────────────────────────
class CIDAInferenceEngine:
    """
    Production-ready wrapper for CIDA role-conditioned deliberation.
    Accepts raw strings, runs DeBERTa encoder, and outputs consensus,
    disagreement, and agent thought-process trace.
    """
    def __init__(self, checkpoint_path, device=DEVICE):
        self.device = device
        # Load config and architecture
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg_dict = ckpt['config']
        cfg = CIDAPluginConfig(**cfg_dict)
        
        # Load raw encoder & apply LoRA config exactly as training
        raw_encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        lora_encoder = get_peft_model(raw_encoder, peft_config)
        
        self.model = RoleCIDA(d_emb=cfg.d_input, d_hidden=cfg.d_hidden, 
                              num_classes=cfg.num_classes, plugin_cls=CIDAPlugin, 
                              plugin_cfg=cfg, encoder_model=lora_encoder).to(device)
        self.model.load_state_dict(ckpt['ema_shadow'] if 'ema_shadow' in ckpt else ckpt['plugin'])
        self.model.eval()
        
        # Load DeBERTa tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
            
    @torch.no_grad()
    def analyze(self, texts):
        """
        Analyze a list of texts.
        Returns detailed consensus metrics and agent beliefs per round.
        """
        # Embed dynamically using the embedded LoRA encoder
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        out = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        # Format results
        results = []
        for idx in range(len(texts)):
            trace = []
            for r_idx in range(len(out['b_all'])):
                beliefs = out['b_all'][r_idx][idx].cpu().tolist()
                trace.append({
                    "round": r_idx + 1,
                    "prosecutor_belief": beliefs[0],
                    "defender_belief": beliefs[1],
                    "skeptic_belief": beliefs[2],
                    "integrator_belief": beliefs[3]
                })
            
            results.append({
                "text": texts[idx],
                "final_prediction": out['p_final'][idx].cpu().tolist(),
                "predicted_class": int(out['p_final'][idx].argmax().item()),
                "uncertainty": float(out['uncertainty'][idx].item()),
                "disagreement": float(out['disagreement'][idx].item()),
                "deliberation_trace": trace
            })
        return results

if __name__ == '__main__':
    main()
