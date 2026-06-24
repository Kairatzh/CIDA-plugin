"""CIDA Role Pre-Training v2 — Main training script. Run in Colab with A100."""
import sys, os, time, math, warnings, json
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
os.makedirs('ckpt_role_v2', exist_ok=True)
os.makedirs('results_role_v2', exist_ok=True)
HISTORY = []
D_EMB, D_HIDDEN = 384, 256

print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ── 1. Frozen MiniLM encoder ────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
_minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(DEVICE)
for p in _minilm.parameters(): p.requires_grad = False

@torch.no_grad()
def embed_texts(texts, batch_size=512):
    vecs = []
    for i in range(0, len(texts), batch_size):
        v = _minilm.encode(texts[i:i+batch_size], convert_to_tensor=True,
                           device=DEVICE, show_progress_bar=False)
        vecs.append(v.cpu())
    return torch.cat(vecs)

# ── 2. Import CIDA + v2 modules ─────────────────────────────────────────────
if '__file__' in globals():
    base_dir = os.path.dirname(os.path.abspath(__file__))
else:
    base_dir = os.path.abspath('.')

sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, '..'))
sys.path.insert(0, os.path.join(base_dir, 'cida_role_pretrain_v2'))
for _p in ['/content/CIDA-plugin', '/content/CIDA-plugin/cida_role_pretrain_v2']:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cida_plugin import CIDAPlugin, CIDAPluginConfig
from cida_plugin.agent import RoleNames
from role_identity import RoleIdentityLayer, ContrastiveMessageLoss, RoleTimeDynamics
from model_and_losses import RoleCIDA, RolePreTrainLossV2, UncertaintyCalibrationLossV2, EMAModel

# ── 3. Load datasets (Bypassing HF datasets due to API issues) ──────────────
from sklearn.datasets import fetch_20newsgroups
print('Loading 20newsgroups dataset (bypassing HuggingFace Datasets)...')
raw_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
texts, roles = [], []
for t, r in zip(raw_data.data, raw_data.target):
    clean_t = t.strip()
    if len(clean_t) > 20:
        texts.append(clean_t[:1000])
        roles.append(int(r % 3))
    if len(texts) >= 20000:
        break

# ── 4. Pre-compute embeddings ───────────────────────────────────────────────
print(f'Embedding {len(texts):,} texts using MiniLM...')
arg_embs = embed_texts(texts)
arg_roles = torch.tensor(roles, dtype=torch.long)
cd_embs = arg_embs
cd_roles = arg_roles

class EmbeddingDataset(Dataset):
    def __init__(self, embs, roles): self.embs, self.roles = embs, roles
    def __len__(self): return len(self.roles)
    def __getitem__(self, idx): return {'emb': self.embs[idx], 'role': self.roles[idx]}

def make_loaders(embs, roles, val_ratio=0.05, bs=256):
    n = len(roles); nv = max(500, int(n*val_ratio))
    idx = torch.randperm(n)
    tr = EmbeddingDataset(embs[idx[nv:]], roles[idx[nv:]])
    vl = EmbeddingDataset(embs[idx[:nv]], roles[idx[:nv]])
    kw = dict(batch_size=bs, num_workers=4, pin_memory=True, persistent_workers=True)
    return DataLoader(tr, shuffle=True, **kw), DataLoader(vl, shuffle=False, **kw)

BS = 256 if DEVICE=='cuda' else 64
s1_trl, s1_vll = make_loaders(arg_embs, arg_roles, bs=BS)
s3_trl, s3_vll = make_loaders(cd_embs, cd_roles, bs=BS)

# ── 5. Build model (v2: connected graph) ────────────────────────────────────
cfg = CIDAPluginConfig(
    d_input=D_EMB, d_hidden=D_HIDDEN, d_message=D_HIDDEN,
    num_agents=4, num_classes=3, max_rounds=3, multi_label=False,
    num_attn_heads=8, role_prior_blend=0.40, lambda_cal=0.3, lambda_ac=0.1,
    min_disagreement=0.08, comm_dropout=0.15, use_perspective_projector=True,
    anonymous_messages=True, sparse_comm_k=2, use_dynamic_trust=True,
    trust_ema_gamma=0.9, positive_class_idx=0,
)
model = RoleCIDA(D_EMB, D_HIDDEN, 3, CIDAPlugin, cfg, embed_fn=embed_texts).to(DEVICE)
total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'RoleCIDA v2: {total_p/1e6:.2f}M total | {train_p/1e6:.2f}M trainable')

# ── 6. Gradient flow verification ───────────────────────────────────────────
print('\n=== Gradient Flow Verification ===')
test_x = torch.randn(4, D_EMB, device=DEVICE, requires_grad=True)
out = model(test_x)
loss = out['p_final'].sum()
loss.backward()
anchor_grad = model.role_layer.anchors.grad
print(f'Anchor grad exists: {anchor_grad is not None}')
if anchor_grad is not None:
    print(f'Anchor grad norm: {anchor_grad.norm():.6f}')
    print(f'✓ CONNECTED GRAPH: loss gradients flow into RoleIdentityLayer')
else:
    print('✗ DISCONNECTED: anchors receive no gradients!')
model.zero_grad()

# ── 7. Losses + Optimizer ───────────────────────────────────────────────────
role_loss_fn = RolePreTrainLossV2(n_classes=3)
calib_loss_fn = UncertaintyCalibrationLossV2()
STAGE_EPOCHS = {1: 2, 2: 2, 3: 2}

optimizer = optim.AdamW([
    {'params': model.role_layer.parameters(), 'lr': 5e-4, 'weight_decay': 0.0},
    {'params': model.plugin.parameters(), 'lr': 2e-4, 'weight_decay': 1e-2},
    {'params': model.role_msg.parameters(), 'lr': 3e-4, 'weight_decay': 1e-3},
    {'params': model.msg_contrast.parameters(), 'lr': 3e-4, 'weight_decay': 0.0},
    {'params': model.time_dynamics.parameters(), 'lr': 1e-4, 'weight_decay': 0.0},
], fused=(DEVICE=='cuda'))

total_steps = sum(STAGE_EPOCHS[s]*len(s1_trl if s!=3 else s3_trl) for s in [1,2,3])
scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
    max_lr=[8e-4, 3e-4, 5e-4, 5e-4, 2e-4],
    total_steps=total_steps, pct_start=0.05, anneal_strategy='cos')
ema = EMAModel(model, decay=0.999)

# ── 8. Training functions ───────────────────────────────────────────────────
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
        x, y = batch['emb'].to(DEVICE), batch['role'].to(DEVICE).clamp(0, 2)
        with torch.autocast(device_type='cuda' if DEVICE=='cuda' else 'cpu', dtype=DTYPE, enabled=IS_AMP):
            out = mdl(x)
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
        x = batch['emb'].to(DEVICE, non_blocking=True)
        y = batch['role'].to(DEVICE, non_blocking=True).clamp(0, 2)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda' if DEVICE=='cuda' else 'cpu', dtype=DTYPE, enabled=IS_AMP):
            out = mdl(x)
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

# ── 9. Training loop ────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=3, metric='asi', mode='max'):
        self.patience, self.metric, self.mode = patience, metric, mode
        self.best = float('-inf') if mode=='max' else float('inf')
        self.counter = 0
    def __call__(self, ev):
        val = ev[self.metric]
        better = val > self.best if self.mode=='max' else val < self.best
        if better: self.best = val; self.counter = 0; return False
        self.counter += 1; return self.counter >= self.patience

best_asi = 0.0; global_ep = 0
stage_loaders = {1: s1_trl, 2: s1_trl, 3: s3_trl}
stage_val = {1: s1_vll, 2: s1_vll, 3: s3_vll}

for stage in [1, 2, 3]:
    names = {1:'Role Identity', 2:'Uncertainty Calibration', 3:'Cross-Domain'}
    print(f'\n{"="*60}\n  STAGE {stage}: {names[stage]}\n{"="*60}')
    es = EarlyStopping(patience=3, metric='asi' if stage<3 else 'disagreement', mode='max')
    for ep in range(STAGE_EPOCHS[stage]):
        global_ep += 1
        anneal = ep / max(STAGE_EPOCHS[stage]-1, 1)
        t0 = time.time()
        avg_loss = train_epoch(model, stage_loaders[stage], optimizer, scheduler, stage, anneal)
        # Evaluate with EMA weights
        ema.apply(model)
        ev = evaluate(model, stage_val[stage])
        ema.restore(model)
        dt = time.time() - t0
        HISTORY.append({'stage':stage,'epoch':global_ep,'loss':avg_loss,'time':dt,
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
            torch.save({'plugin': model.plugin.state_dict(),
                        'role_layer': model.role_layer.state_dict(),
                        'role_msg': model.role_msg.state_dict(),
                        'msg_contrast': model.msg_contrast.state_dict(),
                        'time_dynamics': model.time_dynamics.state_dict(),
                        'ema': ema.shadow,
                        'config': model.plugin.config.to_dict(),
                        'history': HISTORY}, 'ckpt_role_v2/best_full.pt')
            print(f'  ✓ Best (ASI={best_asi:.4f})')
        if es(ev):
            print(f'  Early stop'); break

# ── 10. Analysis ─────────────────────────────────────────────────────────────
print(f'\n{"="*60}\nRole Pre-Training v2 Complete! Best ASI: {best_asi:.4f}\n{"="*60}')

# t-SNE of message embeddings
try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    model.eval()
    with torch.no_grad():
        sample = next(iter(s1_vll))
        out = model(sample['emb'].to(DEVICE))
    if out['m_all']:
        m_last = out['m_all'][-1].cpu().numpy()  # (B, M, d)
        B, M, d = m_last.shape
        m_flat = m_last.reshape(B*M, d)
        labels = np.tile(np.arange(M), B)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        coords = tsne.fit_transform(m_flat)
        colors = ['#EF4444','#3B82F6','#F59E0B','#10B981']
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(M):
            mask = labels == i
            ax.scatter(coords[mask,0], coords[mask,1], c=colors[i],
                      label=RoleNames.get(i), alpha=0.6, s=30)
        ax.legend(fontsize=12); ax.set_title('t-SNE: Agent Message Embeddings (should form 4 clusters)')
        plt.savefig('results_role_v2/tsne_messages.png', dpi=150, bbox_inches='tight')
        plt.show()
        print('t-SNE saved to results_role_v2/tsne_messages.png')
except Exception as e:
    print(f'Visualization skipped: {e}')

# Transfer test
print('\n=== Transfer Test (AG News) ===')
model.eval()
with torch.no_grad():
    out_new = model(cd_embs[:500].to(DEVICE))
asi_new = compute_asi(out_new['b_all'])
ev_train = evaluate(model, s1_vll, max_batches=20)
ratio = asi_new / max(ev_train['asi'], 1e-6)
print(f'ASI: Train={ev_train["asi"]:.4f} | Transfer={asi_new:.4f} | Ratio={ratio:.2f}')
print(f'{"✓ TRANSFER SUCCESS" if ratio >= 0.8 else "⚠ Partial transfer"}')
