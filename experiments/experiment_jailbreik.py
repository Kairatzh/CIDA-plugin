"""
experiment_jailbreak_fixed.py — Исправление role priors и orth_loss.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Tuple, Optional

from cida_plugin.config import CIDAPluginConfig
from cida_plugin.core import CIDAPlugin
from cida_plugin.losses import CIDALoss
from cida_plugin.agent import AgentState, apply_role_prior, compute_role_orthogonality_loss


# =====================================================================
# 1. ИСПРАВЛЕННЫЙ CIDA PLUGIN (с принудительным применением role priors)
# =====================================================================
class FixedCIDAPlugin(CIDAPlugin):
    """
    Переопределяем forward, чтобы apply_role_prior применялся на каждом раунде.
    """
    def forward(self, pooled, seq_output=None, mask=None):
        cfg = self.config
        B, M = pooled.size(0), cfg.num_agents

        h_cls = self.input_proj(pooled)
        if seq_output is not None:
            H = self.seq_proj(seq_output)
        else:
            H = None

        # Инициализация состояний
        if self.perspective_projector is not None:
            s_t = self.perspective_projector(pooled)
        else:
            s_t = h_cls.unsqueeze(1).expand(B, M, cfg.d_hidden).clone()

        seq_len = H.size(1) if H is not None else 1
        e_init = torch.zeros(B, M, cfg.d_hidden, device=pooled.device)
        r_init = torch.zeros_like(e_init)

        if cfg.multi_label:
            b_init = torch.full((B, M, cfg.num_classes), 0.5, device=pooled.device)
        else:
            b_init = torch.zeros(B, M, cfg.num_classes, device=pooled.device)

        state_t = AgentState(
            s=s_t, b=b_init, u=torch.ones(B, M, 1, device=pooled.device),
            p=torch.zeros(B, M, seq_len, device=pooled.device), e=e_init,
            alpha=torch.ones(B, M, cfg.num_classes, device=pooled.device)
        )

        # Начальный update
        state_t = self.updater(state_t, r_init, e_init)

        # -- ПРИМЕНЯЕМ ROLE PRIORS К НАЧАЛЬНЫМ УБЕЖДЕНИЯМ --
        beliefs = []
        for i in range(M):
            b_i = apply_role_prior(
                state_t.b[:, i, :].clone(),
                role_idx=i,
                multi_label=cfg.multi_label,
                positive_class_idx=cfg.positive_class_idx,
            )
            beliefs.append(b_i)
        state_t.b = torch.stack(beliefs, dim=1)

        # Deliberation loop
        b_all, u_all = [], []
        for t in range(cfg.max_rounds):
            # Evidence
            if cfg.abl_no_pointers or H is None:
                p = torch.ones(B, M, seq_len, device=pooled.device) / seq_len
                e_t = state_t.s
            else:
                p, e_t = self.evidence_extractor(state_t.s, H, mask)
            state_t.p, state_t.e = p, e_t

            # Message
            if cfg.abl_no_messages:
                m_t = torch.zeros(B, M, cfg.d_message, device=pooled.device)
            else:
                m_t = self.message_formulator(state_t.s, state_t.e, state_t.b, state_t.u)

            # Communication
            if cfg.abl_no_communication:
                r_t = torch.zeros_like(state_t.s)
            else:
                r_t = self.communication(state_t.s, m_t, state_t.b, state_t.e)
                gate = self.comm_gate(state_t.s)
                r_t = gate * r_t
                if self.training and cfg.comm_dropout > 0:
                    r_t = F.dropout(r_t, p=cfg.comm_dropout, training=True)

            # Update
            state_t = self.updater(state_t, r_t, e_t)

            # -- ПРИМЕНЯЕМ ROLE PRIORS ПОСЛЕ КАЖДОГО РАУНДА --
            beliefs = []
            for i in range(M):
                b_i = apply_role_prior(
                    state_t.b[:, i, :].clone(),
                    role_idx=i,
                    multi_label=cfg.multi_label,
                    positive_class_idx=cfg.positive_class_idx,
                )
                beliefs.append(b_i)
            state_t.b = torch.stack(beliefs, dim=1)

            b_all.append(state_t.b)
            u_all.append(state_t.u)

            # Early stopping
            if not self.training and cfg.early_stop_threshold is not None:
                tmp_p, _, tmp_d = self.consensus(state_t.b)
                if self.halter.should_halt(tmp_d).all():
                    break
                if tmp_p.max(dim=-1).values.mean().item() >= cfg.early_stop_threshold:
                    break

        # Final consensus
        p_final, uncertainty, disagreement = self.consensus(state_t.b, state_t.u)

        # Orthogonality loss (исправленная версия)
        L_orth = compute_role_orthogonality_loss(state_t.s)

        return {
            "p_final": p_final,
            "b_all": b_all,
            "u_all": u_all,
            "uncertainty": uncertainty,
            "disagreement": disagreement,
            "L_orth": L_orth,
            "state_final": state_t,
            "rounds_used": len(b_all),
        }


# =====================================================================
# 2. АРХИТЕКТУРА ЩИТА (использует FixedCIDAPlugin)
# =====================================================================
class CIDAShieldPipeline(nn.Module):
    def __init__(self, cida_config: CIDAPluginConfig):
        super().__init__()
        self.cfg = cida_config
        self.cida = FixedCIDAPlugin(cida_config)  # ← исправленный плагин

        total_hidden_dim = cida_config.num_agents * cida_config.d_hidden
        self.purification_head = nn.Sequential(
            nn.Linear(total_hidden_dim, cida_config.d_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(cida_config.d_hidden, cida_config.d_input)
        )

    def forward(self, pooled, seq_output=None, mask=None):
        cida_out = self.cida(pooled=pooled, seq_output=seq_output, mask=mask)
        s_final = cida_out["state_final"].s
        B, M, d_hid = s_final.shape
        flat_states = s_final.view(B, M * d_hid)
        purified_emb = self.purification_head(flat_states)
        return {
            "purified_emb": purified_emb,
            "p_final": cida_out["p_final"],
            "b_all": cida_out["b_all"],
            "disagreement": cida_out["disagreement"],
            "L_orth": cida_out["L_orth"],
        }


# =====================================================================
# 3. ДАТАСЕТ
# =====================================================================
class RealJailbreakDataset(Dataset):
    def __init__(self, encoder_name="sentence-transformers/all-mpnet-base-v2", max_samples=400, max_length=256, device="cpu"):
        print("Loading dataset 'deepset/prompt-injections'...")
        raw_ds = load_dataset("deepset/prompt-injections", split="train")
        self.clean_texts = [row["text"] for row in raw_ds if row["label"] == 0][:max_samples]
        self.attack_texts = [row["text"] for row in raw_ds if row["label"] == 1][:max_samples]
        min_len = min(len(self.clean_texts), len(self.attack_texts))
        self.clean_texts = self.clean_texts[:min_len]
        self.attack_texts = self.attack_texts[:min_len]
        print(f"Collected {min_len} pairs [Clean <-> Attack]")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name).to(device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.max_length = max_length

    def __len__(self):
        return len(self.clean_texts)

    @torch.no_grad()
    def __getitem__(self, idx):
        def process(text):
            inputs = self.tokenizer(text, return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.encoder(**inputs)
            emb = out.last_hidden_state[:, 0, :].squeeze(0).cpu()
            seq = out.last_hidden_state.squeeze(0).cpu()
            mask = inputs["attention_mask"].squeeze(0).cpu()
            return emb, seq, mask
        emb_c, seq_c, mask_c = process(self.clean_texts[idx])
        emb_a, seq_a, mask_a = process(self.attack_texts[idx])
        return emb_c, seq_c, mask_c, emb_a, seq_a, mask_a


def collate_fn(batch):
    emb_c = torch.stack([b[0] for b in batch])
    seq_c = torch.stack([b[1] for b in batch])
    mask_c = torch.stack([b[2] for b in batch])
    emb_a = torch.stack([b[3] for b in batch])
    seq_a = torch.stack([b[4] for b in batch])
    mask_a = torch.stack([b[5] for b in batch])
    return emb_c, seq_c, mask_c, emb_a, seq_a, mask_a


# =====================================================================
# 4. ФУНКЦИЯ ПОТЕРЬ
# =====================================================================
class CIDAShieldLoss(nn.Module):
    def __init__(self, base_loss, lambda_purify=1.2, lambda_friction=15.0, margin=0.25):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_purify = lambda_purify
        self.lambda_friction = lambda_friction
        self.margin = margin

    def _trajectory_disagreement(self, b_all):
        if not b_all:
            return torch.tensor(0.0)
        dis = []
        for b_step in b_all:
            dis.append(b_step.std(dim=1).mean(dim=-1))
        return torch.stack(dis, dim=0).mean(dim=0).mean()

    def forward(self, out_c, out_a, emb_clean_true):
        device = emb_clean_true.device
        B = emb_clean_true.size(0)
        labels_c = torch.zeros(B, dtype=torch.long, device=device)
        labels_a = torch.ones(B, dtype=torch.long, device=device)

        loss_c, _ = self.base_loss(out_c["p_final"], labels_c, out_c["b_all"])
        loss_a, _ = self.base_loss(out_a["p_final"], labels_a, out_a["b_all"])
        l_cida = loss_c + loss_a

        l_purify = F.mse_loss(out_c["purified_emb"], emb_clean_true) + \
                   F.mse_loss(out_a["purified_emb"], emb_clean_true)

        fric_c = self._trajectory_disagreement(out_c["b_all"])
        fric_a = self._trajectory_disagreement(out_a["b_all"])
        
        # --- АСИММЕТРИЧНЫЙ ЛОСС ---
        # 1. На чистых: жёстко требуем низкого disagreement (через relu с верхней границей)
        #    Вместо fric_c^2 используем hinge: penalty if fric_c > 0.05
        target_clean = 0.05
        l_fric_clean = F.relu(fric_c - target_clean) ** 2
        
        # 2. На атаках: требуем, чтобы разница (fric_a - fric_c) была > margin
        #    Если fric_a - fric_c < margin → штраф
        diff = fric_a - fric_c
        l_fric_attack = F.relu(self.margin - diff) ** 2
        
        # 3. Бонус: если fric_a < 0.15, тоже штрафуем (даже если разница большая)
        min_attack_disagreement = 0.15
        l_min_attack = F.relu(min_attack_disagreement - fric_a) ** 2 * 0.5
        
        l_friction = l_fric_clean + self.lambda_friction * l_fric_attack + l_min_attack

        l_orth = out_c["L_orth"] + out_a["L_orth"]

        total = l_cida + self.lambda_purify * l_purify + l_friction + 0.05 * l_orth
        return total, {
            "l_cida": l_cida.item(),
            "l_purify": l_purify.item(),
            "l_friction": l_friction.item(),
            "l_orth": l_orth.item(),
            "fric_c": fric_c.item(),
            "fric_a": fric_a.item(),
            "l_fric_clean": l_fric_clean.item(),
            "l_fric_attack": l_fric_attack.item(),
        }

# =====================================================================
# 5. ЗАПУСК ОБУЧЕНИЯ
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = CIDAPluginConfig(
        d_input=768, d_hidden=256, d_message=64, num_agents=4, num_classes=2,
        max_rounds=5, multi_label=False,
        lambda_cal=0.4,
        lambda_ac=0.1,          # было 0.3 → уменьшили для снижения f_c
        min_disagreement=0.05,  # было 0.08 → снизили порог
        comm_dropout=0.1,
        abl_no_pointers=False,
        abl_no_messages=False,
        abl_no_communication=False,
        use_perspective_projector=False,
    )
    model = CIDAShieldPipeline(cfg).to(device)
    base_loss = CIDALoss(lambda_cal=0.4, lambda_ac=0.1, min_disagreement=0.05, multi_label=False)
    shield_loss = CIDAShieldLoss(
    base_loss,
    lambda_purify=1.2,
    lambda_friction=15.0,   # было 12.0 → увеличили
    margin=0.25             # было 0.15 → увеличили
)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    dataset = RealJailbreakDataset(max_samples=400, device=device)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)  # smaller batch for CPU

    print("\nStarting training...")
    for epoch in range(15):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader):
            emb_c, seq_c, mask_c, emb_a, seq_a, mask_a = [x.to(device) for x in batch]
            out_c = model(emb_c, seq_c, mask_c)
            out_a = model(emb_a, seq_a, mask_a)
            loss, comp = shield_loss(out_c, out_a, emb_c)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if step % 3 == 0:
                print(f"E{epoch+1:2d} S{step:3d} | Loss={loss.item():.4f} | "
                      f"CIDA={comp['l_cida']:.3f} Purify={comp['l_purify']:.3f} "
                      f"Fric={comp['l_friction']:.3f} | f_c={comp['fric_c']:.4f} f_a={comp['fric_a']:.4f}")
        print(f"Epoch {epoch+1} avg loss = {total_loss/len(loader):.4f}\n")

    os.makedirs("saved_shield_fixed", exist_ok=True)
    model.cida.save_pretrained("saved_shield_fixed/cida_core")
    torch.save(model.state_dict(), "saved_shield_fixed/full_pipeline.bin")
    print("Saved to saved_shield_fixed/")


if __name__ == "__main__":
    main()