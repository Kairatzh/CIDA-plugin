import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from cida_plugin import CIDAPlugin, CIDAPluginConfig, CIDALoss
from cida_latent_plugin import CIDAPlugin as LatentCIDAPlugin, CIDAPluginConfig as LatentConfig

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set low role prior blend to allow higher consensus confidence
import cida_plugin.agent
import cida_latent_plugin.agent
cida_plugin.agent.ROLE_PRIOR_BLEND = 0.05
cida_latent_plugin.agent.ROLE_PRIOR_BLEND = 0.05

DIM_NODE = 32
DIM_EDGE = 64  # DIM_NODE * 2

def generate_batch(batch_size, chain_len):
    # To prevent shortcut learning, negative samples contain all query nodes in the edge list,
    # but the path is disconnected at a random intermediate node.
    num_nodes_per_sample = chain_len + 3
    total_nodes = batch_size * num_nodes_per_sample
    node_embeddings = torch.randn(total_nodes, DIM_NODE)
    
    X_pooled = []
    X_seq = []
    Y = []
    
    for b in range(batch_size):
        offset = b * num_nodes_per_sample
        path = list(range(offset, offset + chain_len + 1))
        is_positive = np.random.rand() > 0.5
        
        edges = []
        if is_positive:
            # Connected path: A -> B -> C -> D
            for i in range(chain_len):
                u, v = path[i], path[i+1]
                edges.append(torch.cat([node_embeddings[u], node_embeddings[v]]))
            query = torch.cat([node_embeddings[path[0]], node_embeddings[path[-1]]])
            label = 1
        else:
            # Disconnected path but all query nodes present: A -> B and E -> D
            # We break the connection at index 1
            bad_node = offset + chain_len + 1
            for i in range(chain_len):
                if i == 1:
                    # Break the connection
                    u, v = bad_node, path[i+1]
                else:
                    u, v = path[i], path[i+1]
                edges.append(torch.cat([node_embeddings[u], node_embeddings[v]]))
            query = torch.cat([node_embeddings[path[0]], node_embeddings[path[-1]]])
            label = 0
            
        X_pooled.append(query)
        X_seq.append(torch.stack(edges))
        Y.append(label)
        
    return torch.stack(X_pooled), torch.stack(X_seq), torch.tensor(Y, dtype=torch.long)

class MLPBaseline(nn.Module):
    def __init__(self, d_input=64, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(d_input * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        
    def forward(self, pooled, seq_output, mask=None):
        seq_mean = seq_output.mean(dim=1)
        x = torch.cat([pooled, seq_mean], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def train_model(model, epochs=50, steps_per_epoch=60, batch_size=32, is_cida=False, jitter_rounds=False):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    cida_loss_fn = CIDALoss(lambda_cal=0.0, lambda_ac=0.0) if is_cida else None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for step in range(steps_per_epoch):
            batch_p, batch_s, batch_labels = generate_batch(batch_size, chain_len=2)
            
            if is_cida and jitter_rounds:
                model.config.max_rounds = int(np.random.randint(2, 5))
                
            optimizer.zero_grad()
            if is_cida:
                out = model(batch_p, seq_output=batch_s)
                loss, comp = cida_loss_fn(out["p_final"], batch_labels, out["b_all"], out.get("L_orth", None))
                logits = out["p_final"]
            else:
                logits = model(batch_p, batch_s)
                loss = criterion(logits, batch_labels)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_size
            
        acc = correct / total
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss/total:.4f} | Train Acc: {acc:.4f}")

def evaluate_model(model, num_eval_steps=20, batch_size=32, chain_len=2, is_cida=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(num_eval_steps):
            eval_p, eval_s, eval_labels = generate_batch(batch_size, chain_len=chain_len)
            if is_cida:
                out = model(eval_p, seq_output=eval_s)
                logits = out["p_final"]
            else:
                logits = model(eval_p, eval_s)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == eval_labels).sum().item()
            total += batch_size
    return correct / total

# Run Experiment
print("\n--- 1. Training MLP Baseline ---")
mlp = MLPBaseline(d_input=DIM_EDGE, hidden_dim=128)
train_model(mlp, epochs=40, is_cida=False)

print("\n--- 2. Training Standard CIDA (Fixed 2 rounds) ---")
cfg_std = CIDAPluginConfig(d_input=DIM_EDGE, d_hidden=128, num_classes=2, max_rounds=2)
cida_std = CIDAPlugin(cfg_std)
train_model(cida_std, epochs=40, is_cida=True)

print("\n--- 3. Training Latent CIDA (Jittered rounds 2-4) ---")
cfg_latent = LatentConfig(d_input=DIM_EDGE, d_hidden=128, num_classes=2, max_rounds=2)
cida_latent = LatentCIDAPlugin(cfg_latent)
train_model(cida_latent, epochs=40, is_cida=True, jitter_rounds=True)

# Evaluate in-distribution (Length-2)
mlp_acc_len2 = evaluate_model(mlp, chain_len=2, is_cida=False)
cida_std_acc_len2 = evaluate_model(cida_std, chain_len=2, is_cida=True)

cida_latent.config.max_rounds = 2
cida_latent_acc_len2 = evaluate_model(cida_latent, chain_len=2, is_cida=True)

# Evaluate out-of-distribution (Length-4 - The complex task)
mlp_acc_len4 = evaluate_model(mlp, chain_len=4, is_cida=False)

# Standard CIDA evaluated with max_rounds=2
cida_std_acc_len4 = evaluate_model(cida_std, chain_len=4, is_cida=True)

# Latent CIDA evaluated with max_rounds=5 (Thinking harder!)
cida_latent.config.max_rounds = 5
cida_latent_acc_len4 = evaluate_model(cida_latent, chain_len=4, is_cida=True)

print("\n" + "="*70)
print(f"{'Model Name':<25} | {'Length-2 (In-Dist) Acc':<22} | {'Length-4 (OOD) Acc':<18}")
print("="*70)
print(f"{'MLP Baseline (Mean Pool)':<25} | {mlp_acc_len2:<22.4f} | {mlp_acc_len4:<18.4f}")
print(f"{'Standard CIDA (2 rounds)':<25} | {cida_std_acc_len2:<22.4f} | {cida_std_acc_len4:<18.4f}")
print(f"{'Latent CIDA (5 rounds)':<25} | {cida_latent_acc_len2:<22.4f} | {cida_latent_acc_len4:<18.4f}")
print("="*70)
