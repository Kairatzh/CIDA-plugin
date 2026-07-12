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

DIM_NODE = 32
DIM_EDGE = 64  # DIM_NODE * 2

def generate_transitive_data(num_samples=1200, chain_len=2):
    num_nodes = 30
    node_embeddings = torch.randn(num_nodes, DIM_NODE)
    
    X_pooled = []  # Queries: (B, 64)
    X_seq = []     # Edges: (B, chain_len, 64)
    Y = []         # Labels: (B,)
    
    for _ in range(num_samples):
        # Generate a path of nodes
        path = np.random.choice(num_nodes, chain_len + 1, replace=False)
        is_positive = np.random.rand() > 0.5
        
        edges = []
        if is_positive:
            for i in range(chain_len):
                u, v = path[i], path[i+1]
                edges.append(torch.cat([node_embeddings[u], node_embeddings[v]]))
            query = torch.cat([node_embeddings[path[0]], node_embeddings[path[-1]]])
            label = 1
        else:
            random_nodes = np.random.choice(num_nodes, chain_len + 1, replace=False)
            for i in range(chain_len):
                u, v = random_nodes[i], random_nodes[i+1]
                edges.append(torch.cat([node_embeddings[u], node_embeddings[v]]))
            query = torch.cat([node_embeddings[path[0]], node_embeddings[path[-1]]])
            label = 0
            
        X_pooled.append(query)
        X_seq.append(torch.stack(edges))
        Y.append(label)
        
    return torch.stack(X_pooled), torch.stack(X_seq), torch.tensor(Y, dtype=torch.long)

print("[DATA] Generating datasets...")
train_pooled, train_seq, train_y = generate_transitive_data(2000, chain_len=2)
test_pooled_len2, test_seq_len2, test_y_len2 = generate_transitive_data(600, chain_len=2)
test_pooled_len4, test_seq_len4, test_y_len4 = generate_transitive_data(600, chain_len=4)

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

def train_model(model, train_p, train_s, train_labels, epochs=50, batch_size=32, is_cida=False, jitter_rounds=False):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    cida_loss_fn = CIDALoss(lambda_cal=0.2, lambda_ac=0.1) if is_cida else None
    
    num_samples = train_p.size(0)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_p = train_p[indices]
            batch_s = train_s[indices]
            batch_labels = train_labels[indices]
            
            # Dynamic jittering of deliberation rounds
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
            
            epoch_loss += loss.item() * len(indices)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += len(indices)
            
        acc = correct / total
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss/total:.4f} | Train Acc: {acc:.4f}")

def evaluate_model(model, test_p, test_s, test_labels, is_cida=False):
    model.eval()
    with torch.no_grad():
        if is_cida:
            out = model(test_p, seq_output=test_s)
            logits = out["p_final"]
        else:
            logits = model(test_p, test_s)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == test_labels).float().mean().item()
    return acc

print("\n--- 1. Training MLP Baseline ---")
mlp = MLPBaseline(d_input=DIM_EDGE)
train_model(mlp, train_pooled, train_seq, train_y, epochs=40, is_cida=False)

print("\n--- 2. Training Standard CIDA (Fixed 2 rounds) ---")
cfg_std = CIDAPluginConfig(d_input=DIM_EDGE, d_hidden=64, num_classes=2, max_rounds=2)
cida_std = CIDAPlugin(cfg_std)
train_model(cida_std, train_pooled, train_seq, train_y, epochs=40, is_cida=True)

print("\n--- 3. Training Latent CIDA (Jittered rounds 2-4) ---")
cfg_latent = LatentConfig(d_input=DIM_EDGE, d_hidden=64, num_classes=2, max_rounds=2)
cida_latent = LatentCIDAPlugin(cfg_latent)
# Train with dynamic rounds
train_model(cida_latent, train_pooled, train_seq, train_y, epochs=40, is_cida=True, jitter_rounds=True)

# Evaluate in-distribution (Length-2)
mlp_acc_len2 = evaluate_model(mlp, test_pooled_len2, test_seq_len2, test_y_len2, is_cida=False)
cida_std_acc_len2 = evaluate_model(cida_std, test_pooled_len2, test_seq_len2, test_y_len2, is_cida=True)

# Latent CIDA evaluated with max_rounds=2
cida_latent.config.max_rounds = 2
cida_latent_acc_len2 = evaluate_model(cida_latent, test_pooled_len2, test_seq_len2, test_y_len2, is_cida=True)

# Evaluate out-of-distribution (Length-4)
mlp_acc_len4 = evaluate_model(mlp, test_pooled_len4, test_seq_len4, test_y_len4, is_cida=False)

# Standard CIDA evaluated with max_rounds=2 (its fixed size)
cida_std_acc_len4 = evaluate_model(cida_std, test_pooled_len4, test_seq_len4, test_y_len4, is_cida=True)

# Latent CIDA evaluated with max_rounds=5 (Thinking harder!)
cida_latent.config.max_rounds = 5
cida_latent_acc_len4 = evaluate_model(cida_latent, test_pooled_len4, test_seq_len4, test_y_len4, is_cida=True)

print("\n" + "="*70)
print(f"{'Model Name':<25} | {'Length-2 (In-Dist) Acc':<22} | {'Length-4 (OOD) Acc':<18}")
print("="*70)
print(f"{'MLP Baseline (Mean Pool)':<25} | {mlp_acc_len2:<22.4f} | {mlp_acc_len4:<18.4f}")
print(f"{'Standard CIDA (2 rounds)':<25} | {cida_std_acc_len2:<22.4f} | {cida_std_acc_len4:<18.4f}")
print(f"{'Latent CIDA (5 rounds)':<25} | {cida_latent_acc_len2:<22.4f} | {cida_latent_acc_len4:<18.4f}")
print("="*70)
