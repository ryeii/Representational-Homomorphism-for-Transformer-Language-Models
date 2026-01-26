import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math

# -----------------------
# Operator module defs
# -----------------------
class LinearOp(nn.Module):
    """Linear operator on concatenated inputs [a; b] -> d"""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, a, b):
        # a,b: (batch, d)
        x = torch.cat([a, b], dim=-1)
        return self.lin(x)

class BilinearConcatOp(nn.Module):
    """
    Bilinear-ish operator implemented by concatenating [a; b; a*b]
    and applying a linear map. Captures multiplicative interactions
    without using a full 3-tensor.
    """
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d * 3, d)

    def forward(self, a, b):
        x = torch.cat([a, b, a * b], dim=-1)
        return self.lin(x)

class MLPOp(nn.Module):
    """2-layer MLP on concatenation [a; b] -> d"""
    def __init__(self, d: int, hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden is None:
            hidden = max(4 * d, 128)
        self.fc1 = nn.Linear(2 * d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu

    def forward(self, a, b):
        x = torch.cat([a, b], dim=-1)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# -----------------------
# Utility helpers
# -----------------------
def _to_tensor(xs):
    """Convert list/tuple of arrays/tensors to single tensor."""
    if isinstance(xs, torch.Tensor):
        return xs
    return torch.stack([torch.as_tensor(x, dtype=torch.get_default_dtype()) for x in xs], dim=0)

def _batch_iter(xs, ys, zs, batch_size):
    N = xs.shape[0]
    perm = torch.randperm(N)
    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        yield xs[idx], ys[idx], zs[idx]

def cosine_error(pred, target, eps=1e-8):
    # return 1 - cosine_similarity averaged
    num = (pred * target).sum(dim=-1)
    den = torch.norm(pred, dim=-1) * torch.norm(target, dim=-1) + eps
    cos = num / den
    return (1.0 - cos).mean().item()

# -----------------------
# Main function
# -----------------------
def compute_layerwise_he(
    layers: int,
    d: int,
    train_pairs_mod: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_pairs_mod: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    train_pairs_seq: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_pairs_seq: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    operator_kinds: List[str] = ["linear", "bilinear", "mlp"],
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    device: Optional[torch.device] = None,
    verbose: bool = False
) -> Dict:
    """
    Train composition operators per layer and compute homomorphism error (MSE and cosine)
    for modifier composition and sequence composition.

    Args:
        layers: number of layers for which we have representations (e.g., 1..L)
        d: representation dimensionality
        train_pairs_mod: list of (prim_vec, mod_vec, prim_with_mod_vec) for training
        val_pairs_mod: validation set (same tuple format)
        train_pairs_seq: list of (e1_vec, e2_vec, e12_vec) for training
        val_pairs_seq: validation set (same)
        operator_kinds: which operators to train: subset of ["linear","bilinear","mlp"]
        epochs, batch_size, lr, weight_decay: training hyperparams
        device: torch.device or None -> auto select cuda if available
    Returns:
        results: dict with keys:
          results['modifier'][layer_idx][op] = {'mse': float, 'cos': float}
          results['sequence'][layer_idx][op] = {...}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert datasets to tensors per layer: user should have passed vectors already pooled.
    # For convenience accept lists of tensors (N, d)
    def _prepare(pairs):
        if len(pairs) == 0:
            return None, None, None
        a = _to_tensor([p[0] for p in pairs]).to(device)  # (N, d)
        b = _to_tensor([p[1] for p in pairs]).to(device)
        c = _to_tensor([p[2] for p in pairs]).to(device)
        return a, b, c

    train_mod_a, train_mod_b, train_mod_c = _prepare(train_pairs_mod)
    val_mod_a, val_mod_b, val_mod_c = _prepare(val_pairs_mod)
    train_seq_a, train_seq_b, train_seq_c = _prepare(train_pairs_seq)
    val_seq_a, val_seq_b, val_seq_c = _prepare(val_pairs_seq)

    results = {'modifier': {}, 'sequence': {}}

    # iterate over layers (0..layers-1)
    for layer_idx in range(layers):
        if verbose:
            print(f"=== Layer {layer_idx} ===")
        results['modifier'][layer_idx] = {}
        results['sequence'][layer_idx] = {}

        # prepare per-layer tensors: assumption: provided vectors already correspond to this layer.
        # (If you store layerwise in arrays, caller must pass per-layer datasets.)
        # Here we simply train operators on the provided tensors (they should be per-layer vectors).
        # If caller wants to do multi-layer in one call, they should call this function per layer or
        # pass flattened per-layer data. For simplicity we assume pairs are already per-layer.
        for op_kind in operator_kinds:
            # instantiate operator
            if op_kind == "linear":
                op_mod = LinearOp(d * 2, d).to(device) if False else LinearOp(d * 2, d).to(device)  # stays as LinearOp(d*2->d)
                # Note: LinearOp defined expects d_in==2*d; our class uses constructor LinearOp(d_in,d_out)
                # But above class signature created LinearOp(d_in,d_out); using d_in=2*d
                # We'll rewrap to ensure compatibility.
                # Recreate operator properly:
                op_mod = LinearOp(2 * d, d).to(device)
                op_seq = LinearOp(2 * d, d).to(device)
            elif op_kind == "bilinear":
                op_mod = BilinearConcatOp(d).to(device)
                op_seq = BilinearConcatOp(d).to(device)
            elif op_kind == "mlp":
                op_mod = MLPOp(d).to(device)
                op_seq = MLPOp(d).to(device)
            else:
                raise ValueError(f"Unknown op kind {op_kind}")

            # train modifier op if data exists
            if train_mod_a is not None:
                optim = torch.optim.Adam(op_mod.parameters(), lr=lr, weight_decay=weight_decay)
                # training loop
                op_mod.train()
                N = train_mod_a.size(0)
                for epoch in range(epochs):
                    epoch_loss = 0.0
                    for i in range(0, N, batch_size):
                        a = train_mod_a[i:i+batch_size]
                        b = train_mod_b[i:i+batch_size]
                        tgt = train_mod_c[i:i+batch_size]
                        pred = op_mod(a, b)
                        loss = F.mse_loss(pred, tgt)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        epoch_loss += loss.item() * a.size(0)
                    # optional small early stop if loss tiny
                    if epoch % 50 == 0 and verbose:
                        avg = epoch_loss / max(1, N)
                        print(f"  mod[{op_kind}] epoch {epoch} loss {avg:.6f}")
                # eval on val
                op_mod.eval()
                with torch.no_grad():
                    pred_val = op_mod(val_mod_a, val_mod_b) if val_mod_a is not None else None
                    if val_mod_a is not None:
                        mse = F.mse_loss(pred_val, val_mod_c).item()
                        cos = cosine_error(pred_val, val_mod_c)
                    else:
                        mse = float('nan'); cos = float('nan')
                results['modifier'][layer_idx][op_kind] = {'mse': mse, 'cos': cos}
            else:
                results['modifier'][layer_idx][op_kind] = {'mse': float('nan'), 'cos': float('nan')}

            # train sequence op if data exists
            if train_seq_a is not None:
                optim2 = torch.optim.Adam(op_seq.parameters(), lr=lr, weight_decay=weight_decay)
                op_seq.train()
                N2 = train_seq_a.size(0)
                for epoch in range(epochs):
                    epoch_loss = 0.0
                    for i in range(0, N2, batch_size):
                        a = train_seq_a[i:i+batch_size]
                        b = train_seq_b[i:i+batch_size]
                        tgt = train_seq_c[i:i+batch_size]
                        pred = op_seq(a, b)
                        loss = F.mse_loss(pred, tgt)
                        optim2.zero_grad()
                        loss.backward()
                        optim2.step()
                        epoch_loss += loss.item() * a.size(0)
                    if epoch % 50 == 0 and verbose:
                        avg = epoch_loss / max(1, N2)
                        print(f"  seq[{op_kind}] epoch {epoch} loss {avg:.6f}")
                op_seq.eval()
                with torch.no_grad():
                    pred_val2 = op_seq(val_seq_a, val_seq_b) if val_seq_a is not None else None
                    if val_seq_a is not None:
                        mse2 = F.mse_loss(pred_val2, val_seq_c).item()
                        cos2 = cosine_error(pred_val2, val_seq_c)
                    else:
                        mse2 = float('nan'); cos2 = float('nan')
                results['sequence'][layer_idx][op_kind] = {'mse': mse2, 'cos': cos2}
            else:
                results['sequence'][layer_idx][op_kind] = {'mse': float('nan'), 'cos': float('nan')}

    return results
