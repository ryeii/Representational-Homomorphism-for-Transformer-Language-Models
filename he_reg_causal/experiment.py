#!/usr/bin/env python3
"""
he_regularization_experiment.py

Causal experiment:

Goal:
  Test whether actively enforcing *modifier homomorphism structure* (low Modifier HE)
  *causally* improves OOD compositional generalization.

Outputs:
  - results JSON in --outdir:
      rolling_results.json (updated after every run)
      he_regularization_results_<timestamp>.json (final snapshot)

Usage:
  python he_regularization_experiment.py
  python he_regularization_experiment.py --outdir results_he_reg_strong
  python he_regularization_experiment.py --lambda_he 0.2
  python he_regularization_experiment.py --track_ood_each_epoch 1

Assumptions about your existing code:
  - gen_data.generate_dataset(...) exists and returns list of (input_tokens, output_tokens)
  - transformers.DecoderOnlyTransformer supports:
        logits, hidden_states = model(x_ids)
        hidden_states is list length n_layers, each (B,T,D)
        model has attributes: n_layers, d_model
  - he_metrics.compute_layerwise_he(...) exists and matches the signature used below

"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from gen_data import generate_dataset, PRIMITIVES, OPERATORS, NOISE
from transformers import DecoderOnlyTransformer
from he_metrics import compute_layerwise_he


# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------
# Dataset wrapper
# ------------------------
class SeqDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        self.token2id = {tok: i for i, tok in enumerate(vocab)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

    def encode(self, seq):
        return torch.tensor([self.token2id[tok] for tok in seq], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp, out = self.data[idx]
        return self.encode(inp), self.encode(out)


def collate_batch(batch):
    inps, outs = zip(*batch)
    inps = nn.utils.rnn.pad_sequence(inps, batch_first=True, padding_value=0)
    outs = nn.utils.rnn.pad_sequence(outs, batch_first=True, padding_value=0)
    return inps, outs


# ------------------------
# Evaluation: token-level accuracy
# ------------------------
def evaluate_model_token_accuracy(model, test_loader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _hidden_states = model(x)  # (B, T_in, V)

            min_len = min(logits.size(1), y.size(1))
            logits = logits[:, :min_len, :]
            y = y[:, :min_len]

            preds = logits.argmax(-1)
            mask = (y != 0)
            correct += ((preds == y) & mask).sum().item()
            total += mask.sum().item()
    return correct / total if total > 0 else 0.0


# ------------------------
# Vocab (fixed across runs to make aggregation easy)
# ------------------------
def make_fixed_vocab() -> List[str]:
    vocab = set()
    vocab.update(PRIMITIVES)
    vocab.update([p.upper() for p in PRIMITIVES])
    vocab.update(OPERATORS)
    vocab.update(NOISE)
    vocab.add("then")
    vocab_list = ["<pad>"] + sorted(vocab)
    return vocab_list


# ------------------------
# OOD test loaders: num_primitives=5..12, noise=0, fixed seed
# ------------------------
def build_ood_test_loaders(vocab: List[str], seed_for_ood: int, max_samples_per_n: int = 200):
    ood = {}
    for n_prim in range(5, 13):
        test_data = generate_dataset(
            num_primitives=n_prim,
            num_noise=0,
            max_samples=max_samples_per_n,
            seed=seed_for_ood,
        )
        test_ds = SeqDataset(test_data, vocab)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_batch)
        ood[n_prim] = test_loader
    return ood


# ------------------------
# Build HE pairs for post-hoc HE computation
# ------------------------
def build_he_pairs(model, dataset: SeqDataset, device: torch.device):
    model.eval()

    prim_tokens = PRIMITIVES
    mod_tokens = OPERATORS

    prim_ids = [dataset.token2id[t] for t in prim_tokens]
    mod_ids = [dataset.token2id[t] for t in mod_tokens]

    train_pairs_mod = []
    train_pairs_seq = []

    with torch.no_grad():
        for inp_ids, _out_ids in dataset:
            inp_ids = inp_ids.to(device).unsqueeze(0)  # (1, T)
            hidden_states = model.get_hidden_states(inp_ids)  # list (B,T,D)
            last_h = hidden_states[-1][0]  # (T,D)

            # Modifier pairs: primitive followed by modifier
            for i, tok_id in enumerate(inp_ids[0]):
                if tok_id in prim_ids and i + 1 < inp_ids.size(1) and inp_ids[0, i + 1] in mod_ids:
                    prim_vec = last_h[i]
                    mod_vec = last_h[i + 1]
                    comp_vec = last_h[i : i + 2].mean(dim=0)
                    train_pairs_mod.append((prim_vec, mod_vec, comp_vec))

            # Sequence pairs: split into primitive-initiated segments
            parts = []
            current_part = []
            for tok_id in inp_ids[0]:
                if tok_id in prim_ids:
                    if current_part:
                        parts.append(current_part)
                    current_part = [tok_id]
                else:
                    current_part.append(tok_id)
            if current_part:
                parts.append(current_part)

            idx = 0
            part_vecs = []
            for part in parts:
                length = len(part)
                vec = last_h[idx : idx + length].mean(dim=0)
                part_vecs.append(vec)
                idx += length

            for i in range(len(part_vecs) - 1):
                e1_vec = part_vecs[i]
                e2_vec = part_vecs[i + 1]
                comp_vec = (e1_vec + e2_vec) / 2
                train_pairs_seq.append((e1_vec, e2_vec, comp_vec))

    # 80/20 split
    val_split = 0.2
    n_mod = len(train_pairs_mod)
    n_seq = len(train_pairs_seq)

    val_pairs_mod = train_pairs_mod[int(n_mod * (1 - val_split)) :]
    train_pairs_mod = train_pairs_mod[: int(n_mod * (1 - val_split))]

    val_pairs_seq = train_pairs_seq[int(n_seq * (1 - val_split)) :]
    train_pairs_seq = train_pairs_seq[: int(n_seq * (1 - val_split))]

    return train_pairs_mod, val_pairs_mod, train_pairs_seq, val_pairs_seq


# ------------------------
# Modifier HE Regularizer
# - cached pool of (sequence_ids, prim_pos, mod_pos, mod_token_id)
# - per-modifier small MLP on concat([prim_vec, mod_vec]) -> predicted_comp_vec
# - target comp_vec = mean([prim_vec, mod_vec]) to match your combined representation definition
# - applied at multiple layers (e.g., layers 2 and 4), averaged
# ------------------------
def mine_modifier_pair_pool(
    train_ds: SeqDataset,
    max_pairs: int,
) -> List[Tuple[torch.Tensor, int, int, int]]:
    """
    Returns a list of tuples:
      (seq_ids_tensor (T,), prim_pos, mod_pos, mod_token_id)
    where mod_pos = prim_pos + 1 and token at mod_pos is a modifier.
    """
    prim_ids = set(train_ds.token2id[t] for t in PRIMITIVES)
    mod_ids = set(train_ds.token2id[t] for t in OPERATORS)

    pool: List[Tuple[torch.Tensor, int, int, int]] = []
    for i in range(len(train_ds)):
        x_ids, _y_ids = train_ds[i]
        T = x_ids.size(0)
        if T < 2:
            continue
        for t in range(T - 1):
            if int(x_ids[t]) in prim_ids and int(x_ids[t + 1]) in mod_ids:
                pool.append((x_ids.clone(), t, t + 1, int(x_ids[t + 1])))
                if len(pool) >= max_pairs:
                    return pool
    return pool


def collate_he_pool_batch(items: List[Tuple[torch.Tensor, int, int, int]]):
    seqs = [it[0] for it in items]
    prim_pos = torch.tensor([it[1] for it in items], dtype=torch.long)
    mod_pos = torch.tensor([it[2] for it in items], dtype=torch.long)
    mod_id = torch.tensor([it[3] for it in items], dtype=torch.long)

    x = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return x, prim_pos, mod_pos, mod_id


class PerModifierMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)


class StrongModifierHERegularizer(nn.Module):
    def __init__(self, d_model: int, modifier_token_ids: List[int], mlp_hidden: int = 256):
        super().__init__()
        self.d_model = d_model
        self.mod_ids = list(modifier_token_ids)
        self.ops = nn.ModuleDict({str(mid): PerModifierMLP(d_model, mlp_hidden) for mid in self.mod_ids})

    def forward(
        self,
        hidden_states_at_layers: List[torch.Tensor],  # list of (B,T,D) for selected layers
        x_ids: torch.Tensor,                          # (B,T)
        prim_pos: torch.Tensor,                       # (B,)
        mod_pos: torch.Tensor,                        # (B,)
        mod_tok: torch.Tensor,                        # (B,)
    ) -> torch.Tensor:
        """
        Computes mean MSE across layers and across modifiers present in batch.
        """
        # Gather per layer
        layer_losses = []
        for h in hidden_states_at_layers:
            # h: (B,T,D)
            B, T, D = h.shape
            pv = h[torch.arange(B, device=h.device), prim_pos.to(h.device), :]  # (B,D)
            mv = h[torch.arange(B, device=h.device), mod_pos.to(h.device), :]   # (B,D)
            tgt = (pv + mv) / 2.0                                               # (B,D)

            losses = []
            for mid in self.mod_ids:
                mask = (mod_tok == mid)
                if mask.any():
                    inp = torch.cat([pv[mask], mv[mask]], dim=-1)
                    pred = self.ops[str(mid)](inp)
                    losses.append(nn.functional.mse_loss(pred, tgt[mask], reduction="mean"))
            if len(losses) == 0:
                layer_losses.append(h.new_tensor(0.0))
            else:
                layer_losses.append(torch.stack(losses, dim=0).mean())

        if len(layer_losses) == 0:
            return x_ids.new_tensor(0.0, dtype=torch.float32)

        return torch.stack(layer_losses, dim=0).mean()


# ------------------------
# Training
# ------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    condition: str,
    lambda_he: float,
    he_pool: Optional[List[Tuple[torch.Tensor, int, int, int]]],
    he_batch_size: int,
    he_layers: List[int],  # 1-indexed layer numbers to regularize
    train_ds: SeqDataset,
    track_ood_each_epoch: bool,
    ood_test_loaders: Dict[int, DataLoader],
) -> Dict[str, Any]:
    """
    Training with optional strong HE-regularization.
    Logs per-epoch:
      ce_loss, he_loss, total_loss
    Optionally logs per-epoch OOD accuracies.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Setup HE regularizer if needed
    if condition == "he_reg":
        if he_pool is None or len(he_pool) == 0:
            raise RuntimeError("HE regularization requested but he_pool is empty.")
        mod_ids = [train_ds.token2id[t] for t in OPERATORS]
        he_reg = StrongModifierHERegularizer(
            d_model=model.d_model,
            modifier_token_ids=mod_ids,
            mlp_hidden=256,
        ).to(device)
        params = list(model.parameters()) + list(he_reg.parameters())
    else:
        he_reg = None
        params = list(model.parameters())

    optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.98))

    # Convert he_layers to indices into hidden_states list
    # hidden_states list is 0-indexed, with length model.n_layers
    he_layer_indices = [l - 1 for l in he_layers]

    logs: Dict[str, Any] = {
        "ce_loss": [],
        "he_loss": [],
        "total_loss": [],
        "ood_each_epoch": {},  # epoch -> {n_prim: acc} (optional)
    }

    # For reproducible HE batch sampling per epoch
    rng = np.random.default_rng(12345)

    model.train()
    for ep in range(1, epochs + 1):
        ce_sum = 0.0
        he_sum = 0.0
        tot_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits, hidden_states = model(x)  # logits (B,T,V), hidden_states list (B,T,D)

            min_len = min(logits.size(1), y.size(1))
            logits2 = logits[:, :min_len, :]
            y2 = y[:, :min_len]

            ce_loss = criterion(logits2.transpose(1, 2), y2)

            if condition == "he_reg":
                # Sample an HE batch from the cached pool (separate forward pass)
                idxs = rng.integers(low=0, high=len(he_pool), size=he_batch_size)
                items = [he_pool[int(i)] for i in idxs]
                he_x, prim_pos, mod_pos, mod_tok = collate_he_pool_batch(items)
                he_x = he_x.to(device)
                prim_pos = prim_pos.to(device)
                mod_pos = mod_pos.to(device)
                mod_tok = mod_tok.to(device)

                _he_logits, he_hidden_states = model(he_x)
                selected_layers = [he_hidden_states[j] for j in he_layer_indices]
                he_loss = he_reg(selected_layers, he_x, prim_pos, mod_pos, mod_tok)

                total = ce_loss + lambda_he * he_loss
            else:
                he_loss = torch.tensor(0.0, device=device)
                total = ce_loss

            total.backward()
            optimizer.step()

            ce_sum += float(ce_loss.item())
            he_sum += float(he_loss.item())
            tot_sum += float(total.item())
            n_batches += 1

        logs["ce_loss"].append(ce_sum / max(1, n_batches))
        logs["he_loss"].append(he_sum / max(1, n_batches))
        logs["total_loss"].append(tot_sum / max(1, n_batches))

        if track_ood_each_epoch:
            # Evaluate OOD each epoch (more compute, but useful for dynamics plots)
            accs = {}
            for n_prim, loader in ood_test_loaders.items():
                accs[int(n_prim)] = evaluate_model_token_accuracy(model, loader)
            logs["ood_each_epoch"][int(ep)] = accs

        print(
            f"  epoch {ep:03d}/{epochs} | CE={logs['ce_loss'][-1]:.6f} "
            f"| HE={logs['he_loss'][-1]:.6f} | total={logs['total_loss'][-1]:.6f}"
        )

    return logs


# ------------------------
# Run one configuration
# ------------------------
def run_one(
    seed: int,
    num_noise: int,
    condition: str,
    vocab: List[str],
    ood_test_loaders: Dict[int, DataLoader],
    epochs: int,
    lr: float,
    lambda_he: float,
    he_pool_max_pairs: int,
    he_batch_size: int,
    he_layers: List[int],
    he_probe_epochs: int,
    track_ood_each_epoch: bool,
    train_max_samples: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train data: fixed 2 primitives, varying noise (as in your original noise experiment)
    train_data = generate_dataset(
        num_primitives=2,
        num_noise=num_noise,
        max_samples=train_max_samples,
        seed=seed,
    )
    train_ds = SeqDataset(train_data, vocab)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_batch)

    model = DecoderOnlyTransformer(
        vocab_size=len(vocab),
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=256,
        max_len=50,
    ).to(device)

    # Pre-mine HE pair pool (only needed for he_reg)
    he_pool = None
    if condition == "he_reg":
        he_pool = mine_modifier_pair_pool(train_ds, max_pairs=he_pool_max_pairs)
        if len(he_pool) == 0:
            raise RuntimeError(
                "No modifier pairs found in training set. "
                "Check dataset construction; HE-reg requires primitive+modifier occurrences."
            )

    # Train
    logs = train_model(
        model=model,
        train_loader=train_loader,
        epochs=epochs,
        lr=lr,
        condition=condition,
        lambda_he=lambda_he,
        he_pool=he_pool,
        he_batch_size=he_batch_size,
        he_layers=he_layers,
        train_ds=train_ds,
        track_ood_each_epoch=track_ood_each_epoch,
        ood_test_loaders=ood_test_loaders,
    )

    # Final OOD accuracy by complexity
    accs = {}
    for n_prim, loader in ood_test_loaders.items():
        accs[int(n_prim)] = evaluate_model_token_accuracy(model, loader)

    # Final post-hoc HE (same as your current HE pipeline)
    train_pairs_mod, val_pairs_mod, train_pairs_seq, val_pairs_seq = build_he_pairs(model, train_ds, device)

    he_results = compute_layerwise_he(
        layers=model.n_layers,
        d=model.d_model,
        train_pairs_mod=train_pairs_mod,
        val_pairs_mod=val_pairs_mod,
        train_pairs_seq=train_pairs_seq,
        val_pairs_seq=val_pairs_seq,
        operator_kinds=["linear", "bilinear", "mlp"],
        epochs=he_probe_epochs,
        device=device,
        verbose=False,
    )

    # Average across operator families per layer
    he_modifier = {
        int(layer): float(np.mean([m["mse"] for m in op_dict.values()]))
        for layer, op_dict in he_results["modifier"].items()
    }
    he_sequence = {
        int(layer): float(np.mean([m["mse"] for m in op_dict.values()]))
        for layer, op_dict in he_results["sequence"].items()
    }

    return {
        "config": {
            "seed": seed,
            "num_noise": num_noise,
            "condition": condition,
            "epochs": epochs,
            "lr": lr,
            "lambda_he": (0.0 if condition == "baseline" else lambda_he),
            "he_layers": (None if condition == "baseline" else he_layers),
            "he_pool_max_pairs": (None if condition == "baseline" else he_pool_max_pairs),
            "he_batch_size": (None if condition == "baseline" else he_batch_size),
            "he_probe_epochs": he_probe_epochs,
            "train_max_samples": train_max_samples,
            "model": {
                "type": "DecoderOnlyTransformer",
                "d_model": 128,
                "n_layers": 4,
                "n_heads": 4,
                "d_ff": 256,
            },
            "vocab_size": len(vocab),
        },
        "train_logs": logs,  # per-epoch
        "ood_accuracy": {int(k): float(v) for k, v in accs.items()},
        "he_modifier_layerwise": he_modifier,
        "he_sequence_layerwise": he_sequence,
        "he_pair_counts": {
            "modifier_pairs_total": int(len(train_pairs_mod) + len(val_pairs_mod)),
            "sequence_pairs_total": int(len(train_pairs_seq) + len(val_pairs_seq)),
        },
        "he_pool_count": (0 if he_pool is None else int(len(he_pool))),
    }


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--outdir", type=str, default="results_he_reg_strong", help="Output directory for JSON results")

    # Compute / run sizing
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs (paper uses 50)")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate (paper uses 1e-4)")
    ap.add_argument("--train_max_samples", type=int, default=2000, help="Max train samples per run (increase for stronger effect)")

    # HE regularizer strength and compute
    ap.add_argument("--lambda_he", type=float, default=0.2, help="HE regularization strength (try 0.1~0.5)")
    ap.add_argument(
        "--he_layers",
        type=str,
        default="2,4",
        help="Comma-separated layer numbers (1-indexed) to apply HE-reg on (e.g. '2,4')",
    )
    ap.add_argument("--he_pool_max_pairs", type=int, default=2048, help="Max cached modifier pairs in HE pool")
    ap.add_argument("--he_batch_size", type=int, default=256, help="HE batch size sampled from pool each step")

    # Post-hoc HE probe training epochs (more = slower, but more stable)
    ap.add_argument("--he_probe_epochs", type=int, default=50, help="Epochs for post-hoc HE probe training")

    # OOD test sets
    ap.add_argument("--ood_seed", type=int, default=123, help="Seed used to generate fixed OOD test sets")
    ap.add_argument("--ood_max_samples_per_n", type=int, default=200, help="OOD samples per complexity level")

    # Run grid defaults (bigger than prior script)
    ap.add_argument("--noise_levels", type=str, default="0,3,6,9,12,15", help="Comma-separated noise levels")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds")

    # Optional: log OOD each epoch
    ap.add_argument("--track_ood_each_epoch", type=int, default=0, help="Set 1 to evaluate OOD each epoch (slower)")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    noise_levels = [int(x.strip()) for x in args.noise_levels.split(",") if x.strip() != ""]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip() != ""]
    he_layers = [int(x.strip()) for x in args.he_layers.split(",") if x.strip() != ""]

    vocab = make_fixed_vocab()
    ood_test_loaders = build_ood_test_loaders(
        vocab=vocab,
        seed_for_ood=args.ood_seed,
        max_samples_per_n=args.ood_max_samples_per_n,
    )

    # Run both conditions for each (seed, noise)
    conditions = ["baseline", "he_reg"]

    all_runs: List[Dict[str, Any]] = []

    for seed in seeds:
        for n_noise in noise_levels:
            for cond in conditions:
                print(f"\n=== RUN: seed={seed} noise={n_noise} condition={cond} ===")
                out = run_one(
                    seed=seed,
                    num_noise=n_noise,
                    condition=cond,
                    vocab=vocab,
                    ood_test_loaders=ood_test_loaders,
                    epochs=args.epochs,
                    lr=args.lr,
                    lambda_he=args.lambda_he,
                    he_pool_max_pairs=args.he_pool_max_pairs,
                    he_batch_size=args.he_batch_size,
                    he_layers=he_layers,
                    he_probe_epochs=args.he_probe_epochs,
                    track_ood_each_epoch=bool(args.track_ood_each_epoch),
                    train_max_samples=args.train_max_samples,
                )
                all_runs.append(out)

                # Rolling checkpoint
                rolling_path = os.path.join(args.outdir, "rolling_results.json")
                with open(rolling_path, "w") as f:
                    json.dump(
                        {
                            "meta": {
                                "created": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                "device": str(device),
                                "noise_levels": noise_levels,
                                "seeds": seeds,
                                "conditions": conditions,
                                "args": vars(args),
                            },
                            "vocab": vocab,
                            "runs": all_runs,
                        },
                        f,
                        indent=2,
                    )

    # Final save
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_path = os.path.join(args.outdir, f"he_regularization_results_{stamp}.json")
    with open(final_path, "w") as f:
        json.dump(
            {
                "meta": {
                    "created": stamp,
                    "device": str(device),
                    "noise_levels": noise_levels,
                    "seeds": seeds,
                    "conditions": conditions,
                    "args": vars(args),
                },
                "vocab": vocab,
                "runs": all_runs,
            },
            f,
            indent=2,
        )

    print(f"\nSaved results to: {final_path}")
    print(f"Rolling checkpoint: {os.path.join(args.outdir, 'rolling_results.json')}")


if __name__ == "__main__":
    main()
