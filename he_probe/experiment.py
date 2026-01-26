# experiment.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from gen_data import generate_dataset
from transformers import DecoderOnlyTransformer
from he_metrics import compute_layerwise_he  # from your previous implementation

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
# Training & evaluation
# ------------------------
def train_model(model, train_loader, vocab_size, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index

    model.train()
    for ep in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits, _ = model(x)  # (B, T_in, V)
            # Align lengths: sometimes logits and targets mismatch
            min_len = min(logits.size(1), y.size(1))
            logits = logits[:, :min_len, :]
            y = y[:, :min_len]

            logits = logits.transpose(1, 2)  # (B, V, T)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep+1}: loss={total_loss/len(train_loader):.4f}")


def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)  # (B, T_in, V)

            # Align lengths
            min_len = min(logits.size(1), y.size(1))
            logits = logits[:, :min_len, :]
            y = y[:, :min_len]

            preds = logits.argmax(-1)  # (B, T)
            mask = (y != 0)
            correct += ((preds == y) & mask).sum().item()
            total += mask.sum().item()
    return correct / total if total > 0 else 0.0

def build_he_pairs(model, dataset, device):
    """
    Build modifier and sequence pairs for homomorphism error computation.

    Returns:
        train_pairs_mod: list of (prim_vec, mod_vec, prim_mod_vec)
        val_pairs_mod: same
        train_pairs_seq: list of (part1_vec, part2_vec, composed_vec)
        val_pairs_seq: same
    """
    model.eval()

    # Map primitives, modifiers, and noise to IDs
    prim_tokens = ['walk', 'jump', 'look', 'turn']
    mod_tokens = ['twice', 'thrice']
    noise_tokens = ['foo', 'bar', 'baz']
    then_token = 'then'

    prim_ids = [dataset.token2id[t] for t in prim_tokens]
    mod_ids = [dataset.token2id[t] for t in mod_tokens]
    # noise_ids = [dataset.token2id[t] for t in noise_tokens]
    # then_id = dataset.token2id.get(then_token, None)

    train_pairs_mod = []
    train_pairs_seq = []

    with torch.no_grad():
        for inp_ids, out_ids in dataset:
            inp_ids = inp_ids.to(device).unsqueeze(0)  # (1, T)
            hidden_states = model.get_hidden_states(inp_ids)  # list of (B, T, D)
            last_h = hidden_states[-1][0]  # (T, D)

            # --- Modifier pairs ---
            for i, tok_id in enumerate(inp_ids[0]):
                if tok_id in prim_ids and i + 1 < inp_ids.size(1) and inp_ids[0, i+1] in mod_ids:
                    prim_vec = last_h[i]
                    mod_vec = last_h[i+1]
                    comp_vec = last_h[i:i+2].mean(dim=0)  # combined representation
                    train_pairs_mod.append((prim_vec, mod_vec, comp_vec))

            # --- Sequence pairs between consecutive parts ---
            # Split into parts: each part starts at a primitive and includes all tokens up to the next primitive
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

            # Map each part to a single vector (mean over tokens)
            idx = 0
            part_vecs = []
            for part in parts:
                length = len(part)
                vec = last_h[idx:idx+length].mean(dim=0)
                part_vecs.append(vec)
                idx += length

            # Build consecutive part pairs
            for i in range(len(part_vecs) - 1):
                e1_vec = part_vecs[i]
                e2_vec = part_vecs[i + 1]
                comp_vec = (e1_vec + e2_vec) / 2  # composition vector
                train_pairs_seq.append((e1_vec, e2_vec, comp_vec))

    # Split into train/val 80/20
    val_split = 0.2
    n_mod = len(train_pairs_mod)
    n_seq = len(train_pairs_seq)

    val_pairs_mod = train_pairs_mod[int(n_mod*(1-val_split)):]
    train_pairs_mod = train_pairs_mod[:int(n_mod*(1-val_split))]

    val_pairs_seq = train_pairs_seq[int(n_seq*(1-val_split)):]
    train_pairs_seq = train_pairs_seq[:int(n_seq*(1-val_split))]

    return train_pairs_mod, val_pairs_mod, train_pairs_seq, val_pairs_seq



# ------------------------
# Main experiment
# ------------------------
def run_all_experiments(seed=42, epochs=15, lr=1e-3):
    """
    Run all experiments (layer depth, sparsity, noise) with a fixed random seed.
    Returns a dictionary containing accuracy and HE results.
    """
    print(f"Running experiments with seed {seed}...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 1. Generate OOD test sets (primitives 5 to 12) ---
    print("Generating OOD test sets...")
    ood_test_sets = {}
    for n_prim in range(5, 13):
        test_data = generate_dataset(num_primitives=n_prim, num_noise=0, max_samples=200, seed=seed)
        vocab = set()
        for inp, out in test_data:
            vocab.update(inp + out)
        vocab = ["<pad>"] + sorted(vocab)
        test_ds = SeqDataset(test_data, vocab)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_batch)
        ood_test_sets[n_prim] = (test_ds, test_loader, vocab)

    # --- Helper: train + eval ---
    def train_and_eval(train_data, vocab, n_layers):
        train_ds = SeqDataset(train_data, vocab)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_batch)

        model = DecoderOnlyTransformer(
            vocab_size=len(vocab),
            d_model=128,
            n_layers=n_layers,
            n_heads=4,
            d_ff=256,
            max_len=50,
        ).to(device)

        train_model(model, train_loader, len(vocab), epochs=epochs, lr=lr)

        # Evaluate on OOD sets
        accs = {}
        for n_prim, (test_ds, test_loader, test_vocab) in ood_test_sets.items():
            if vocab != test_vocab:  # align vocab
                test_ds = SeqDataset([(inp, out) for inp, out in test_ds.data], vocab)
                test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_batch)
            accs[n_prim] = evaluate_model(model, test_loader)

        # Homomorphism errors
        train_pairs_mod, val_pairs_mod, train_pairs_seq, val_pairs_seq = build_he_pairs(model, train_ds, device)
        he_results = compute_layerwise_he(
            layers=n_layers,
            d=model.d_model,
            train_pairs_mod=train_pairs_mod,
            val_pairs_mod=val_pairs_mod,
            train_pairs_seq=train_pairs_seq,
            val_pairs_seq=val_pairs_seq,
            operator_kinds=['linear', 'bilinear', 'mlp'],
            epochs=50,
            device=device,
        )

        he_modifier = {layer: np.mean([m['mse'] for m in op_dict.values()])
                       for layer, op_dict in he_results['modifier'].items()}
        he_sequence = {layer: np.mean([m['mse'] for m in op_dict.values()])
                       for layer, op_dict in he_results['sequence'].items()}

        return accs, he_modifier, he_sequence

    # --- 2. Experiment 1: Varying number of layers ---
    print("Experiment 1: Varying number of layers...")
    layer_range = range(1, 11)
    train_data_base = generate_dataset(num_primitives=2, num_noise=3, max_samples=1000, seed=seed)
    vocab_base = set()
    for inp, out in train_data_base:
        vocab_base.update(inp + out)
    vocab_base = ["<pad>"] + sorted(vocab_base)

    base_train_data = generate_dataset(num_primitives=2, num_noise=0, max_samples=1000, seed=seed)

    layer_accs, layer_he_mod, layer_he_seq = {}, {}, {}
    for n_layers in layer_range:
        accs, he_mod, he_seq = train_and_eval(base_train_data, vocab_base, n_layers)
        layer_accs[n_layers] = accs
        layer_he_mod[n_layers] = he_mod
        layer_he_seq[n_layers] = he_seq

    # --- 3. Experiment 2: Varying data sparsity ---
    print("Experiment 2: Varying data sparsity...")
    sparsity_levels = [1, 2, 3, 4]
    sparsity_accs, sparsity_he_mod, sparsity_he_seq = {}, {}, {}
    for n_prim_train in sparsity_levels:
        train_data = []
        for i in range(1, n_prim_train + 1):
            train_data += generate_dataset(num_primitives=i, num_noise=0, max_samples=1000, seed=seed)
        accs, he_mod, he_seq = train_and_eval(train_data, vocab_base, n_layers=4)
        sparsity_accs[n_prim_train] = accs
        sparsity_he_mod[n_prim_train] = he_mod
        sparsity_he_seq[n_prim_train] = he_seq

    # --- 4. Experiment 3: Varying noise scale ---
    print("Experiment 3: Varying noise scale...")
    noise_levels = range(0, 16)
    noise_accs, noise_he_mod, noise_he_seq = {}, {}, {}
    for n_noise in noise_levels:
        train_data = generate_dataset(num_primitives=2, num_noise=n_noise, max_samples=1000, seed=seed)
        accs, he_mod, he_seq = train_and_eval(train_data, vocab_base, n_layers=4)
        noise_accs[n_noise] = accs
        noise_he_mod[n_noise] = he_mod
        noise_he_seq[n_noise] = he_seq

    return {
        "layer": (layer_accs, layer_he_mod, layer_he_seq),
        "sparsity": (sparsity_accs, sparsity_he_mod, sparsity_he_seq),
        "noise": (noise_accs, noise_he_mod, noise_he_seq),
    }


# --- Plotting helpers with error bars ---
def plot_accuracy(exp_dicts, xlabel, title):
    plt.figure(figsize=(8,5))
    keys = sorted(exp_dicts[0].keys())
    for k in keys:
        x = sorted(exp_dicts[0][k].keys())
        ys = [[d[k][n] for d in exp_dicts] for n in x]  # collect across seeds
        y_mean = [np.mean(vals) for vals in ys]
        y_std = [np.std(vals) for vals in ys]
        color = plt.cm.viridis(k / len(keys))
        # plt.errorbar(x, y_mean, yerr=y_std, marker='o', label=str(k), color=color)
        # use a small horizontal bar at the end of error bars
        plt.errorbar(x, y_mean, yerr=y_std, marker='o', label=str(k), color=color, capsize=5, elinewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel("OOD Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{title.replace(' ', '_')}_accuracy.pdf")


def plot_he(exp_dicts, xlabel, title, he_type="Modifier"):
    plt.figure(figsize=(8,5))
    keys = sorted(exp_dicts[0].keys())
    for k in keys:
        x = sorted(exp_dicts[0][k].keys())
        ys = [[d[k][n] for d in exp_dicts] for n in x]
        y_mean = [np.mean(vals) for vals in ys]
        y_std = [np.std(vals) for vals in ys]
        color = plt.cm.viridis(k / len(keys))
        plt.errorbar(x, y_mean, yerr=y_std, marker='o', label=str(k), color=color, capsize=5, elinewidth=2)
    plt.xlabel("Layer")
    plt.ylabel("MSE")
    plt.title(f"{title} ({he_type} HE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{title.replace(' ', '_')}_{he_type}_he.pdf")


def plot_experiment_triplet(acc_dicts, he_mod_dicts, he_seq_dicts, xlabel, title):
    """
    Plot accuracy, modifier HE, and sequence HE side by side with a shared legend.

    acc_dicts, he_mod_dicts, he_seq_dicts : list of dicts (one per seed)
        Each dict should be of form {k: {x: val}}.
    xlabel : str
        Label for x-axis.
    title : str
        Title of the experiment.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    # keys correspond to different experimental conditions
    keys = sorted(acc_dicts[0].keys())
    cmap = plt.cm.viridis
    colors = {k: cmap(i / len(keys)) for i, k in enumerate(keys)}

    # Helper function
    def plot_metric(ax, exp_dicts, ylabel, panel_title):
        for k in keys:
            x = sorted(exp_dicts[0][k].keys())
            ys = [[d[k][n] for d in exp_dicts] for n in x]
            y_mean = [np.mean(vals) for vals in ys]
            y_std = [np.std(vals) for vals in ys]
            ax.errorbar(
                x, y_mean, yerr=y_std,
                marker='o', label=str(k),
                color=colors[k], capsize=5, elinewidth=2
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.grid(True)

    # Plot accuracy
    plot_metric(axes[0], acc_dicts, ylabel="OOD Accuracy", panel_title="Accuracy")

    # Plot modifier HE
    plot_metric(axes[1], he_mod_dicts, ylabel="MSE", panel_title="Modifier HE")

    # Plot sequence HE
    plot_metric(axes[2], he_seq_dicts, ylabel="MSE", panel_title="Sequence HE")

    # Shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(keys), bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # leave space for legend + title
    plt.savefig(f"figures/{title.replace(' ', '_')}_triplet.pdf")
    plt.close(fig)



if __name__ == "__main__":
    # run_experiment()


    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    from gen_data import generate_dataset
    from transformers import DecoderOnlyTransformer
    import numpy as np
    import json
    from datetime import datetime


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = [42, 43, 44, 45, 46]
    results = [run_all_experiments(seed=s) for s in seeds]

    # save results to a json file
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = f"experiment_results_{date_time}.json"
    with open(results_path, "w") as f:
        json.dump(results, f)

    layer_accs = [r["layer"][0] for r in results]
    layer_he_mod = [r["layer"][1] for r in results]
    layer_he_seq = [r["layer"][2] for r in results]

    sparsity_accs = [r["sparsity"][0] for r in results]
    sparsity_he_mod = [r["sparsity"][1] for r in results]
    sparsity_he_seq = [r["sparsity"][2] for r in results]

    noise_accs = [r["noise"][0] for r in results]
    noise_he_mod = [r["noise"][1] for r in results]
    noise_he_seq = [r["noise"][2] for r in results]

    plot_accuracy(layer_accs, xlabel="Test set primitives", title="Varying number of layers")
    plot_he(layer_he_mod, xlabel="Layer", title="Varying number of layers", he_type="Modifier")
    plot_he(layer_he_seq, xlabel="Layer", title="Varying number of layers", he_type="Sequence")

    plot_accuracy(sparsity_accs, xlabel="Test set primitives", title="Varying data sparsity")
    plot_he(sparsity_he_mod, xlabel="Layer", title="Varying data sparsity", he_type="Modifier")
    plot_he(sparsity_he_seq, xlabel="Layer", title="Varying data sparsity", he_type="Sequence")

    plot_accuracy(noise_accs, xlabel="Test set primitives", title="Varying noise scale")
    plot_he(noise_he_mod, xlabel="Layer", title="Varying noise scale", he_type="Modifier")
    plot_he(noise_he_seq, xlabel="Layer", title="Varying noise scale", he_type="Sequence")

    plot_experiment_triplet(layer_accs, layer_he_mod, layer_he_seq, xlabel="Layer", title="Layer Ablation")
    plot_experiment_triplet(sparsity_accs, sparsity_he_mod, sparsity_he_seq, xlabel="Sparsity", title="Sparsity Ablation")
    plot_experiment_triplet(noise_accs, noise_he_mod, noise_he_seq, xlabel="Noise Level", title="Noise Ablation")


    # # --- 1. Generate OOD test sets (primitives 5 to 8) ---
    # print("Generating OOD test sets...")
    # ood_test_sets = {}
    # for n_prim in range(5, 9):
    #     test_data = generate_dataset(num_primitives=n_prim, num_noise=0, max_samples=200, seed=42)
    #     vocab = set()
    #     for inp, out in test_data:
    #         vocab.update(inp + out)
    #     vocab = ["<pad>"] + sorted(vocab)
    #     test_ds = SeqDataset(test_data, vocab)
    #     test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_batch)
    #     ood_test_sets[n_prim] = (test_ds, test_loader, vocab)
    #     print(f"Test set with {n_prim} primitives: {len(test_data)} samples, vocab size {len(vocab)}")

    # # --- Helper function to train and evaluate a model ---
    # def train_and_eval(train_data, vocab, n_layers, epochs=15, lr=1e-3):
    #     train_ds = SeqDataset(train_data, vocab)
    #     train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_batch)
        
    #     model = DecoderOnlyTransformer(
    #         vocab_size=len(vocab),
    #         d_model=128,
    #         n_layers=n_layers,
    #         n_heads=4,
    #         d_ff=256,
    #         max_len=50,
    #     ).to(device)
        
    #     train_model(model, train_loader, len(vocab), epochs=epochs, lr=lr)
        
    #     accs = {}
    #     for n_prim, (test_ds, test_loader, test_vocab) in ood_test_sets.items():
    #         # Ensure vocab alignment
    #         if vocab != test_vocab:
    #             # Rebuild test dataset with train vocab
    #             test_ds = SeqDataset([(inp, out) for inp, out in test_ds.data], vocab)
    #             test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_batch)
    #         acc = evaluate_model(model, test_loader)
    #         accs[n_prim] = acc

    #     # Homomorphism error
    #     train_pairs_mod, val_pairs_mod, train_pairs_seq, val_pairs_seq = build_he_pairs(model, train_ds, device)
    #     he_results = compute_layerwise_he(
    #         layers=n_layers,
    #         d=model.d_model,
    #         train_pairs_mod=train_pairs_mod,
    #         val_pairs_mod=val_pairs_mod,
    #         train_pairs_seq=train_pairs_seq,
    #         val_pairs_seq=val_pairs_seq,
    #         operator_kinds=['linear', 'bilinear', 'mlp'],
    #         epochs=50,
    #         device=device,
    #     )

    #     # Average MSE across operators
    #     he_modifier = {layer: sum(m['mse'] for m in op_dict.values()) / len(op_dict)
    #                 for layer, op_dict in he_results['modifier'].items()}
    #     he_sequence = {layer: sum(m['mse'] for m in op_dict.values()) / len(op_dict)
    #                 for layer, op_dict in he_results['sequence'].items()}

    #     return accs, he_modifier, he_sequence

    # seeds = [42, 43, 44, 45, 46]

    # # --- 2. Experiment 1: Varying number of layers (1 to 10) ---
    # layer_range = range(1, 11)
    # train_data_base = generate_dataset(num_primitives=2, num_noise=3, max_samples=1000, seed=42)
    # vocab_base = set()
    # for inp, out in train_data_base:
    #     vocab_base.update(inp + out)
    # vocab_base = ["<pad>"] + sorted(vocab_base)

    # train_data_base = generate_dataset(num_primitives=2, num_noise=0, max_samples=1000, seed=42)

    # layer_accs, layer_he_mod, layer_he_seq = {}, {}, {}
    # for n_layers in layer_range:
    #     print(f"\n=== Training model with {n_layers} layers ===")
    #     accs, he_mod, he_seq = train_and_eval(train_data_base, vocab_base, n_layers)
    #     layer_accs[n_layers] = accs
    #     layer_he_mod[n_layers] = he_mod
    #     layer_he_seq[n_layers] = he_seq

    # # --- 3. Experiment 2: Varying data sparsity ---
    # # Data sparsity: add primitives 1->2->3->4 in train set
    # sparsity_levels = [1, 2, 3, 4]
    # sparsity_accs, sparsity_he_mod, sparsity_he_seq = {}, {}, {}
    # for n_prim_train in sparsity_levels:
    #     train_data = []
    #     for i in range(1, n_prim_train + 1):
    #         train_data += generate_dataset(num_primitives=i, num_noise=0, max_samples=1000, seed=42)
    #     print(f"\n=== Training with train primitives up to {n_prim_train} ===")
    #     accs, he_mod, he_seq = train_and_eval(train_data, vocab_base, n_layers=4)
    #     sparsity_accs[n_prim_train] = accs
    #     sparsity_he_mod[n_prim_train] = he_mod
    #     sparsity_he_seq[n_prim_train] = he_seq

    # # --- 4. Experiment 3: Varying noise scale ---
    # noise_levels = range(0, 11)
    # noise_accs, noise_he_mod, noise_he_seq = {}, {}, {}
    # for n_noise in noise_levels:
    #     train_data = generate_dataset(num_primitives=2, num_noise=n_noise, max_samples=1000, seed=42)
    #     print(f"\n=== Training with noise tokens {n_noise} ===")
    #     accs, he_mod, he_seq = train_and_eval(train_data, vocab_base, n_layers=4)
    #     noise_accs[n_noise] = accs
    #     noise_he_mod[n_noise] = he_mod
    #     noise_he_seq[n_noise] = he_seq


    # # --- 5. Plot accuracy vs. number of primitives in test set ---
    # def plot_accuracy(exp_dict, xlabel, title):
    #     plt.figure(figsize=(8,5))
    #     for k, accs in exp_dict.items():
    #         # use color gradient for each line in the plot
    #         color = plt.cm.viridis(k / len(exp_dict))
    #         x = sorted(accs.keys())
    #         y = [accs[n] for n in x]
    #         plt.plot(x, y, marker='o', label=str(k), color=color)
    #     plt.xlabel(xlabel)
    #     plt.ylabel("OOD Accuracy")
    #     plt.title(title)
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     # plt.show()
    #     # save figure
    #     plt.savefig(f"figures/{title.replace(' ', '_')}_accuracy.pdf")

    # def plot_he(exp_he, xlabel, title, he_type="Modifier"):
    #     plt.figure(figsize=(8,5))
    #     for k, he_dict in exp_he.items():
    #         color = plt.cm.viridis(k / len(exp_he))
    #         x = sorted(he_dict.keys())
    #         y = [he_dict[n] for n in x]
    #         plt.plot(x, y, marker='o', label=str(k), color=color)
    #     plt.xlabel("Layer")
    #     plt.ylabel("MSE")
    #     plt.title(f"{title} ({he_type} HE)")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     # plt.show()
    #     # save figure
    #     plt.savefig(f"figures/{title.replace(' ', '_')}_{he_type}_he.pdf")


    # # Plot results
    # plot_accuracy(layer_accs, xlabel="Test set primitives", title="Varying number of layers")
    # plot_he(layer_he_mod, xlabel="Layer", title="Varying number of layers", he_type="Modifier")
    # plot_he(layer_he_seq, xlabel="Layer", title="Varying number of layers", he_type="Sequence")

    # plot_accuracy(sparsity_accs, xlabel="Test set primitives", title="Varying training data sparsity")
    # plot_he(sparsity_he_mod, xlabel="Layer", title="Varying training data sparsity", he_type="Modifier")
    # plot_he(sparsity_he_seq, xlabel="Layer", title="Varying training data sparsity", he_type="Sequence")

    # plot_accuracy(noise_accs, xlabel="Test set primitives", title="Varying noise tokens in training data")
    # plot_he(noise_he_mod, xlabel="Layer", title="Varying noise tokens", he_type="Modifier")
    # plot_he(noise_he_seq, xlabel="Layer", title="Varying noise tokens", he_type="Sequence")
