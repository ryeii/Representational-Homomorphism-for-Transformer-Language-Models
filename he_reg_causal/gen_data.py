import random

PRIMITIVES = ["walk", "jump", "look", "turn"]
OPERATORS = ["twice", "thrice"]
NOISE = ["foo", "bar", "baz"]

def apply_operator(primitive, op):
    if op is None:
        return [primitive.upper()]
    elif op == "twice":
        return [primitive.upper()] * 2
    elif op == "thrice":
        return [primitive.upper()] * 3
    else:
        raise ValueError(f"Unknown operator {op}")

def generate_expression(prims, ops, noise_tokens=0):
    expr_parts = []
    for i, (p, o) in enumerate(zip(prims, ops)):
        expr_parts.append(p)
        if o is not None:
            expr_parts.append(o)
        if i < len(prims) - 1:
            expr_parts.append("then")
    for _ in range(noise_tokens):
        pos = random.randint(0, len(expr_parts))
        expr_parts.insert(pos, random.choice(NOISE))
    return expr_parts

def generate_output(prims, ops):
    out = []
    for p, o in zip(prims, ops):
        out.extend(apply_operator(p, o))
    return out

def generate_dataset(num_primitives, num_noise, max_samples=1000, seed=0):
    """
    Generate dataset of (token list, output token list) pairs.
    Uses on-the-fly sampling to avoid combinatorial explosion.
    """
    random.seed(seed)
    dataset = set()
    attempts = 0
    max_attempts = max_samples * 10  # avoid infinite loop

    while len(dataset) < max_samples and attempts < max_attempts:
        prims = [random.choice(PRIMITIVES) for _ in range(num_primitives)]
        ops = [random.choice([None] + OPERATORS) for _ in range(num_primitives)]
        expr = generate_expression(prims, ops, noise_tokens=num_noise)
        out = generate_output(prims, ops)
        key = (tuple(expr), tuple(out))
        if key not in dataset:
            dataset.add(key)
        attempts += 1

    return [(list(expr), list(out)) for expr, out in dataset]



# --- Example usage ---
# dataset = generate_dataset(num_primitives=1, num_noise=0, max_samples=1000, seed=42)
# for expr, out in dataset:
#     print(expr, "=>", out)
