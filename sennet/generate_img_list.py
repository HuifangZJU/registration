# -----------------------------
# Generate training and testing pair lists
# -----------------------------

# parameters
output_train = "train.txt"
output_test = "test.txt"
base_path = "/sennet"
max_index = 42  # inclusive

# ---- generate pairs ----
train_pairs = []
test_pairs = []

for i in range(max_index + 1):
    # forward direction
    fwd = f"{base_path}/{i}_0 {base_path}/{i}_1"
    # backward direction
    bwd = f"{base_path}/{i}_1 {base_path}/{i}_0"

    # add to train (both directions)
    train_pairs.append(fwd)
    train_pairs.append(bwd)

    # add to test (only forward direction)
    test_pairs.append(fwd)

# ---- write train.txt ----
with open(output_train, "w") as f:
    for line in train_pairs:
        f.write(line + "\n")

# ---- write test.txt ----
with open(output_test, "w") as f:
    for line in test_pairs:
        f.write(line + "\n")

print(f"✅ Generated {output_train} and {output_test}")
print(f"- train.txt: {len(train_pairs)} pairs")
print(f"- test.txt: {len(test_pairs)} pairs")
