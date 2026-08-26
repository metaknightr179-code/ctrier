#!/usr/bin/env python3
"""Demo: calculate ILD using the ILD_tensor method from script.py"""
import numpy as np
import torch

# Load item2vec (multi-hot category vectors)
item2vec = torch.tensor(np.load('./KuaiRec_variants/kuairec_vec.npy'))
print(f"item2vec shape: {item2vec.shape}")
print()

# Sample recommendation list: 5 items
sample = torch.tensor([[1, 2, 100, 3, 50]])

# Show their category vectors
for item_id in sample[0]:
    cats = np.where(item2vec[item_id].numpy() == 1.0)[0]
    print(f"Item {item_id:4d}: categories {cats}  vector={item2vec[item_id].numpy().astype(int)}")

print()

# Replicate ILD_tensor logic from script.py line 110-117
topk = sample.shape[1]  # 5
gen_vector = item2vec[sample]  # [1, 5, 31]
print(f"gen_vector shape: {gen_vector.shape}")

ILD_list = torch.cdist(gen_vector, gen_vector) / (topk * (topk - 1))
print(f"ILD_list shape: {ILD_list.shape}")
print()

# Print the pairwise distance matrix
dist_matrix = torch.cdist(gen_vector, gen_vector)[0]  # [5, 5]
print("Pairwise Euclidean distance matrix:")
items = sample[0].tolist()
header = "         " + "  ".join(f"item{i:>4d}" for i in items)
print(header)
for i, item_i in enumerate(items):
    row = "  ".join(f"{dist_matrix[i][j].item():7.4f}" for j in range(len(items)))
    print(f"item{item_i:>4d}  {row}")

print()
# Final ILD
ILD = ILD_list.sum(-1).sum(-1)
off_diag_sum = (dist_matrix.sum() - torch.diagonal(dist_matrix).sum()).item()
print(f"topk = {topk}")
print(f"Normalization factor: topk*(topk-1) = {topk*(topk-1)}")
print(f"Sum of ALL distances (including diagonal zeros): {dist_matrix.sum().item():.4f}")
print(f"Sum of off-diagonal distances:                  {off_diag_sum:.4f}")
print(f"ILD = {ILD.item():.6f}")
