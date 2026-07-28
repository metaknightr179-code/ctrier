import torch
import numpy as np
import sys
import os
sys.path.insert(0, '.')

from trier_pt import TRIER_PT
from trier_rt import TRIER_RT
from dataset_duorec import TestDataset
from script import get_args

# Parse arguments
args = get_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load item embeddings
item2vec = np.load("./Yelp/yelp_vec.npy")
item2vec = torch.tensor(item2vec)

# Initialize RT model (pre-trained)
rt_model = TRIER_RT(14588, 2, 1, 64, 0.5, 16, args)
rt_model_path = './save/model/duorec-10.pth'
if os.path.exists(rt_model_path):
    rt_model.load_state_dict(torch.load(rt_model_path, map_location=args.device))
for param in rt_model.parameters():
    param.requires_grad = False
rt_model.eval()

# Initialize PT model (2 layers to match saved model)
model = TRIER_PT(14588, 2, 1, 64, 0.5, 16, args)
model.load_state_dict(torch.load('./save_consec/model/duorec-5.pth', map_location=args.device))
model.eval()

# Move to device
if torch.cuda.is_available():
    model.cuda()
    rt_model.cuda()
    item2vec = item2vec.cuda()

# Load test dataset
dataset = TestDataset('./Yelp/test-v0.txt', './Yelp/Yelp-random-sample_size=99-seed=4444.txt', 14588, 72)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

# Calculate consecutive similarity
consec_sims = []
total_rec_lists = []

with torch.no_grad():
    for batch in dataloader:
        input_session_ids, targets, negatives, input_reverse_ids = batch
        
        if torch.cuda.is_available():
            input_session_ids = input_session_ids.cuda()
            input_reverse_ids = input_reverse_ids.cuda()
        
        # Generate recommendations
        output = model.test_forward(input_session_ids, input_reverse_ids, rt_model, False)
        output = torch.matmul(output, model.item_embedding.weight.T)
        _, rec_list = output.log_softmax(-1).topk(k=20, axis=-1)
        
        # Store recommendations
        total_rec_lists.extend(rec_list.cpu().tolist())
        
        # Calculate consecutive similarity
        for i in range(rec_list.shape[0]):
            rec = rec_list[i]
            for j in range(19):
                if rec[j] > 0 and rec[j+1] > 0:
                    vec_i = item2vec[rec[j]]
                    vec_j = item2vec[rec[j+1]]
                    sim = torch.nn.functional.cosine_similarity(vec_i, vec_j, dim=0)
                    consec_sims.append(sim.item())

# Print results
print(f"Number of test samples: {len(total_rec_lists)}")
print(f"Number of consecutive pairs: {len(consec_sims)}")
print(f"\n=== Consecutive Similarity Analysis ===")
print(f"Average consecutive similarity: {np.mean(consec_sims):.4f}")
print(f"Min similarity: {np.min(consec_sims):.4f}")
print(f"Max similarity: {np.max(consec_sims):.4f}")
print(f"Standard deviation: {np.std(consec_sims):.4f}")
print(f"Median similarity: {np.median(consec_sims):.4f}")

# Compare with previous results (from session history: avg 0.8014)
prev_avg = 0.8014
improvement = ((prev_avg - np.mean(consec_sims)) / prev_avg) * 100
print(f"\n=== Comparison with Previous Model ===")
print(f"Previous avg consecutive similarity: {prev_avg:.4f}")
print(f"Current avg consecutive similarity: {np.mean(consec_sims):.4f}")
print(f"Improvement: {improvement:.2f}%")

# Calculate coverage
all_rec_items = set()
for rec in total_rec_lists:
    all_rec_items.update([x for x in rec if x > 0])
print(f"\n=== Coverage ===")
print(f"Unique items recommended: {len(all_rec_items)}")
print(f"Coverage (% of total items): {len(all_rec_items)/14588*100:.2f}%")
