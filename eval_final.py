import torch
import torch.utils.data as Data
import numpy as np
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trier_pt import TRIER_PT
from trier_rt import TRIER_RT
from dataset_duorec import TestDataset
from script import *

args = get_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

item_num = 14588
hidden_unit = 64
head_num = 4
layer_num = 2
dropout_rate = 0.1
batch_size = 64

print("Loading item embeddings...")
item2vec = np.load("./Yelp/yelp_vec.npy")
item2vec = torch.tensor(item2vec)

print("Loading RT model...")
rt_model = TRIER_RT(item_num, 2, head_num, hidden_unit, dropout_rate, batch_size, args)
rt_model_path = './save_hyperopt_final/model/duorec-10.pth'
rt_model.load_state_dict(torch.load(rt_model_path, map_location=args.device))
for param in rt_model.parameters():
    param.requires_grad = False
rt_model.eval()

print("Loading PT model (epoch 100)...")
model = TRIER_PT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)
model.load_state_dict(torch.load('./save_hyperopt_final/model/duorec-100.pth', map_location=args.device))
model.eval()

if torch.cuda.is_available():
    model.cuda()
    rt_model.cuda()
    item2vec = item2vec.cuda()

print("Loading test dataset...")
dataset = TestDataset('./Yelp/test-v0.txt', './Yelp/Yelp-random-sample_size=99-seed=4444.txt', item_num, 50)
dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("Evaluating...")
total_result = []
with torch.no_grad():
    for batch in dataloader:
        input_session_ids, targets, negatives, input_reverse_ids = batch
        if torch.cuda.is_available():
            input_session_ids = input_session_ids.cuda()
            targets = targets.cuda()
            negatives = negatives.cuda()
            input_reverse_ids = input_reverse_ids.cuda()
        
        output = model.test_forward(input_session_ids, input_reverse_ids, rt_model, False)
        output = torch.matmul(output, model.item_embedding.weight.T)
        _, rec_list = output.log_softmax(-1).topk(k=20, axis=-1)
        
        result = evaluate_function_with_full(targets, rec_list, item2vec=item2vec)
        total_result.extend(result)

print("\nFinal Model (Epoch 100) Evaluation Results:")
metrics = {}
metrics['recall@5_f'] = get_metrics_full('recall@5_f', total_result)
metrics['recall@10_f'] = get_metrics_full('recall@10_f', total_result)
metrics['recall@20_f'] = get_metrics_full('recall@20_f', total_result)
metrics['mrr@5_f'] = get_metrics_full('mrr@5_f', total_result)
metrics['mrr@10_f'] = get_metrics_full('mrr@10_f', total_result)
metrics['mrr@20_f'] = get_metrics_full('mrr@20_f', total_result)
metrics['ndcg@5_f'] = get_metrics_full('ndcg@5_f', total_result)
metrics['ndcg@10_f'] = get_metrics_full('ndcg@10_f', total_result)
metrics['ndcg@20_f'] = get_metrics_full('ndcg@20_f', total_result)
metrics['ILD@5'] = get_metrics_full('ILD@5', total_result)
metrics['ILD@10'] = get_metrics_full('ILD@10', total_result)
metrics['ILD@20'] = get_metrics_full('ILD@20', total_result)

print(metrics)

print("\nComparison with Paper Metrics (Yelp):")
print("| Metric | Our Model | Paper | Gap |")
print("|--------|-----------|-------|-----|")
print(f"| NDCG@20 | {metrics['ndcg@20_f']:.4f} | 0.0444 | {metrics['ndcg@20_f'] - 0.0444:.4f} |")
print(f"| NDCG@10 | {metrics['ndcg@10_f']:.4f} | 0.0330 | {metrics['ndcg@10_f'] - 0.0330:.4f} |")
print(f"| NDCG@5  | {metrics['ndcg@5_f']:.4f} | 0.0244 | {metrics['ndcg@5_f'] - 0.0244:.4f} |")
print(f"| HR@20   | {metrics['recall@20_f']:.4f} | 0.1245 | {metrics['recall@20_f'] - 0.1245:.4f} |")
print(f"| HR@10   | {metrics['recall@10_f']:.4f} | 0.0781 | {metrics['recall@10_f'] - 0.0781:.4f} |")
print(f"| HR@5    | {metrics['recall@5_f']:.4f} | 0.0496 | {metrics['recall@5_f'] - 0.0496:.4f} |")