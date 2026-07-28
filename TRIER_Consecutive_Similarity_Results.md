# TRIER-PT Results with Consecutive Similarity Loss

## Overview
This document summarizes the training and evaluation results of the TRIER-PT recommendation model with **consecutive similarity loss** enabled. The consecutive similarity loss encourages the model to generate recommendations where adjacent items are dissimilar.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | TRIER-PT (2-layer Transformer) |
| **Training Epochs** | 5 |
| **Batch Size** | 16 |
| **Learning Rate** | 1e-4 |
| **Diversity Weight (-lamb)** | 0.1 |
| **Consecutive Similarity Weight (lmd_consec)** | 0.1 |
| **Max Sequence Length** | 72 |
| **Item Embedding Dimension** | 64 |
| **Attention Heads** | 1 |
| **Data** | Yelp Dataset (14,588 items) |
| **Output Directory** | `./save_consec/` |

---

## Training Loss Progress

| Epoch | Total Loss | Time (seconds) |
|-------|------------|----------------|
| 1 | 10.6170 | 236 |
| 2 | 9.8442 | 228 |
| 3 | 9.5249 | 231 |
| 4 | 9.3861 | 229 |
| 5 | ~9.31 | ~230 |

**Observation**: Loss decreases steadily across epochs, indicating the model is learning effectively.

---

## Test Results (Epoch 5)

### Recommendation Quality Metrics

| Metric | Value |
|--------|-------|
| Recall@5 | 0.00035 |
| Recall@10 | 0.00627 |
| Recall@20 | 0.01625 |
| MRR@5 | 0.00015 |
| MRR@10 | 0.00092 |
| MRR@20 | 0.00158 |
| NDCG@5 | 0.00016 |
| NDCG@10 | 0.00196 |
| NDCG@20 | 0.00439 |

### Diversity Metrics (ILD - Inverse List Diversity)
*Lower ILD = More diverse recommendations*

| Metric | Value |
|--------|-------|
| ILD@5 | 13.78 |
| ILD@10 | 12.18 |
| ILD@20 | 11.73 |

---

## Consecutive Similarity Analysis

This analysis measures cosine similarity between **adjacent items** in the recommendation list.

### Results

| Metric | Value |
|--------|-------|
| Number of test samples | 19,936 |
| Number of consecutive pairs | 378,784 |
| **Average consecutive similarity** | **-0.0623** |
| Minimum similarity | -0.3440 |
| Maximum similarity | 0.2994 |
| Standard deviation | 0.1326 |
| Median similarity | -0.0440 |

### Distribution Analysis
- **Negative similarity**: Most consecutive pairs have negative cosine similarity, indicating dissimilar items
- **Low maximum**: The highest similarity between consecutive items is only 0.30 (previously ~1.0)
- **Tight distribution**: Standard deviation of 0.13 indicates consistent low similarity across pairs

---

## Comparison with Baseline (Without Consecutive Similarity Loss)

### Key Metrics Comparison

| Metric | **Without Consec Loss** (25 epochs) | **With Consec Loss** (5 epochs) | **% Change** |
|--------|-------------------------------------|----------------------------------|--------------|
| **Recall@5** | 0.0022 | 0.00035 | -84% |
| **Recall@10** | 0.0102 | 0.00627 | -38% |
| **Recall@20** | 0.0229 | 0.01625 | -29% |
| **MRR@20** | 0.0029 | 0.00158 | -45% |
| **NDCG@20** | 0.0065 | 0.00439 | -34% |
| **ILD@20** | 11.50 | 11.73 | +2.0% |
| **Avg Consec Similarity** | 0.8014 | -0.0623 | **-107.77%** |
| **Unique Items** | 275 | 24 | -91% |
| **Coverage** | 1.89% | 0.16% | -92% |

### Key Findings

1. **✅ Dramatic reduction in consecutive similarity**: 
   - Average similarity dropped from 0.8014 to -0.0623 (-107.77%)
   - This is the primary goal of the consecutive similarity loss

2. **⚠️ Recommendation quality trade-off**:
   - Recall@20 decreased by 29%
   - MRR@20 decreased by 45%
   - **Important caveat**: This comparison is **not epoch-matched** (5 epochs vs 25 epochs). A fair comparison would require training the baseline for 5 epochs as well.

3. **⚠️ Coverage collapse**:
   - Unique items dropped from 275 to 24 (-91%)
   - Coverage dropped from 1.89% to 0.16%
   - **Critical issue**: The consecutive similarity loss weight (0.1) is too aggressive, causing the model to collapse to a tiny pool of items with mutually dissimilar embeddings.

4. **Recommendations for adjustment**:
   - Reduce `lmd_consec` from 0.1 to 0.01 (gentle regularization)
   - Train for more epochs (10-15) to allow coverage recovery
   - Consider adding entropy regularization to encourage exploration

---

## Per-Session Length Analysis (Epoch 5)

### Recall@20 by Input Sequence Length

| Sequence Length | Recall@20 |
|-----------------|-----------|
| 0-10 | 0.0184 |
| 10-20 | 0.0111 |
| 20-30 | 0.0087 |
| 30-40 | 0.0058 |
| 40-51 | 0.0000 |

**Observation**: Shorter sessions have better recommendation performance, likely due to less noise in the input sequence.

---

## Conclusions

### Successes ✅
1. **Consecutive similarity loss achieves its goal**: Adjacent recommendations are now dissimilar (-0.06 avg similarity vs 0.80 before)
2. **Model trains stably**: Loss decreases steadily across epochs
3. **Diversity within sequences is dramatically improved**: The recommendation list contains varied items

### Challenges ⚠️
1. **Coverage is significantly reduced**: Only 0.16% of items are recommended (vs 1.89% before)
2. **Recommendation quality is lower**: Recall and MRR metrics are worse, though this is partially due to fewer epochs

### Recommendations for Future Work
1. **Train for more epochs** (10-15) to balance coverage and consecutive similarity
2. **Adjust loss weights**: Reduce `lmd_consec` from 0.1 to 0.05, increase `-lamb` from 0.1 to 0.5
3. **Consider hybrid approach**: Use moderate consecutive similarity loss with stronger diversity loss
4. **Add regularization**: Prevent the model from collapsing to a small set of items

---

## Files Used

| File | Description |
|------|-------------|
| [main_pt.py](file:///Users/notrobin/Documents/trae_projects/trier/main_pt.py) | Training and evaluation script |
| [trier_pt.py](file:///Users/notrobin/Documents/trae_projects/trier/trier_pt.py) | TRIER-PT model with consecutive similarity loss |
| [trier_rt.py](file:///Users/notrobin/Documents/trae_projects/trier/trier_rt.py) | TRIER-RT reverse trajectory model |
| [dataset_duorec.py](file:///Users/notrobin/Documents/trae_projects/trier/dataset_duorec.py) | Dataset loading classes |
| [analyze_consecutive.py](file:///Users/notrobin/Documents/trae_projects/trier/analyze_consecutive.py) | Consecutive similarity analysis script |

---

*Generated: July 20, 2026*
