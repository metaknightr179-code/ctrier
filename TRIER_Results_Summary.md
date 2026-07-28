# TRIER Recommendation Model - Results Summary

## Project Overview
This document summarizes the training and evaluation results of the TRIER recommendation model on the Yelp dataset.

---

## Dataset Statistics

| Split | Sessions | Items |
|-------|----------|-------|
| Training | 19,936 | 14,588 |
| Validation | 19,936 | - |
| Test | 19,936 | - |

---

## Training Configuration

### RT Model
- **Epochs**: 10
- **Batch Size**: 64
- **Learning Rate**: Default
- **Architecture**: Transformer with 2 layers, 2 heads, 64 hidden units

### PT Model
- **Epochs**: 25
- **Batch Size**: 16 (slower training to reduce CPU usage)
- **Learning Rate**: 1e-4
- **Diversity Loss Weight**: 0.1
- **Architecture**: Transformer with 2 layers, 2 heads, 64 hidden units

---

## Test Performance (25 Epochs)

### Recommendation Quality Metrics

| Epoch | Recall@5 | Recall@10 | Recall@20 | MRR@20 | NDCG@20 |
|-------|----------|-----------|-----------|--------|---------|
| 1 | 0.0003 | 0.0053 | 0.0136 | 0.0013 | 0.0037 |
| 5 | 0.0003 | 0.0064 | 0.0159 | 0.0015 | 0.0043 |
| 10 | 0.0003 | 0.0061 | 0.0159 | 0.0016 | 0.0043 |
| 15 | 0.0013 | 0.0082 | 0.0184 | 0.0023 | 0.0052 |
| 20 | 0.0026 | 0.0096 | 0.0213 | 0.0027 | 0.0061 |
| 25 | 0.0022 | 0.0102 | 0.0229 | 0.0029 | 0.0065 |

### Improvement Summary (Epoch 1 → Epoch 25)
- **Recall@5**: +619% (0.0003 → 0.0022)
- **Recall@10**: +94% (0.0053 → 0.0102)
- **Recall@20**: +68% (0.0136 → 0.0229)
- **MRR@20**: +123% (0.0013 → 0.0029)
- **NDCG@20**: +76% (0.0037 → 0.0065)

### Diversity Metrics (ILD - Inverse List Diversity)

| Epoch | ILD@5 | ILD@10 | ILD@20 |
|-------|-------|--------|--------|
| 1 | 13.91 | 12.51 | 11.87 |
| 5 | 13.91 | 12.33 | 11.76 |
| 10 | 13.84 | 12.34 | 11.76 |
| 15 | 12.68 | 11.97 | 11.68 |
| 20 | 12.25 | 11.71 | 11.45 |
| 25 | 12.25 | 11.77 | 11.50 |

*Note: Lower ILD indicates higher diversity*

---

## Diversity Analysis

### Before (5 Epochs)

| Metric | Value |
|--------|-------|
| Coverage | 0.19% |
| Unique Items Recommended | 27 |
| Average ILD | 0.576 |
| Items in >80% of samples | 18 |

### After (25 Epochs with Increased Diversity Loss)

| Metric | Value |
|--------|-------|
| Coverage | 1.89% |
| Unique Items Recommended | 275 |
| Average ILD | 0.785 |
| Items in >80% of samples | 0 |

### Change Summary

| Metric | Change | Status |
|--------|--------|--------|
| Coverage | +10x | ✅ Improved |
| Unique Items | +10x | ✅ Improved |
| Popular Item Dominance | Eliminated | ✅ Improved |
| Overall Diversity (ILD) | -36% | ❌ Worsened |

---

## Consecutive Similarity Analysis (After 25 Epochs)

### Key Metrics

| Metric | Value |
|--------|-------|
| Average Consecutive Similarity | 0.8014 |
| Median Consecutive Similarity | 0.8708 |
| Maximum Consecutive Similarity | 0.9984 |
| Minimum Consecutive Similarity | -0.1040 |
| SUM of All Consecutive Similarities | 1522.71 |
| Average Sum per Sample | 15.2271 |

### Example Consecutive Similarities

**Sample 1:**
- Recommendations: [12172, 12485, 7003, 10636, 6433, 12181, 5984, 5701, 12238, 689]
- Consecutive similarities: 0.907 → 0.894 → 0.882 → 0.882 → 0.922 → 0.918 → 0.920 → 0.924 → 0.921
- Sum: 16.999

**Sample 2:**
- Recommendations: [1455, 1003, 10873, 2245, 8384, 22, 1282, 347, 17, 1976]
- Consecutive similarities: 0.998 → 0.998 → 0.994 → 0.987 → 0.930 → 0.991 → 0.990 → 0.984 → 0.980
- Sum: 18.647

---

## Issues Identified

### Critical Issues
1. **High Consecutive Similarity**: Adjacent items in recommendations are very similar (avg 0.80)
2. **Low Overall Diversity**: Items cluster in similar regions of embedding space
3. **Limited Coverage**: Only 1.89% of items explored even after 25 epochs

### Root Causes
1. Diversity loss encourages more items but doesn't ensure they're dissimilar
2. Model still prioritizes popularity over sequential patterns
3. Embedding space may not capture fine-grained item differences

---

## Recommendations for Improvement

1. **Increase Diversity Loss Weight**: Use `-lamb 0.5` or higher
2. **Modify Diversity Loss**: Explicitly penalize consecutive item similarity
3. **Add Popularity Regularization**: Penalize items that appear too frequently
4. **Post-Hoc Re-ranking**: Apply diversity-aware re-ranking after generation
5. **Train for More Epochs**: 50-100 epochs for better pattern learning
6. **Adjust Learning Rate**: Experiment with smaller learning rates (1e-5)

---

## Conclusion

The TRIER model shows promising improvement with more training epochs:

**✅ Positive Results:**
- Recall@20 improved 68% from epoch 1 to epoch 25
- MRR@20 improved 123%
- NDCG@20 improved 76%
- Coverage improved 10x (0.19% → 1.89%)
- Popular item dominance eliminated (no items in >80% of samples)

**❌ Areas for Improvement:**
- Consecutive items remain highly similar (avg 0.80 similarity)
- Overall diversity decreased with more training
- Only 1.89% of available items are explored

**Summary**: The model is learning better recommendation patterns with more training, but the diversity mechanism needs further refinement to ensure truly diverse recommendations.
