#!/bin/bash
# Batch script to generate all 4 Kuairec dataset variations
# 
# Combinations:
#   1. highest_watch_ratio + individual filtering (DEFAULT)
#   2. highest_watch_ratio + average filtering
#   3. first_appearance + individual filtering
#   4. first_appearance + average filtering

INPUT_DIR="./KuaiRec/data"
OUTPUT_BASE="./KuaiRec_variants"
SCRIPT="convert_kuairec_to_yelp.py"

# Check if input exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR not found."
    echo "Please download Kuairec dataset first:"
    echo "  wget https://zenodo.org/records/18164998/files/KuaiRec.zip"
    echo "  unzip KuaiRec.zip"
    exit 1
fi

# Check if main interaction file exists
if [ ! -f "$INPUT_DIR/big_matrix.csv" ]; then
    echo "Error: big_matrix.csv not found in $INPUT_DIR"
    exit 1
fi

# Create base output directory
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Generating Kuairec Dataset Variants"
echo "=========================================="
echo ""

# -----------------------------------------------
# Variant 1: highest_watch_ratio + individual
# -----------------------------------------------
echo "[1/4] highest_watch_ratio + individual filtering"
echo "      Output: $OUTPUT_BASE/kuairec_highest_individual"
python3 "$SCRIPT" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_BASE/kuairec_highest_individual" \
  --min_watch_ratio 0.1 \
  --dedup_strategy highest_watch_ratio
echo "      Done."
echo ""

# -----------------------------------------------
# Variant 2: highest_watch_ratio + average
# -----------------------------------------------
echo "[2/4] highest_watch_ratio + average filtering"
echo "      Output: $OUTPUT_BASE/kuairec_highest_average"
python3 "$SCRIPT" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_BASE/kuairec_highest_average" \
  --min_watch_ratio 0.1 \
  --use_avg_watch_ratio \
  --dedup_strategy highest_watch_ratio
echo "      Done."
echo ""

# -----------------------------------------------
# Variant 3: first_appearance + individual
# -----------------------------------------------
echo "[3/4] first_appearance + individual filtering"
echo "      Output: $OUTPUT_BASE/kuairec_first_individual"
python3 "$SCRIPT" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_BASE/kuairec_first_individual" \
  --min_watch_ratio 0.1 \
  --dedup_strategy first_appearance
echo "      Done."
echo ""

# -----------------------------------------------
# Variant 4: first_appearance + average
# -----------------------------------------------
echo "[4/4] first_appearance + average filtering"
echo "      Output: $OUTPUT_BASE/kuairec_first_average"
python3 "$SCRIPT" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_BASE/kuairec_first_average" \
  --min_watch_ratio 0.1 \
  --use_avg_watch_ratio \
  --dedup_strategy first_appearance
echo "      Done."
echo ""

# -----------------------------------------------
# Summary
# -----------------------------------------------
echo "=========================================="
echo "All datasets generated!"
echo "=========================================="
echo ""
echo "Directory structure:"
echo "$OUTPUT_BASE/"
echo "├── kuairec_highest_individual/   # Best engagement + individual filter"
echo "├── kuairec_highest_average/      # Best engagement + average filter"
echo "├── kuairec_first_individual/     # First seen + individual filter"
echo "└── kuairec_first_average/        # First seen + average filter"
echo ""
echo "Each variant contains:"
echo "  - train-v0.txt"
echo "  - valid-v0.txt"
echo "  - test-v0.txt"
echo "  - kuairec_cate.txt"
echo "  - KuaiRec-random-sample_size=99-seed=4444.txt"
echo ""

# Print quick stats for each variant
echo "=========================================="
echo "Quick Statistics"
echo "=========================================="
for dir in "$OUTPUT_BASE"/*/; do
    name=$(basename "$dir")
    train_count=$(wc -l < "$dir/train-v0.txt" 2>/dev/null || echo 0)
    test_count=$(wc -l < "$dir/test-v0.txt" 2>/dev/null || echo 0)
    echo "$name: train=$train_count users, test=$test_count users"
done