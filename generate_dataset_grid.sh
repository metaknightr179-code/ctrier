#!/bin/bash
# Extended batch script to generate many Kuairec dataset variants
# 
# This creates a grid of datasets with varying:
#   - dedup_strategy: highest_watch_ratio, first_appearance
#   - use_avg_watch_ratio: true, false
#   - min_watch_ratio: 0.0, 0.05, 0.1, 0.2
#   - min_item_users: 0 (none), 5, 10

INPUT_DIR="./KuaiRec/data"
OUTPUT_BASE="./KuaiRec_variants_grid"
SCRIPT="convert_kuairec_to_yelp.py"

# Check if input exists
if [ ! -d "$INPUT_DIR" ] || [ ! -f "$INPUT_DIR/big_matrix.csv" ]; then
    echo "Error: KuaiRec dataset not found."
    echo "Please download first:"
    echo "  wget https://zenodo.org/records/18164998/files/KuaiRec.zip"
    echo "  unzip KuaiRec.zip"
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

# Arrays of parameters to try
DEDUP_STRATEGIES=("highest_watch_ratio" "first_appearance")
AVG_OPTIONS=("" "--use_avg_watch_ratio")
WATCH_RATIOS=(0.0 0.05 0.1 0.2)
MIN_ITEM_USERS=(0 5 10)

total=0
for strategy in "${DEDUP_STRATEGIES[@]}"; do
    for avg_flag in "${AVG_OPTIONS[@]}"; do
        for wr in "${WATCH_RATIOS[@]}"; do
            for min_users in "${MIN_ITEM_USERS[@]}"; do
                total=$((total + 1))
            done
        done
    done
done

echo "=========================================="
echo "Generating $total Kuairec Dataset Variants"
echo "=========================================="
echo ""

count=0
for strategy in "${DEDUP_STRATEGIES[@]}"; do
    for avg_flag in "${AVG_OPTIONS[@]}"; do
        for wr in "${WATCH_RATIOS[@]}"; do
            for min_users in "${MIN_ITEM_USERS[@]}"; do
                count=$((count + 1))
                
                # Build variant name
                if [ "$strategy" = "highest_watch_ratio" ]; then
                    strat_short="high"
                else
                    strat_short="first"
                fi
                
                if [ -n "$avg_flag" ]; then
                    avg_short="avg"
                else
                    avg_short="ind"
                fi
                
                variant_name="kuairec_${strat_short}_${avg_short}_wr${wr}_minusers${min_users}"
                output_dir="$OUTPUT_BASE/$variant_name"
                
                echo "[$count/$total] $variant_name"
                echo "      Output: $output_dir"
                
                # Build command
                cmd="python3 $SCRIPT"
                cmd="$cmd --input_dir $INPUT_DIR"
                cmd="$cmd --output_dir $output_dir"
                cmd="$cmd --min_watch_ratio $wr"
                cmd="$cmd --dedup_strategy $strategy"
                cmd="$cmd --min_item_users $min_users"
                if [ -n "$avg_flag" ]; then
                    cmd="$cmd --use_avg_watch_ratio"
                fi
                
                # Run
                eval $cmd 2>&1 | tail -3
                
                echo "      Done."
                echo ""
            done
        done
    done
done

# Summary
echo "=========================================="
echo "All $total datasets generated!"
echo "=========================================="
echo ""
echo "Directory: $OUTPUT_BASE/"
echo ""
echo "Grid layout:"
echo "  - Dedup: highest_watch_ratio (high) / first_appearance (first)"
echo "  - Filter: individual (ind) / average (avg)"
echo "  - Watch ratio threshold: 0.0, 0.05, 0.1, 0.2"
echo "  - Min item users: 0, 5, 10"
echo ""
echo "Quick stats:"
echo "---"

# Print stats for each variant
for dir in "$OUTPUT_BASE"/*/; do
    name=$(basename "$dir")
    train_count=$(wc -l < "$dir/train-v0.txt" 2>/dev/null || echo 0)
    test_count=$(wc -l < "$dir/test-v0.txt" 2>/dev/null || echo 0)
    echo "$name: train=$train_count, test=$test_count"
done | column -t

echo ""
echo "To train TRIER on a specific variant:"
echo "  python3 main_rt.py -tf $OUTPUT_BASE/<variant>/train-v0.txt \\"
echo "                     -vf $OUTPUT_BASE/<variant>/valid-v0.txt \\"
echo "                     -ef $OUTPUT_BASE/<variant>/test-v0.txt \\"
echo "                     -cat $OUTPUT_BASE/<variant>/kuairec_cate.txt \\"
echo "                     -vn $OUTPUT_BASE/<variant>/KuaiRec-random-sample_size=99-seed=4444.txt \\"
echo "                     -en $OUTPUT_BASE/<variant>/KuaiRec-random-sample_size=99-seed=4444.txt \\"
echo "                     -n <item_count> -e 100 -b 64 -l 5e-4 \\"
echo "                     -o ./save_<variant>/ -early_stop -patience 50"