#!/usr/bin/env bash
set -euo pipefail

#
# Benchmark Annif operations (train/suggest/eval) for ONE project.
#
# Usage:
#   ./bench_project.sh <project-id> <train-data> <suggest-eval-data>
#
# Example:
#   ./bench-project.sh yso-tfidf-en \
#     ../Annif-tutorial/data-sets/yso-nlf/yso-finna.tsv.gz
#     ../Annif-tutorial/data-sets/yso-nlf/docs/test/
#

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <project-id> <train-data-pattern> <suggest-eval-data-pattern>"
  exit 1
fi

source Annif/.venv/bin/activate

PROJECT="$1"
TRAIN_DATA_PATTERN="$2"
SUGGEST_EVAL_DATA_PATTERN="$3"

OPERATIONS=("train" "eval")
JOBS=("1" "6")

TIMESTAMP=$(date +"%Y%m%d-%H%M")
OUTDIR="benchmarks/${TIMESTAMP}_${PROJECT}"
mkdir -p "$OUTDIR"

echo "=== Benchmarking project: $PROJECT ==="
echo "Train data:   $TRAIN_DATA_PATTERN"
echo "Suggest/Eval: $SUGGEST_EVAL_DATA_PATTERN"
echo "Results will be saved under: $OUTDIR"
echo

# Determine Annif model directory
MODEL_DIR="data/projects/$PROJECT"

for op in "${OPERATIONS[@]}"; do
  for j in "${JOBS[@]}"; do

    OUTFILE="$OUTDIR/${op}-j${j}.txt"

    # Choose correct dataset per operation
    if [[ "$op" == "train" ]]; then
      DATA="$TRAIN_DATA_PATTERN"
    else
      DATA="$SUGGEST_EVAL_DATA_PATTERN"
    fi

    echo "Running: annif $op -j$j $PROJECT $DATA"
    echo "Writing to: $OUTFILE"

    /usr/bin/time -v annif "$op" -j"$j" "$PROJECT" $DATA \
      &> "$OUTFILE"

    # After training, compute model size
    if [[ "$op" == "train" ]]; then
      if [[ -d "$MODEL_DIR" ]]; then
        MODEL_SIZE_BYTES=$(du -sb "$MODEL_DIR" | awk '{print $1}')
        echo "" >> "$OUTFILE"
        echo "=== Model size after training ===" >> "$OUTFILE"
        echo "Model directory: $MODEL_DIR" >> "$OUTFILE"
        echo "Model size (bytes): $MODEL_SIZE_BYTES" >> "$OUTFILE"
        echo "✔ Recorded model size"
      else
        echo "Warning: model directory not found: $MODEL_DIR" | tee -a "$OUTFILE"
      fi
    fi

    echo "✔ Completed $op -j$j"
    echo
  done
done

echo "=== All benchmarks complete ==="
echo "Results stored in: $OUTDIR"
