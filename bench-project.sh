#!/usr/bin/env bash
set -euo pipefail

#
# Benchmark Annif operations (train -> eval) for ONE project, per job count.
#
# Usage:
#   ./bench_project.sh <project-id> <train-data> <suggest-eval-data>
#
# Example:
#   ./bench_project.sh yso-tfidf-en \
#     ../Annif-tutorial/data-sets/yso-nlf/yso-finna.tsv.gz \
#     ../Annif-tutorial/data-sets/yso-nlf/docs/test/
#

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <project-id> <train-data-pattern> <suggest-eval-data-pattern>"
  exit 1
fi

# Activate Annif virtual environment
source Annif/.venv/bin/activate

PROJECT="$1"
TRAIN_DATA_PATTERN="$2"
SUGGEST_EVAL_DATA_PATTERN="$3"

JOBS=("6" "1")

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

for j in "${JOBS[@]}"; do
  echo "=== Job setting: -j$j ==="

  # ----- TRAIN -----
  TRAIN_OUTFILE="$OUTDIR/train-j${j}.txt"
  echo "Running: annif train -j$j $PROJECT $TRAIN_DATA_PATTERN"
  echo "Writing to: $TRAIN_OUTFILE"

  /usr/bin/time -v annif train -j"$j" "$PROJECT" $TRAIN_DATA_PATTERN \
    &> "$TRAIN_OUTFILE"

  # After training, compute model size
  if [[ -d "$MODEL_DIR" ]]; then
    MODEL_SIZE_BYTES=$(du -sb "$MODEL_DIR" | awk '{print $1}')
    {
      echo ""
      echo "=== Model size after training ==="
      echo "Model directory: $MODEL_DIR"
      echo "Model size (bytes): $MODEL_SIZE_BYTES"
    } >> "$TRAIN_OUTFILE"
    echo "✔ Recorded model size"
  else
    echo "Warning: model directory not found: $MODEL_DIR" | tee -a "$TRAIN_OUTFILE"
  fi

  echo "✔ Completed train -j$j"
  echo

  # ----- EVAL -----
  EVAL_OUTFILE="$OUTDIR/eval-j${j}.txt"
  echo "Running: annif eval -j$j $PROJECT $SUGGEST_EVAL_DATA_PATTERN"
  echo "Writing to: $EVAL_OUTFILE"

  /usr/bin/time -v annif eval -j"$j" "$PROJECT" $SUGGEST_EVAL_DATA_PATTERN \
    &> "$EVAL_OUTFILE"

  echo "✔ Completed eval -j$j"
  echo
done

echo "=== All benchmarks complete ==="
echo "Results stored in: $OUTDIR"
