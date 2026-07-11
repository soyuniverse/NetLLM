#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/workspace/NetLLM"
SOURCE_ROOT="/workspace/NetLLM-source"
VP_ROOT="$SOURCE_ROOT/viewport_prediction"
PYTHON_BIN="/venv/vp_netllm/bin/python"
OUTPUT_ROOT="$PROJECT_ROOT/experiments/vp/phase1_5b_runtime"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$OUTPUT_ROOT/logs"
MODEL_ROOT="$OUTPUT_ROOT/models"
RESULT_ROOT="$OUTPUT_ROOT/results"
LOG_PATH="$LOG_DIR/${TIMESTAMP}_vp_regression_cpu_jin2022.log"
BEFORE_PATH="$LOG_DIR/${TIMESTAMP}_upstream_before.txt"
AFTER_PATH="$LOG_DIR/${TIMESTAMP}_upstream_after.txt"
STATUS_PATH="$LOG_DIR/${TIMESTAMP}_run_status.txt"
SUMMARY_PATH="$RESULT_ROOT/regression/Jin2022/5Hz/result_his_10_fut_20_ss_15_epochs_40_bs_64_lr_0.0002_seed_1.csv"
DETAIL_PATH="$RESULT_ROOT/regression/Jin2022/5Hz/details_his_10_fut_20_ss_15_epochs_40_bs_64_lr_0.0002_seed_1.csv"
EXPECTED_MAE="19.473604202270508"
EXPECTED_RMSE="77.839111328125"

mkdir -p "$LOG_DIR"

record_integrity() {
    local destination="$1"
    {
        echo "commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
        echo "status_begin"
        git -C "$SOURCE_ROOT" status --porcelain=v2 --untracked-files=all
        echo "status_end"
        echo "diff_begin"
        git -C "$SOURCE_ROOT" diff --name-status HEAD
        echo "diff_end"
    } > "$destination"
}

record_integrity "$BEFORE_PATH"
before_status="$(git -C "$SOURCE_ROOT" status --porcelain=v2 --untracked-files=all)"
before_diff="$(git -C "$SOURCE_ROOT" diff --name-status HEAD)"

if [[ -n "$before_status" || -n "$before_diff" ]]; then
    echo "Phase 1.5B aborted: upstream is not clean before execution." | tee "$LOG_PATH"
    exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Phase 1.5B aborted: Python executable not found: $PYTHON_BIN" | tee "$LOG_PATH"
    exit 2
fi

if [[ -e "$SUMMARY_PATH" || -e "$DETAIL_PATH" ]]; then
    echo "Phase 1.5B aborted: result output already exists; nothing was overwritten." | tee "$LOG_PATH"
    exit 2
fi

unset PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$VP_ROOT"
export PHASE1_5B_MODEL_ROOT="$MODEL_ROOT"
export PHASE1_5B_RESULT_ROOT="$RESULT_ROOT"

start_epoch="$(date +%s)"

cd "$VP_ROOT" || exit 2
"$PYTHON_BIN" -B - <<'PY' 2>&1 | tee "$LOG_PATH"
import os
from types import SimpleNamespace

import run_baseline as entry

entry.cfg.models_dir = os.environ["PHASE1_5B_MODEL_ROOT"]
entry.cfg.results_dir = os.environ["PHASE1_5B_RESULT_ROOT"]

original_create_dataset = entry.create_dataset
original_record = entry.ResultNotebook.record
audit_state = {"shape_printed": False}


def audited_create_dataset(*args, **kwargs):
    datasets = original_create_dataset(*args, **kwargs)
    include = kwargs.get("include", ("train", "valid", "test"))
    dataset_name = args[0] if args else kwargs.get("dataset")
    print(f"PHASE1_5B_DATASET_NAME={dataset_name}")
    print(f"PHASE1_5B_DATASET_ROOT={entry.cfg.dataset[dataset_name]}")
    print(f"PHASE1_5B_DATASET_INCLUDE={include}")
    print(f"PHASE1_5B_DATASET_SPLIT_LENGTHS={[len(dataset) for dataset in datasets]}")
    return datasets


def audited_record(self, prediction, ground_truth, videos, users, timesteps):
    if not audit_state["shape_printed"]:
        print(f"PHASE1_5B_FIRST_BATCH_PREDICTION_SHAPE={tuple(prediction.shape)}")
        print(f"PHASE1_5B_FIRST_BATCH_GROUND_TRUTH_SHAPE={tuple(ground_truth.shape)}")
        print(f"PHASE1_5B_FIRST_BATCH_VIDEO_SHAPE={tuple(videos.shape)}")
        audit_state["shape_printed"] = True
    return original_record(self, prediction, ground_truth, videos, users, timesteps)


entry.create_dataset = audited_create_dataset
entry.ResultNotebook.record = audited_record

args = SimpleNamespace(
    train=False,
    test=True,
    device="cpu",
    model="regression",
    compile=False,
    resume=False,
    train_dataset="Jin2022",
    test_dataset="Jin2022",
    his_window=10,
    fut_window=20,
    trim_head=30,
    trim_tail=60,
    dataset_frequency=5,
    sample_step=15,
    epochs=40,
    epochs_per_valid=3,
    lr=2e-4,
    weight_decay=1e-5,
    bs=64,
    model_path=None,
    seed=1,
)

print(f"PHASE1_5B_PYTHON={os.sys.executable}")
print(f"PHASE1_5B_MODEL_ROOT={entry.cfg.models_dir}")
print(f"PHASE1_5B_RESULT_ROOT={entry.cfg.results_dir}")
print(f"PHASE1_5B_ARGS={args}")
entry.run(args)
PY
baseline_status=${PIPESTATUS[0]}

end_epoch="$(date +%s)"
elapsed_seconds=$((end_epoch - start_epoch))

record_integrity "$AFTER_PATH"
after_status="$(git -C "$SOURCE_ROOT" status --porcelain=v2 --untracked-files=all)"
after_diff="$(git -C "$SOURCE_ROOT" diff --name-status HEAD)"

final_status=$baseline_status
integrity_clean=yes
if [[ -n "$after_status" || -n "$after_diff" ]]; then
    integrity_clean=no
    if [[ $final_status -eq 0 ]]; then final_status=3; fi
fi

total_mae=""
total_rmse=""
metric_match=no
if [[ -f "$SUMMARY_PATH" ]]; then
    total_row="$(tail -n 1 "$SUMMARY_PATH" | tr -d '\r')"
    IFS=',' read -r _ _ total_mae total_rmse <<< "$total_row"
    if "$PYTHON_BIN" -B - "$total_mae" "$total_rmse" "$EXPECTED_MAE" "$EXPECTED_RMSE" <<'PY'
import math
import sys

actual_mae, actual_rmse, expected_mae, expected_rmse = map(float, sys.argv[1:])
assert math.isclose(actual_mae, expected_mae, rel_tol=0.0, abs_tol=1e-12)
assert math.isclose(actual_rmse, expected_rmse, rel_tol=0.0, abs_tol=1e-12)
PY
    then
        metric_match=yes
    elif [[ $final_status -eq 0 ]]; then
        final_status=4
    fi
elif [[ $final_status -eq 0 ]]; then
    final_status=5
fi

{
    echo "baseline_exit_code=$baseline_status"
    echo "final_exit_code=$final_status"
    echo "elapsed_seconds=$elapsed_seconds"
    echo "upstream_clean_before=yes"
    echo "upstream_clean_after=$integrity_clean"
    echo "summary_path=$SUMMARY_PATH"
    echo "detail_path=$DETAIL_PATH"
    echo "total_mae=$total_mae"
    echo "total_rmse=$total_rmse"
    echo "metric_match_phase1=$metric_match"
    echo "log_path=$LOG_PATH"
    echo "before_integrity_path=$BEFORE_PATH"
    echo "after_integrity_path=$AFTER_PATH"
} > "$STATUS_PATH"

echo "PHASE1_5B_BASELINE_EXIT_CODE=$baseline_status"
echo "PHASE1_5B_FINAL_EXIT_CODE=$final_status"
echo "PHASE1_5B_ELAPSED_SECONDS=$elapsed_seconds"
echo "PHASE1_5B_TOTAL_MAE=$total_mae"
echo "PHASE1_5B_TOTAL_RMSE=$total_rmse"
echo "PHASE1_5B_METRIC_MATCH_PHASE1=$metric_match"
echo "PHASE1_5B_UPSTREAM_CLEAN_AFTER=$integrity_clean"
echo "PHASE1_5B_STATUS_PATH=$STATUS_PATH"

if [[ "$integrity_clean" != "yes" ]]; then
    echo "Phase 1.5B integrity violation detected. No cleanup or reset was attempted."
fi

exit "$final_status"
