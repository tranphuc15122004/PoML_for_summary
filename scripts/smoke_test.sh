#!/bin/bash
# =============================================================================
# Smoke Test - Kiểm tra pipeline trước khi submit batch job
# Chạy trên interactive GPU node
# NOTE: legacy smoke test for the pre-Qwen3 data/model schema; not the canonical evaluation protocol.
# Usage: bash scripts/smoke_test.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/scratch/jp09/dd9648/PoML_for_summary"
VENV_PATH="${PROJECT_ROOT}/.venv"
SRC_PATH="${PROJECT_ROOT}/src"
DATA_DIR="${PROJECT_ROOT}/data/smoke_test"
MODELS_DIR="${PROJECT_ROOT}/models/smoke_test"

mkdir -p "${DATA_DIR}" "${MODELS_DIR}" "${PROJECT_ROOT}/logs"
LOG_FILE="${PROJECT_ROOT}/logs/smoke_test_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo "SMOKE TEST - PoML for Summary"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "================================================================"

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${SRC_PATH}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOCAL_MODEL="/g/data/hn98/dd9648/models/Qwen2.5-3B-Instruct"
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="/scratch/jp09/dd9648/.cache/huggingface/datasets"
mkdir -p "${HF_DATASETS_CACHE}"
export TRANSFORMERS_OFFLINE=1

PASS=0
FAIL=0

check() {
    local name="$1"
    shift
    echo ""
    echo "--- [TEST] ${name} ---"
    if "$@"; then
        echo "  PASS: ${name}"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: ${name}"
        FAIL=$((FAIL + 1))
    fi
}

# ============================================================
# 1. ENVIRONMENT
# ============================================================
echo ""
echo "=== 1. ENVIRONMENT ==="

check "Python version" python -c "
import sys
assert sys.version_info >= (3, 10)
print(f'Python {sys.version}')
"

check "PyTorch + CUDA" python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

check "Key packages" python -c "
import transformers, peft, trl, datasets, accelerate
print(f'transformers={transformers.__version__}, peft={peft.__version__}')
print(f'trl={trl.__version__}, datasets={datasets.__version__}')
"

check "Project imports" python -c "
import sys; sys.path.insert(0, '${SRC_PATH}')
from dataset.dataset import VietNewsDataset, WikiLinguaDataset, DatasetConfig
from dataset.augmenter import PromptAugmenter, build_all_splits
from SFT_GRPO.config import SFTConfig, GRPOConfig, ModelConfig, EvalConfig
from SFT_GRPO.rewards import compute_all_rewards
from SFT_GRPO.metrics_logger import MetricsTracker
print('All project imports OK')
"

# ============================================================
# 2. MODEL ACCESS
# ============================================================
echo ""
echo "=== 2. MODEL ACCESS ==="

check "Tokenizer (local)" python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${LOCAL_MODEL}', trust_remote_code=True, local_files_only=True)
print(f'Tokenizer: {tok.__class__.__name__}, vocab={len(tok)}')
enc = tok('Xin chào thế giới', return_tensors='pt')
print(f'Encode OK, shape: {enc.input_ids.shape}')
"

# ============================================================
# 3. DATA PIPELINE
# ============================================================
echo ""
echo "=== 3. DATA PIPELINE ==="

check "Dataset classes importable" python -c "
import sys; sys.path.insert(0, '${SRC_PATH}')
from dataset.dataset import WikiLinguaDataset, VietNewsDataset, VLSPDataset, DatasetConfig
# Verify WikiLingua can iterate if expected path exists
import os
wl_path = '${PROJECT_ROOT}/VDT_Textsum/wikilingua/wikilingua/train.json'
wl_flat = '${PROJECT_ROOT}/VDT_Textsum/wikilingua/train.json'
if os.path.exists(wl_path):
    cfg = DatasetConfig(data_root='${PROJECT_ROOT}/VDT_Textsum', mode='raw')
    ds = WikiLinguaDataset(cfg)
    samples = [s for _, s in zip(range(3), ds)]
    print(f'WikiLingua (nested) OK: {len(samples)} samples')
elif os.path.exists(wl_flat):
    print(f'NOTE: WikiLingua data at flat path {wl_flat}')
    print(f'      dataset.py expects wikilingua/wikilingua/ subdirectory')
    print(f'      Raw data loader will fail - but JSONL pipeline is unaffected')
else:
    print('WikiLingua data not found - skipping')
print('Dataset classes importable OK')
"

check "Synthetic data creation" python -c "
import json, os, random
DATA_DIR = '${DATA_DIR}'
os.makedirs(DATA_DIR, exist_ok=True)

src = 'Hội đồng Nhân dân thành phố Hà Nội vừa thông qua nghị quyết về phát triển kinh tế số giai đoạn 2024-2030 với tổng vốn đầu tư dự kiến lên đến 50.000 tỷ đồng. Theo đó, thành phố sẽ ưu tiên đầu tư vào hạ tầng số, phát triển nguồn nhân lực và chuyển đổi số cho doanh nghiệp vừa và nhỏ.'
ref = 'Hà Nội thông qua nghị quyết phát triển kinh tế số 2024-2030 với 50.000 tỷ đồng đầu tư vào hạ tầng số.'

sft = [{'messages': [
    {'role': 'system', 'content': 'Bạn là chuyên gia tóm tắt văn bản tiếng Việt.'},
    {'role': 'user', 'content': f'Tóm tắt (khoảng 50 từ, phong cách báo chí):\n\n{src}'},
    {'role': 'assistant', 'content': ref}
], 'meta': {'length_requirement': 'khoảng 50 từ', 'style': 'báo chí'}} for _ in range(20)]

grpo = [{'prompt': [
    {'role': 'system', 'content': 'Bạn là chuyên gia tóm tắt văn bản tiếng Việt.'},
    {'role': 'user', 'content': f'Tóm tắt (khoảng 50 từ, phong cách báo chí):\n\n{src}'}
], 'reference': ref, 'meta': {'length_requirement': 'khoảng 50 từ', 'style': 'báo chí'}} for _ in range(10)]

test_data = [{'source': src, 'reference': ref, 'meta': {'length_requirement': 'khoảng 50 từ', 'style': 'báo chí'}} for _ in range(5)]

for fname, data in [('sft_train.jsonl', sft[:16]), ('sft_val.jsonl', sft[16:]),
                    ('grpo_train.jsonl', grpo[:8]), ('grpo_val.jsonl', grpo[8:]),
                    ('test.jsonl', test_data)]:
    with open(os.path.join(DATA_DIR, fname), 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'  {fname}: {len(data)} lines')
print('Data OK')
"

# ============================================================
# 4. REWARDS
# ============================================================
echo ""
echo "=== 4. REWARDS ==="

check "Reward functions" python -c "
import sys; sys.path.insert(0, '${SRC_PATH}')
from SFT_GRPO.rewards import compute_all_rewards
gen = 'Hà Nội thông qua nghị quyết phát triển kinh tế số với 50.000 tỷ đồng.'
ref = 'Hà Nội thông qua nghị quyết kinh tế số 2024-2030.'
rewards = compute_all_rewards(gen, ref, length_requirement='khoảng 10 từ', style='báo chí')
print(f'Rewards: {rewards}')
assert isinstance(rewards, dict) and len(rewards) >= 1
print('Rewards OK')
"

# ============================================================
# 5. SFT TRAINING (~8 steps)
# ============================================================
echo ""
echo "=== 5. SFT TRAINING ==="

GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
if echo "${GPU_NAME}" | grep -qiE "H100|H200|Hopper|A100"; then
    BF16="true"; FP16="false"
else
    BF16="false"; FP16="true"
fi
echo "GPU: ${GPU_NAME} -> bf16=${BF16}"

cat > /tmp/smoke_sft.json << EOF
{
    "train_data_path": "${DATA_DIR}/sft_train.jsonl",
    "val_data_path":   "${DATA_DIR}/sft_val.jsonl",
    "num_train_epochs": 0.5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_seq_length": 512,
    "logging_steps": 1,
    "save_steps": 9999,
    "eval_strategy": "no",
    "report_to": "none",
    "bf16": ${BF16},
    "fp16": ${FP16},
    "gradient_checkpointing": false
}
EOF

check "SFT ~8 steps" python "${PROJECT_ROOT}/src/SFT_GRPO/train_sft.py" \
    --model_name "${LOCAL_MODEL}" \
    --output_dir "${MODELS_DIR}/sft" \
    --bf16 "${BF16}" \
    --fp16 "${FP16}" \
    --config /tmp/smoke_sft.json

# ============================================================
# 6. GRPO TRAINING (5 steps)
# ============================================================
echo ""
echo "=== 6. GRPO TRAINING ==="

# Use greedy decoding (do_sample=false) to avoid NaN from sampling with synthetic data
cat > /tmp/smoke_grpo.json << EOF
{
    "do_sample": false,
    "max_new_tokens": 64,
    "max_seq_length": 512,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "logging_steps": 1,
    "save_steps": 9999
}
EOF

check "GRPO 5 steps" python "${PROJECT_ROOT}/src/SFT_GRPO/train_grpo.py" \
    --model_name "${LOCAL_MODEL}" \
    --output_dir "${MODELS_DIR}/grpo" \
    --train_data "${DATA_DIR}/grpo_train.jsonl" \
    --val_data "${DATA_DIR}/grpo_val.jsonl" \
    --total_steps 5 \
    --num_generations 2 \
    --config /tmp/smoke_grpo.json

# ============================================================
# 7. EVALUATION (config + module check)
# ============================================================
echo ""
echo "=== 7. EVALUATION ==="

check "Evaluate config + module" python -c "
import sys; sys.path.insert(0, '${SRC_PATH}')
from SFT_GRPO.evaluate import run_evaluation
from SFT_GRPO.config import EvalConfig
cfg = EvalConfig(
    test_data_path='${DATA_DIR}/test.jsonl',
    output_dir='${MODELS_DIR}/eval',
    batch_size=2,
    model_paths={'base': '${LOCAL_MODEL}'}
)
print(f'EvalConfig OK: {cfg.test_data_path}')
print(f'Model: {cfg.model_paths}')
"

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "================================================================"
echo "SMOKE TEST RESULTS"
echo "================================================================"
echo "  PASSED: ${PASS}"
echo "  FAILED: ${FAIL}"
echo "  Log: ${LOG_FILE}"
echo "================================================================"

if [ "${FAIL}" -gt 0 ]; then
    echo "RESULT: FAIL - ${FAIL} test(s) failed."
    exit 1
else
    echo "RESULT: PASS - Pipeline ready for batch job submission."
    echo ""
    echo "Model path for batch jobs: ${LOCAL_MODEL}"
    echo "Update PBS scripts or pass: --model_name ${LOCAL_MODEL}"
fi
