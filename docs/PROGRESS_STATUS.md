# Progress snapshot: NCI clean evaluation

This snapshot records the state at the user-requested pause. No PBS job was
cancelled and no working-tree changes were reverted.

## PBS job

- Job: `174156827.gadi-pbs`
- Queue/resource: `gpuhopper`, one NVIDIA H200, 12 CPUs, 120 GB, 12-hour limit
- Last observed state: `R` (running); do not cancel it
- Test manifest: `data/test_clean_single_document.jsonl`
- Samples: 2,490 (1,990 VietNews + 500 WikiLingua)
- Models discovered: 16 Qwen3 Base/Instruct checkpoints
- Decoding: greedy, max 256 new tokens
- Output root: `models/eval_results/clean_single_greedy`
- PBS logs are configured as `logs/eval_nci.out` and `logs/eval_nci.err`; live
  output is staged by PBS when the job exits.

The last live log snapshot showed Base pretrained and Base SFT completed. The
Base fresh-v3 model had started generation. The observed interim metrics were
not used to edit the paper.

## Files changed for this work

- `scripts/pbs/eval.pbs`: NCI PBS evaluation job with clean-test, decoding,
  seed, top-p, output-directory, and optional BARTScore controls.
- `scripts/launch/eval.py`: Qwen3 family discovery plus deterministic/seeded
  evaluation arguments.
- `src/evaluation/evaluate.py`: greedy-vs-sampling generation, seed handling,
  canonical source hashes, and manifest/decoding provenance in `summary.json`.
- `scripts/tools/audit_data_integrity.py`: Unicode-normalized source-hash audit
  and clean single-document manifest generation.
- `docs/PROJECT_AUDIT.md`: clean-manifest counts and provenance boundary.
- `FormalReport_VDT/latex/acl_latex.tex`: single-document scope, contribution
  framing, bounded claims, and hidden multi-document/external-baseline tables.

Generated or ignored artefacts:

- `data/test_clean_single_document.jsonl`
- `models/data_audit/integrity_report.json`
- `FormalReport_VDT/latex/acl_latex.pdf` (last successful build: 15 pages)

## Validation already completed

- Python compilation for evaluator, launcher, and audit script: passed.
- PBS shell syntax check: passed.
- CPU-only `save_results` provenance smoke test: passed.
- Two-pass XeLaTeX build: passed with no undefined references; only layout
  warnings remain.

## Continue after PBS completion

1. Read the staged PBS log and locate the timestamped run directory below
   `models/eval_results/clean_single_greedy/`.
