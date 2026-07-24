# ACL baseline evaluation contract

This document is the source of truth for the external-baseline comparison in
the ACL paper. The historical values in `VDT_report.tex` are retained as
context only until this contract has been executed for every baseline.
The machine-readable registry is `configs/acl_external_baselines.json`.

## Required baseline suite

The suite is pinned by model identifier and revision before generation:

- VietAI;
- GPT-4o;
- GPT-3.5-Turbo (the historical report does not contain a GPT-3.3 model);
- Qwen3-14B;
- Llama3.3-70B-Instruct;
- Phi-4-14B;
- Sailor-20B-chat; and
- VinBigdata-7B.

The project controls are Qwen3-4B-Base/Instruct pretrained, SFT-only,
fresh-GRPO v5, and SFT-initialized-GRPO v5.

## Shared protocol

Every model receives the same source-disjoint manifest, semantic instruction,
reference-derived length/sentence requirements, source truncation policy, and
output cap of 256 new tokens. Each backend may use its native chat wrapper,
but the rendered semantic prompt and all decoding parameters must be recorded.
The run manifest must include:

- model ID, revision, license and parameter-count metadata;
- prompt/template hash and manifest hash;
- deterministic decoding settings (`do_sample=False` where supported);
- generation files and failed/empty-output counts; and
- the scorer package/version and whitespace/sentence-count definitions.

ROUGE-2, absolute length distance and relative length error are computed by
the same post-processing code for every model. Sentence exact/tolerant hit and
MAE are supporting metrics. BARTScore is not used because it is unavailable in
the offline project environment. The saved-generation audit additionally
records the manifest SHA-256, per-model counts, and paired bootstrap metadata.

## Validity gate

An external row is allowed into the direct-comparison table only when it has a
complete generation file, the same manifest hash, and a verified protocol
record. Rows copied from the historical VDT benchmark remain in a separate
contextual table and must never be bolded or used for a direct superiority
claim. API credentials, model access, and any compute/cost approval are
required before running the suite; no credentials are stored in this repo.

## Multi-document extension

The current ViMs/VLSP test records overlap project training or validation. A
generalization claim therefore requires a new source-disjoint cluster split
and project retraining after the split is frozen, followed by the same baseline
protocol. Existing ViMs/VLSP numbers are diagnostic historical evidence only.
