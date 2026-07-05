# Related Work Research Memo

This note records the source groups and claim boundaries used to rewrite `FormalReport_VDT/latex/acl_latex.tex` without drifting beyond the audit limits in `docs/PROJECT_AUDIT.md`.

## Source groups

### Vietnamese summarization

- `nguyen2019vnds` for the original Vietnamese news benchmark.
- `tran2020vims` for the multi-document Vietnamese abstractive benchmark.
- `ladhak2020wikilingua` for the held-out non-news single-document domain.
- `tran2023abmusu` for the VLSP multi-document shared-task framing.
- `phan2022vit5` and `lam2023viesum` for Vietnamese modeling gains beyond data construction.

### Controllable and length-aware summarization

- `kikuchi2016controlling`, `fan2018controllable`, and `takase2019positional` for early length control.
- `he2020ctrlsum` and `chan2021cmdp` for prompt-based and constrained-RL control.
- `miculicich2023precise`, `liu2023instrusum`, `jie2024lengthcontrol`, and `instructcmp2024` for precise or instruction-based length control.

### Post-training

- `hu2021lora` for parameter-efficient adaptation.
- `stiennon2020summarize` and `bohm2019betterrewards` for reward-based summarization training.
- `shao2024deepseekmath` for the GRPO reference point.

### Reward design and hacking

- `amodei2016concrete` and `skalse2022rewardhacking` for the safety framing.
- `gao2022scaling` for proxy reward overoptimization.

## Claim mapping

- Vietnamese summarization still has a smaller modeling literature than English.
- Most prior controllable-summarization work focuses on one control attribute or on English benchmarks.
- Parameter-efficient post-training is the right abstraction for the project's LoRA-SFT plus GRPO recipe.
- The v3-to-v5 changes are a mitigation package, not a single-component causal ablation.
