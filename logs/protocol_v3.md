# experimental protocol v3

## status

this version is the **final frozen intrinsic experimental protocol** for the dissertation run campaign.

it is justified by:

- completed corpus construction and split finalisation,
- successful l40s smoke validation for both `LoRA` and `QLoRA`,
- the first matched 8b phase-1 `LoRA` vs `QLoRA` runs on target hardware,
- the observed validation-curve behaviour in those matched runs.

after this version is frozen:

- no hyperparameter tuning is permitted,
- no batching / memory-policy changes are permitted,
- no dataset / split / sequence-length changes are permitted,
- no run-matrix changes are permitted.

the only allowed reruns are **identical reruns** of the same config when a job fails because of infrastructure or launch-path issues rather than the experimental recipe itself.

---

## 1. study framing

### title
**quantisation effects in parameter-efficient domain adaptation under fixed resource budgets:  
a comparative study of lora and qlora on uk employment-law text**

### primary aim
to quantify how 4-bit quantisation changes adaptation quality, memory use, runtime efficiency, and training stability in parameter-efficient domain-adaptive continued pretraining, by comparing `LoRA` and `QLoRA` under fixed budgets on UK employment-law text.

### role of this protocol version
this protocol defines the **final intrinsic dissertation runs**.

it exists to:

1. lock the final matched-budget intrinsic comparison,
2. provide seed-replicated results suitable for dissertation reporting,
3. prevent further recipe drift after target-hardware validation has already succeeded.

---

## 2. what the march 18 l40s runs justify

the successful l40s smoke and phase-1 runs justify the following claims:

- `LoRA` 8b and `QLoRA` 8b both complete successfully at:
  - `max_seq_length=1024`
  - `per_device_train_batch_size=2`
  - `gradient_accumulation_steps=4`
  - `gradient_checkpointing=false`
- both methods were stable under:
  - `AdamW`
  - `lr=2e-4`
  - cosine schedule
  - warmup `0.05`
- both matched phase-1 runs reached their **best overall validation perplexity at step 350**, then degraded afterwards.

### concrete matched-run evidence

- `LoRA` phase-1:
  - baseline ppl `7.96124154702041`
  - best ppl `7.273406827525719` at `checkpoint-350`
  - final ppl at step `750`: `7.556342419382136`
- `QLoRA` phase-1:
  - baseline ppl `8.316989216966332`
  - best ppl `7.407259106615078` at `checkpoint-350`
  - final ppl at step `750`: `7.6764739765545285`

### interpretation

the current recipe does **not** show a primary instability problem.

instead, both methods show the same pattern:

- strong early improvement,
- minimum overall validation perplexity around `step=350`,
- later degradation under continued training.

this is strong evidence that the previous `750`-step budget was longer than necessary for the fixed train split and that the final protocol should reduce `max_steps` rather than begin a new learning-rate search.

---

## 3. model

- **base model:** `meta-llama/Meta-Llama-3-8B`
- **tokenizer:** LLaMA-3 tokenizer (`vocab_size=128256`)
- **context length / max_seq_length:** `1024`
- **dtype:** `bfloat16`

### status

all model settings above are **frozen**.

### justification

the model family, tokenizer, and sequence length are part of the dissertation design rather than exploratory tuning variables. they were already validated successfully on target hardware in the smoke and phase-1 runs.

---

## 4. data

- **data type:** domain-adaptive language modelling on curated UK employment-law text
- **corpus manifest:** `data/domain_corpus/corpus_manifest.json`
- **corpus directory:** `data/domain_corpus/corpus_final`
- **split file:** `data/splits/intrinsic_splits.json`

### corpus totals

- **total tokens:** `1,802,427`
- **layer A:** `1,602,190` tokens, `170` files
- **layer B:** `200,237` tokens, `60` files

### split policy

- **train/val split:** `85/15`, document-level
- **stratification:** by token-length quartile within layer
- **split seed:** `42`
- **document cap:** `20,000` tokens max per document, paragraph-boundary aware

### derived train-set quantities

- **train tokens:** `1,540,529`
- **train chunks at seq_len 1024:** `1504`
- **validation chunks:** `255`

### status

all data settings above are **frozen**.

### justification

the dissertation is a controlled comparative study under fixed data conditions. changing corpus membership, split policy, or sequence length after the target-hardware runs would weaken comparability more than it would strengthen optimisation.

---

## 5. optimisation settings

### frozen across all final runs

- **optimiser:** `AdamW`
- **betas:** `(0.9, 0.999)`
- **eps:** `1e-8`
- **weight_decay:** `0.01`
- **scheduler:** cosine
- **warmup_ratio:** `0.05`
- **gradient clip:** `1.0`

### learning rate

- **learning rate:** `2e-4`

### status

all optimisation settings above are **frozen**.

### justification

the march 18 l40s runs showed:

- smooth early validation improvement for both methods,
- no `nan`, `inf`, `nonfinite_grad_norm`, or `oom` events,
- the same best-checkpoint location for both methods.

this is evidence that `2e-4` is already a workable matched setting on target hardware. the main problem was overlong training, not an obviously unsafe or ineffective learning rate.

---

## 6. runtime and memory policy

### frozen policy

- **per_device_train_batch_size:** `2`
- **per_device_eval_batch_size:** `1`
- **gradient_accumulation_steps:** `4`
- **effective batch size:** `8` sequences / optimiser step
- **gradient checkpointing:** `false`
- **attention implementation:** `sdpa`
- **flash_sdp:** `true`
- **mem_efficient_sdp:** `false`
- **math_sdp:** `false`
- **mixed precision / compute dtype:** `bf16`

### status

all runtime and memory-policy settings above are **frozen**.

### justification

these exact settings were proven feasible on the l40s cluster for both `LoRA` and `QLoRA`.

keeping `gradient_checkpointing=false` is important because:

- it is no longer required for feasibility on target hardware,
- it avoids adding a runtime/memory confound after the comparison path is already working,
- it keeps the final runs aligned with the validated march 18 recipe.

---

## 7. step budget and cadence

### final budget

- **max_steps:** `350`
- **eval_steps:** `50`
- **save_steps:** `50`
- **logging_steps:** `10`

### derived budget quantities

- **tokens per optimiser step:** `8192`
- **tokens processed in 350 steps:** `2,867,200`
- **steps per epoch-equivalent:** `188`
- **epoch-equivalents at 350 steps:** `~1.86`

### status

all budget and cadence settings above are **frozen**.

### justification

`350` steps is chosen because both matched phase-1 runs reached their minimum overall validation perplexity at exactly `checkpoint-350` and then degraded substantially by later checkpoints.

keeping `eval_steps=50` and `save_steps=50` is justified because:

- that cadence already captured the turning point cleanly,
- it provides checkpoint-selection granularity without excessive overhead,
- it has already been validated operationally and analytically.

this change preserves the dissertation’s fixed-budget design while removing clearly unhelpful extra training.

---

## 8. adapter configurations

### 8.1 LoRA configuration

- **target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **rank:** `16`
- **alpha:** `32`
- **dropout:** `0.05`

### 8.2 QLoRA configuration

- **quantisation:** 4-bit `NF4`
- **double quantisation:** `true`
- **compute dtype:** `bf16`
- **adapter target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **dropout:** `0.05`

### 8.3 QLoRA rank sweep

| condition | rank | alpha |
|---|---:|---:|
| qlora_r16 | 16 | 32 |
| qlora_r32 | 32 | 64 |
| qlora_r64 | 64 | 128 |

### status

all adapter settings above are **frozen**.

### justification

the dissertation questions are specifically about:

- `LoRA` vs `QLoRA` under matched budgets,
- adapter capacity within quantisation.

changing target modules or dropout now would broaden the study unnecessarily and blur interpretation.

---

## 9. final intrinsic run matrix

all runs share identical:

- dataset and split,
- sequence length,
- optimiser and scheduler,
- runtime policy,
- step budget,
- evaluation cadence,
- checkpoint rule.

### seed policy

- **training seeds:** `42`, `43`, `44`
- **split seed remains fixed at:** `42`

### phase 1 — core comparison

| run group | method | rank | alpha | seeds |
|---|---|---:|---:|---|
| p1_lora_r16 | LoRA | 16 | 32 | 42, 43, 44 |
| p1_qlora_r16 | QLoRA | 16 | 32 | 42, 43, 44 |

### phase 2 — quantisation capacity comparison

| run group | method | rank | alpha | seeds |
|---|---|---:|---:|---|
| p2_qlora_r32 | QLoRA | 32 | 64 | 42, 43, 44 |
| p2_qlora_r64 | QLoRA | 64 | 128 | 42, 43, 44 |

### total final intrinsic runs

- **conditions:** `4`
- **seeds per condition:** `3`
- **total intrinsic training runs:** `12`

### justification

three seeds are a good compromise between:

- replicability,
- reporting of optimisation variance,
- the practical compute budget for the dissertation.

the split remains fixed so that cross-seed variation reflects **training variance**, not data-resampling variance.

---

## 10. baseline evaluation policy

every run must begin with a **baseline evaluation before any training**.

### required baselines

| run type | baseline model |
|---|---|
| LoRA | bf16 base model |
| QLoRA | 4-bit NF4 quantised base model |

### metrics recorded

- validation loss (overall)
- validation perplexity (overall)
- validation perplexity (layer A)
- validation perplexity (layer B)
- baseline eval peak VRAM allocated
- baseline eval peak VRAM reserved

### justification

the baseline is part of the scientific comparison because it captures the pre-adaptation cost of quantisation and allows reporting of both:

- absolute best/final perplexity,
- improvement from each method’s own starting point.

---

## 11. evaluation and checkpoint policy

### metrics retained per evaluation

- validation loss (overall)
- validation perplexity (overall)
- validation perplexity (layer A)
- validation perplexity (layer B)
- train step-time statistics over the preceding interval
- eval peak GPU memory

### checkpoint selection rule

best checkpoint = **minimum overall validation perplexity**

layer-stratified perplexities are retained for analysis but do not determine checkpoint selection.

### retained checkpoints

- best checkpoint by overall validation perplexity
- final checkpoint

### justification

the march 18 runs show that the best checkpoint and final checkpoint can differ materially. retaining both is therefore scientifically necessary rather than optional.

---

## 12. reporting requirements

every run must emit structured artefacts sufficient for dissertation reporting.

### required per-run reports

- `run_manifest.json`
- `resolved_config.yaml`
- `environment.json`
- `dataset_summary.json`
- `budget_summary.json`
- `metrics_history.jsonl`
- `eval_summary.csv`
- `final_metrics.json`
- `memory_summary.json`
- `timing_summary.json`
- `stability_summary.json`
- `stability_events.jsonl`
- `comparison_row.json`
- `run.log`

### cross-run reporting requirement

the final run campaign must also produce:

- raw comparison rows aggregated across all runs,
- condition-level summaries across seeds,
- plot-ready merged evaluation curves across runs.

### justification

the dissertation results chapter requires:

- run-level evidence,
- seed-level aggregation,
- visual analysis of learning curves.

the logging surface must therefore remain fixed across all v3 runs.

---

## 13. aggregation and reporting policy

### primary reported intrinsic metrics

- baseline, best, and final perplexity:
  - overall
  - layer A
  - layer B
- memory:
  - `train_peak_vram_*`
  - `max_eval_peak_vram_*`
  - `run_peak_vram_*`
- runtime:
  - `train_runtime_s`
  - `optimizer_step_time_mean_s`
  - `optimizer_step_time_p50_s`
  - `optimizer_step_time_p95_s`
  - `optimizer_step_time_std_s`
  - `num_step_time_samples`
- stability:
  - `num_nan_loss_events`
  - `num_inf_loss_events`
  - `num_nonfinite_grad_norm_events`
  - `num_oom_events`
  - `terminated_early`
  - `termination_reason`

### cross-seed summaries

for each condition, report at minimum:

- `mean`
- `std`
- `min`
- `max`
- `n`

### note on intrinsic evaluation semantics

the intrinsic study remains **train/validation only**.

results must therefore be reported as **held-out validation perplexity** rather than unbiased test performance.

### justification

multiple seeds improve reliability against optimisation noise. they do not convert the intrinsic study into a train/validation/test design. this limitation remains explicit in the final report.

---

## 14. downstream evaluation scope

downstream transfer evaluation is **not part of the frozen intrinsic run matrix** in this protocol.

if implemented later, it should:

- use checkpoints selected from the intrinsic study,
- use its own train/validation/test split,
- be documented as a secondary evaluation stage rather than a reason to alter any intrinsic v3 settings.

### justification

keeping downstream evaluation outside the intrinsic freeze prevents last-minute changes to the main comparative experiment.

---

## 15. freeze rules after v3

after the first `protocol_v3` run starts, the following are not allowed:

- changing `max_steps`
- changing batch size or accumulation
- changing `learning_rate`
- changing `warmup_ratio`
- changing `weight_decay`
- changing `gradient_checkpointing`
- changing attention backend settings
- changing adapter target modules
- changing split policy or corpus contents
- adding or removing conditions from the run matrix
- changing the training seed set

the following are allowed:

1. rerun the exact same config after a launcher / cluster / filesystem / authentication failure,
2. rerun the exact same config if a run is corrupted or incomplete and did not produce valid reports,
3. add downstream evaluation later without altering intrinsic configs.

### final protocol summary

### frozen scientific constants

- base model: `meta-llama/Meta-Llama-3-8B`
- seq_len: `1024`
- train/val split: fixed `85/15`, split seed `42`
- optimiser: `AdamW`
- lr: `2e-4`
- warmup: `0.05`
- weight decay: `0.01`
- per-device train batch: `2`
- grad accumulation: `4`
- effective batch: `8`
- max steps: `350`
- eval/save cadence: `50`
- logging cadence: `10`
- gradient checkpointing: `false`
- LoRA base comparison: `r=16`, `alpha=32`
- QLoRA rank sweep: `r=16/32/64`, `alpha=32/64/128`
- training seeds: `42`, `43`, `44`

### final rationale in one sentence

`protocol_v3` freezes the march 18 target-hardware recipe, shortens the step budget to the empirically justified validation optimum, and reallocates the saved compute into multi-seed replication rather than further single-run tuning.
