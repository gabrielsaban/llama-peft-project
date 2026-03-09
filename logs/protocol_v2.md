# experimental protocol v2

## status

this version freezes the parameters that can now be justified from:

- corpus preparation and split finalisation
- successful 3B local plumbing/calibration runs on the 4070
- the intended dissertation design constraints

this is **not yet the final dissertation protocol**. it is the **early L40 cluster protocol** used to:

1. validate 8B feasibility on target hardware,
2. freeze remaining resource-budget constants,
3. begin the main LoRA vs QLoRA comparison under matched settings.

---

## 1. study framing

### title
**quantisation effects in parameter-efficient domain adaptation under fixed resource budgets:  
a comparative study of lora and qlora on uk employment-law text**

### primary aim
to evaluate how 4-bit quantisation alters memory efficiency, optimisation behaviour, and adaptation quality in parameter-efficient domain-adaptive continued pretraining, by comparing LoRA and QLoRA under fixed budgets on UK employment-law text.

### role of this protocol version
this protocol defines the **initial 8B L40 run conditions**.  
it freezes all settings that should remain constant across methods unless 8B feasibility forces a documented change.

---

## 2. model

- **base model:** `meta-llama/Llama-3-8B`
- **calibration model:** `meta-llama/Llama-3.2-3B`
  - used only for local plumbing / feasibility / instrumentation validation
  - not part of the formal dissertation comparison
- **tokenizer:** LLaMA-3 tokenizer (vocab 128,256)
- **context length / max_seq_length:** **1024**
  - **frozen**
  - justification:
    - central to the intended study design
    - 3B local pilots successfully validated seq_len=1024 end-to-end
    - study focuses on optimisation / adaptation under fixed budgets, not long-context scaling

---

## 3. data

- **data type:** domain-adaptive language modelling on curated UK employment-law corpus
- **corpus manifest:** `data/domain_corpus/corpus_manifest.json`
- **corpus directory:** `data/domain_corpus/corpus_final`
- **split file:** `data/splits/intrinsic_splits.json`

### corpus totals
- **total tokens:** `1,802,427`
- **layer A (tribunal decisions):** `1,602,190` tokens, `170` files, `88.9%`
- **layer B (guidance/doctrine):** `200,237` tokens, `60` files, `11.1%`

### split policy
- **train/val split:** `85/15`, document-level
- **stratification:** by post-cap document token length quartile within layer
- **split seed:** `42`
- **document cap:** `20,000` tokens max per document, paragraph-boundary aware

### status
all data settings above are **frozen** for early L40 runs.

### justification
these values derive from completed corpus construction and split generation, not hardware calibration, so they should not move between conditions.

---

## 4. optimisation settings

## 4.1 frozen across early L40 runs

- **optimiser:** AdamW
- **betas:** `(0.9, 0.999)`
- **eps:** `1e-8`
- **weight_decay:** `0.01`
- **scheduler:** cosine
- **warmup_ratio:** `0.05`
- **gradient_clip:** `1.0`
- **training seed:** `42`

### justification
these are standard, stable settings and should remain identical across LoRA and QLoRA so that observed differences are attributable to quantisation rather than recipe drift.

---

## 4.2 learning rate policy

- **initial learning rate for early 8B L40 runs:** `2e-4`
- **fallback learning rate if early instability is observed:** `1e-4`

### status
- `2e-4` is **tentatively frozen as the initial 8B starting value**
- final dissertation LR is only frozen after the first stable 8B L40 sanity run

### justification
the 3B local pilots completed successfully at `2e-4` for both LoRA and QLoRA without divergence, so `2e-4` is a justified starting point. however, the 3B pilots do **not** fully determine the optimal or safest 8B LR, so 8B confirmation is still required.

---

## 4.3 resource-budget values still to be frozen on L40

the following are **not yet final** and must be confirmed by the first 8B cluster feasibility runs:

- **per_device_train_batch_size**
- **gradient_accumulation_steps**
- **effective_batch_size**
- **max_steps**
- **equivalent_epochs**
- **eval_steps / save_steps**

### why these remain provisional
these values depend directly on actual 8B memory fit and throughput on the L40 48GB hardware. the 3B local pilots provide useful directional information but are not sufficient to lock these constants.

---

## 5. early L40 starting values

these values should be used for the **first 8B feasibility runs** unless immediate hardware evidence forces revision.

### proposed starting configuration
- **per_device_train_batch_size:** `2`
- **per_device_eval_batch_size:** `1`
- **gradient_accumulation_steps:** `4`
- **effective_batch_size:** `8`
- **max_steps:** `750`
- **eval_steps:** `50`
- **save_steps:** `50`

### justification

#### batch / accumulation
`per_device_train_batch_size=2` and `grad_accum=4` is a conservative but sensible first 8B bf16 LoRA starting point on a 48GB L40 at seq_len=1024, while still maintaining a non-trivial effective batch.

#### max_steps
with roughly `1.53M` train tokens after the 85/15 split, and `8192` nominal tokens per optimiser step at effective batch 8 and seq_len 1024:

- steps per epoch-equivalent ≈ `187`
- `750` steps ≈ `4.0` epoch-equivalents

this sits well inside the originally intended ~3–5 epoch-equivalent range.

#### eval / save schedule
`eval_steps=50` yields ~15 evaluation points over 750 steps, which is enough to analyse learning curves and select the best checkpoint without excessive eval overhead.

---

## 6. memory controls

### frozen default policy
- **gradient checkpointing:** **off**
- **attention implementation:** `sdpa`
- **flash_sdp:** on where available
- **mem_efficient_sdp:** off unless explicitly required and kept consistent across conditions
- **math_sdp:** off unless explicitly required and kept consistent across conditions
- **mixed precision / compute dtype:** `bfloat16`

### feasibility fallback rule
if 8B bf16 LoRA at the target seq_len and initial batch settings does not fit on the L40:

1. reduce **microbatch size first**
2. preserve matching optimisation settings across conditions where possible
3. enable **gradient checkpointing for all compared conditions only if required for feasibility**

### justification
the local 3B LoRA run required checkpointing on weaker hardware, but the L40 should be substantially more capable. keeping checkpointing off initially gives a cleaner comparison of VRAM and runtime. it should only be introduced if feasibility requires it.

---

## 7. LoRA configuration

the following are **frozen** across early L40 runs unless a major feasibility issue arises.

- **target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **baseline rank:** `r = 16`
- **baseline alpha:** `32`
- **dropout:** `0.05`
- **adapter dtype:** `bf16`

### justification
attention-only LoRA gives substantial adapter capacity while avoiding the extra trainable parameter growth and interpretability complications of adding MLP modules. keeping the target-module set fixed is important for clean LoRA vs QLoRA comparison.

### trainable parameter counts
these should be logged exactly at model initialisation. expected values are approximately:

- **r=16:** ~13.6M trainable parameters
- **r=32:** ~27.3M
- **r=64:** ~54.5M

exact values from the implementation should be treated as authoritative.

---

## 8. QLoRA configuration

these values are **frozen**.

| parameter | value |
|---|---|
| quantisation | 4-bit NF4 |
| double quantisation | true |
| compute dtype | bf16 |
| adapter config | identical to LoRA |

### canonical bitsandbytes configuration

    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

### justification

NF4 with double quantisation is the standard configuration used in the original QLoRA work and provides the quantised condition required by the study.

---

## 9. baseline evaluation policy

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

baseline evaluation allows measurement of:

1. adaptation improvement relative to the starting model  
2. the initial perplexity cost of quantisation before training

---

## 10. runs matrix for early L40 phase

all runs share identical:

- dataset
- seq_len
- optimiser configuration
- learning rate policy
- scheduler
- warmup
- evaluation schedule
- seed

### phase 1 — core comparison

| run | method | rank | alpha |
|---|---|---|---|
| 1 | LoRA (bf16 base) | 16 | 32 |
| 2 | QLoRA (4-bit base) | 16 | 32 |

### phase 2 — rank sweep (quantisation capacity)

| run | method | rank | alpha |
|---|---|---|---|
| 3 | QLoRA | 32 | 64 |
| 4 | QLoRA | 64 | 128 |

### note

runs 1 and 2 should be stabilised first before expanding to rank sweep experiments.

---

## 11. evaluation schedule

default early-run configuration:

- **eval every:** 50 steps
- **save every:** 50 steps
- **logging_steps:** 10

### metrics recorded per evaluation

- validation loss (overall)
- validation perplexity (overall)
- validation perplexity (layer A)
- validation perplexity (layer B)
- train step-time statistics over preceding window
- evaluation peak GPU memory
- memory summary statistics

### checkpoint selection rule

best checkpoint = **minimum overall validation perplexity**

layer-stratified metrics are retained for analysis but do not determine checkpoint selection.

---

## 12. checkpoint retention

- save adapter weights at each evaluation step
- retain:
  - best checkpoint by overall validation perplexity
  - final checkpoint

downstream evaluation uses **the best checkpoint**.

status: **frozen**

---

## 13. logging requirements

every run must emit structured artefacts sufficient for dissertation reporting.

required reports:

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
- `comparison_row.json`
- `run.log`

### justification

these artefacts support:

- intrinsic adaptation analysis
- memory efficiency comparison
- reproducibility
- automated cross-run aggregation

---

## 14. what the 3B local pilots justify

the completed 3B runs justify the following claims:

- the LoRA and QLoRA pipelines run end-to-end successfully
- seq_len=1024 is feasible with the current data pipeline
- evaluation and logging instrumentation work correctly
- the learning rate `2e-4` is a stable starting value for short pilot runs
- QLoRA shows the expected directional behaviour (lower VRAM, slower step time)

they **do not justify freezing**:

- final 8B batch size
- final 8B max_steps
- final 8B evaluation schedule
- final runtime expectations
- final conclusions about LoRA vs QLoRA quality differences

---

## 15. freeze criteria for protocol v3

after the first successful matched 8B LoRA and QLoRA runs on L40 hardware, the following parameters should be frozen:

- per_device_train_batch_size
- gradient_accumulation_steps
- effective_batch_size
- max_steps
- equivalent_epochs
- eval_steps
- save_steps
- gradient checkpointing policy
- final learning rate choice

only after these are confirmed should the protocol be treated as the **final experimental protocol for the dissertation comparison**.

---

## 16. acceptance criteria for early L40 phase

the early cluster stage is successful when:

1. LoRA 8B completes training at seq_len=1024.
2. QLoRA 8B completes training under matched optimisation settings.
3. both runs produce:
   - baseline perplexity
   - stratified held-out perplexity
   - memory summaries
   - step-time summaries
   - comparison-ready artefacts
4. any feasibility changes are documented and frozen before the full experiment matrix begins.

---

## 17. concise early-run constants summary

### frozen now

- model family and tokenizer
- seq_len = 1024
- dataset construction and split
- AdamW betas / eps / weight decay
- cosine scheduler
- warmup_ratio = 0.05
- gradient clip = 1.0
- seed = 42
- LoRA target modules
- LoRA baseline r=16 / α=32 / dropout=0.05
- QLoRA NF4 + double quantisation
- baseline evaluation before training
- best checkpoint selection rule

### tentatively fixed for first L40 runs

- learning rate = 2e-4
- fallback LR = 1e-4
- per_device_train_batch_size = 2
- gradient_accumulation_steps = 4
- effective_batch_size = 8
- max_steps = 750
- eval_steps = 50
- save_steps = 50
- gradient checkpointing = off

### frozen after first matched 8B runs

- final batch configuration
- final step budget
- final eval/save schedule
- equivalent epochs
- final learning rate confirmation
- checkpointing policy