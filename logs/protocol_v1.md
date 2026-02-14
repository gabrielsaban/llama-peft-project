# experimental protocol v1

## 1. model

- base model: **LLaMA-3-8B**
- calibration model: **LLaMA-3.2-3B** — local validation on 3070 Ti before committing to 8B
- tokenizer: LLaMA-3 (vocab 128,256)
- context length: **1024**
  - justification: prioritises update count over long-context modelling; study measures optimisation dynamics, not context utilisation

## 2. data

- total tokens: **1,802,427**
- layer A (tribunal decisions): 1,602,190 (88.9%), 170 files
- layer B (guidance/doctrine): 200,237 (11.1%), 60 files
- train/val split: **85/15** per layer, document-level, stratified by post-cap document token length (quartile buckets)
- split seed: **42**
- truncation: 20,000 tokens max per document, paragraph-boundary aware
- manifest: `data/domain_corpus/corpus_manifest.json`

## 3. optimisation

- optimiser: **AdamW**
  - betas: (0.9, 0.999)
  - eps: 1e-8
  - weight_decay: 0.01
- lr: **TBD** — chosen via 3B calibration run to avoid divergence, not to maximise perplexity. starting point 2e-4, fall back to 1e-4 if unstable.
- scheduler: **cosine**
- warmup_ratio: **0.05**
- gradient_clip: **1.0**
- seq_len: **1024**
- effective_batch_size: **TBD** — `batch_size × grad_accum`, set during calibration
- max_steps: **TBD** — fixed across all conditions; derived from target epoch-equivalents × sequences_per_epoch (max_steps & effective_batch fixed; equivalent epochs reported for interpretability only)
- total_tokens_processed: `max_steps × seq_len × effective_batch_size`
- equivalent_epochs: **TBD** — expected range 3–5

justification: all optimisation parameters are fixed identically across LoRA and QLoRA conditions. any observed differences are attributable to quantisation, not training recipe.

## 4. memory controls

- gradient checkpointing: **off**
  - justification: cleaner VRAM comparison, fewer moving parts. bf16 LoRA 8B shouldn't struggle to fit on L40s
  - if bf16 lora does not fit at target effective batch, we reduce batch before enabling checkpointing; checkpointing is used only if required for feasibility, and then enabled for all conditions.
- flash attention: **on where available** (otherwise consistent across all conditions)
- mixed precision: **bf16** throughout

## 5. LoRA configuration (fixed across conditions)

- target modules: **q_proj, k_proj, v_proj, o_proj** (attention-full)
  - justification: sufficient adapter capacity without inflating trainable params via MLP modules; keeps rank sweep interpretable
- r: 16 (baseline)
- α: 32
- dropout: 0.05
- dtype: bf16
- trainable params (r=16): **~TBD** — computed and logged at init

## 6. QLoRA configuration

- quantisation: **4-bit NF4**
- double quantisation: **true**
- compute dtype: **bf16**
- adapter config: identical to §5
- bnb config: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`

## 7. runs matrix

all runs share identical: max_steps, effective_batch_size, lr, scheduler, warmup, eval schedule, data split.

| run | method | r | α | notes |
|-----|--------|---|---|-------|
| 1 | LoRA (bf16 base) | 16 | 32 | baseline |
| 2 | QLoRA (4-bit base) | 16 | 32 | main comparison |
| 3 | QLoRA (4-bit base) | 32 | 64 | rank sweep |
| 4 | QLoRA (4-bit base) | 64 | 128 | rank sweep |

- lora rank fixed at r=16; sweep performed only for qlora to isolate quantisation-capacity interaction
- training seed: **42** (single seed per condition)
- justification: single seed is standard for dissertation-scale work; stated explicitly for reproducibility

note: run 2 serves double duty as both the main LoRA/QLoRA comparison and the first rank sweep data point. α scales linearly with r (α = 2r) to hold effective learning rate constant.

## 8. evaluation schedule

- eval_every: **TBD** steps — target ~10–15 eval points per run
- metrics per eval:
  - val loss (overall)
  - val perplexity (overall)
  - val perplexity (layer A held-out)
  - val perplexity (layer B held-out)
  - peak GPU memory (max over run (reset at start))
  - step time (mean over eval interval)
- baseline measurement: evaluate **unadapted base model** on val set before any training (zero-shot reference)

## 9. checkpointing

- save adapter weights at every eval step
- keep: **best checkpoint by overall val perplexity** + final checkpoint
- downstream evaluation uses: best checkpoint by overall val perplexity

## 10. downstream task (stub)

- task: employment tribunal section/paragraph classification
- dataset: **TBD** — annotation scheme and source to be defined
- model: frozen adapted base + linear classification head
  - head input: last token hidden state
  - representation: eos token (or mean pooling) — to be fixed when downstream dataset finalised.
- classifier training:
  - freeze adapted model entirely
  - lr: 1e-3
  - epochs: 10–20 with early stopping by val F1
- evaluation: macro-F1, micro-F1
- which checkpoint: best overall val perplexity from §9

## 11. calibration run (3B on 3070 Ti)

purpose: validate plumbing + estimate step time trends + pick batch/accum strategy

- model: LLaMA-3.2-3B (base)
- GPU: 3070 Ti (8GB)
- steps: short run (~50–100 steps)
- outputs:
  - confirm lr=2e-4 does not diverge
  - measure step time → estimate wall time for 8B
  - measure peak VRAM
  - validate data loading, eval loop, logging, checkpoint saving
- after calibration: freeze all TBD values in this protocol and increment to v2
