## experiment log

### exp-001: 3b lora domain calibration (4070)

date
- 2026-02-17

objective
- validate end-to-end lora plumbing for domain-corpus calibration
- check baseline stability & feasibility
- assign tbd variables for protocol_v2.md

run context
- uni desktop with rtx 4070

config used
- file: configs/llama3_3b_lora_domain_calibration.yaml
- model: meta-llama/Llama-3.2-3B
- data/flags: max_seq_length=1024, gradient_checkpointing=false

what happened
1. training started and failed on first backward pass with:
- torch.cuda.OutOfMemoryError (allocation failure ~502MiB).
- vram visibly spiked rapidly to 11/12GB during first steps
- no competing compute jobs

intepretation
- current local lora baseline with seq_leq=1024 not feasible on 12gb vram
- identifies a local feasibility boundary

### exp-002: 3b lora successful run summary (4070)

config used
- file: configs/llama3_3b_lora_domain_calibration.yaml
- model: meta-llama/Llama-3.2-3B
- data/flags: max_seq_length=512, gradient_checkpointing=false

run summary
- global step reached: 75
- epoch reached: 0.19940174

final eval metrics (step 75):
- `eval_loss`: `2.2668778896331787`
- `eval_layer_a_loss`: `2.3079824447631836`
- `eval_layer_b_loss`: `1.8172916173934937`

throughput at final eval:

- overall eval steps/s: `12.526`
- Layer A eval steps/s: `12.609`
- Layer B eval steps/s: `12.614`

eval trend across checkpoints
- Step 25:
    - `eval_loss`: `2.2856974601745605`
    - `eval_layer_a_loss`: `2.327223300933838`
    - `eval_layer_b_loss`: `1.8313065767288208`
- Step 50:
    - `eval_loss`: `2.2710487842559814`
    - `eval_layer_a_loss`: `2.3122615814208984`
    - `eval_layer_b_loss`: `1.8201690912246704`
- Step 75:
    - `eval_loss`: `2.2668778896331787`
    - `eval_layer_a_loss`: `2.3079824447631836`
    - `eval_layer_b_loss`: `1.8172916173934937`

training stability signals captured
- Logged train loss points:
    - step 10: `2.3488`
    - step 20: `2.2103`
    - step 30: `2.2841`
    - step 40: `2.2434`
    - step 50: `2.1889`
    - step 60: `2.2503`
    - step 70: `2.2815`
- Logged grad norm points:
    - step 10: `0.3754`
    - step 20: `0.3913`
    - step 30: `0.3827`
    - step 40: `0.3474`
    - step 50: `0.4461`
    - step 60: `0.3968`
    - step 70: `0.4024`

software stack
- Python: `3.12.12`
- torch: `2.3.1+cu121`
- torch CUDA runtime: `12.1`
- CUDA available in torch: `True`
- transformers: `4.46.2`
- peft: `0.13.2`
- bitsandbytes: `0.44.1`
- datasets: `3.0.1`
- huggingface_hub: `0.36.0`

gpu / driver (`nvidia-smi`)
- GPU: `NVIDIA GeForce RTX 4070`
- VRAM: `12282 MiB` (12 GB class)
- Driver version: `590.48.01`
- Reported system CUDA version: `13.1`
- Idle snapshot memory usage at capture: `462 MiB / 12282 MiB`
- Display stack active (Xorg/gnome-shell/VS Code/Firefox) at capture time.

host hardware
- CPU: `12th Gen Intel(R) Core(TM) i7-12700`
- Logical CPUs: `20`
- Sockets: `1`
- Threads per core: `2`
- Cores per socket: `12`
- RAM: `30 GiB` total (`free -h`), swap `15 GiB`

not captured by current trainer flow
- peak vram & mean train step time not explicitly logged
- baseline pre-training eval not automated into artifacts
- best-checkpoint enforcement (metric_for_best_model + load_best_model_at_end) not enabled 
- flash attention (will only consider chasing if bottlenecked on l40s)

### exp-003: 3b lora seq=1024 retry on 4070 (instrumented, gc off) -> oom

date
- 2026-02-24

objective
- rerun the 3b lora seq=1024 local feasibility case after trainer instrumentation upgrades
- capture baseline eval artifact + explicit cuda memory diagnostics before/at oom

config used
- file: `configs/llama3_3b_lora_domain_calibration.yaml`
- model: `meta-llama/Llama-3.2-3B`
- data/flags: `max_seq_length=1024`, `gradient_checkpointing=false`
- trainer features active: baseline pre-train eval, best-checkpoint enforcement, protocol metrics, cuda memory snapshots

what happened
1. baseline pre-training eval completed successfully and artifacts were written.
2. memory snapshots were captured after baseline eval cache residency and after explicit cache clear (`post_baseline_eval_preclear`, `pre_train`).
3. training entered step 0 and failed on the first backward pass with `torch.cuda.OutOfMemoryError`.

key artifacts / signals
- output dir: `outputs/llama3_3b_lora_domain_calibration`
- `baseline_eval_metrics.json` written (new instrumentation working)
- `cuda_memory_pre_train.json` written
- `cuda_memory_oom_exception.json` written
- `cuda_memory_oom_summary.txt` written

selected metrics
- baseline overall ppl: `9.137979489678585`
- baseline layer A ppl: `9.512587567361257`
- baseline layer B ppl: `5.805942986192581`
- baseline eval peak VRAM (allocated / reserved): `7.2734 / 7.4063 GB`
- pre-train snapshot reserved VRAM after clear: `6.5305 GB`
- OOM snapshot max reserved VRAM: `~10.63 GB` (`11408506880` bytes)

interpretation
- the lora seq=1024 run remains infeasible on the local 4070 without gradient checkpointing, even after improved cache hygiene and diagnostics.
- this run is still valuable because it validates the upgraded artifact pipeline and gives a defensible memory-boundary record instead of a terminal-only crash.

### exp-004: 3b lora seq=1024 on 4070 with gc on (successful feasibility/plumbing run)

date
- 2026-02-24

objective
- test the minimal memory-control intervention needed to preserve `seq_len=1024` for local lora feasibility
- validate the gc-enabled lora path after fixing the gradient-checkpointing + lora autograd issue (`enable_input_require_grads`)

config used
- same base yaml as exp-003: `configs/llama3_3b_lora_domain_calibration.yaml`
- only substantive feasibility change: `gradient_checkpointing=true`
- output dir used for isolation: `outputs/llama3_3b_lora_domain_calibration_gc`

why gc was chosen over lowering seq length
- preserves protocol-relevant context length (`1024`) for a stronger local feasibility statement
- isolates the memory-control intervention instead of changing the training regime (lower seq alters optimisation/context behaviour)
- provides cleaner evidence for later L40s pilots: whether lora@1024 needs checkpointing vs not

run summary
- completed successfully to `global_step=75`
- best checkpoint selected and loaded by `eval_perplexity` (checkpoint-75)
- train runtime: `508.5208s` (~8m29s)
- train loss: `2.106833979288737`

baseline (pre-train) metrics
- baseline overall ppl: `9.137979489678585`
- baseline layer A ppl: `9.512587567361257`
- baseline layer B ppl: `5.805942986192581`
- baseline eval peak VRAM (allocated / reserved): `7.2734 / 7.4063 GB`

final eval metrics (step 75)
- `eval_loss`: `2.1519527435302734`
- `eval_perplexity`: `8.601638802452952`
- `eval_layer_a_loss`: `2.19038987159729` (`ppl=8.938697379671744`)
- `eval_layer_b_loss`: `1.7192983627319336` (`ppl=5.5806115252768205`)

protocol metrics (captured)
- final interval mean train step time: `3.6621950063999975 s`
- final eval peak VRAM (allocated / reserved): `7.7801 / 8.1680 GB`

eval trend (overall ppl)
- step 25: `8.73842401347553`
- step 50: `8.634866147953371`
- step 75: `8.601638802452952`

interpretation
- lora at seq=1024 is locally feasible on 4070 once gc is enabled and the lora+gc autograd fix is applied.
- this is a strong plumbing/feasibility result, not a basis for freezing final dissertation protocol constants for L40s.

### exp-005: 3b qlora seq=1024 on 4070 (successful feasibility run)

date
- 2026-02-24

objective
- run matched qlora local feasibility calibration at `seq_len=1024` with upgraded trainer artifacts
- compare memory/time/quality signals against the lora local feasibility runs (especially exp-004 gc-on)

config used
- file: `configs/llama3_3b_qlora_domain_calibration.yaml`
- `use_4bit=true`, NF4 + double quant + bf16 compute
- `max_seq_length=1024`, `gradient_checkpointing=false`
- `attn_implementation=sdpa` with explicit SDPA backend toggles logged

run summary
- completed successfully to `global_step=75`
- best checkpoint selected and loaded by `eval_perplexity` (checkpoint-75)
- train runtime: `882.6005s` (~14m43s)
- train loss: `2.138427365620931`

baseline (pre-train) metrics
- baseline overall ppl: `9.536033282186954`
- baseline layer A ppl: `9.925970166141147`
- baseline layer B ppl: `6.0619542541924485`
- baseline eval peak VRAM (allocated / reserved): `4.1301 / 4.8594 GB`

final eval metrics (step 75)
- `eval_loss`: `2.178673505783081`
- `eval_perplexity`: `8.834579464109739`
- `eval_layer_a_loss`: `2.216827869415283` (`ppl=9.178170285076245`)
- `eval_layer_b_loss`: `1.7485909461975098` (`ppl=5.746499841228997`)

protocol metrics (captured)
- final interval mean train step time: `6.340748476640001 s`
- final eval peak VRAM (allocated / reserved): `5.5308 / 6.1113 GB`

eval trend (overall ppl)
- step 25: `9.005389991839246`
- step 50: `8.874519090266665`
- step 75: `8.834579464109739`

comparative notes vs exp-004 (local 4070 only)
- qlora is substantially more memory-efficient than lora+gc at seq=1024 on this hardware (`~6.11 GB` vs `~8.17 GB` eval peak reserved)
- qlora is materially slower in this local setup (`~6.34s` vs `~3.66s` mean train step time)
- both runs show stable learning and improved held-out ppl over 75 steps
- lora+gc achieved lower final ppl than qlora in this short local calibration, but this is not sufficient to generalise to 8b/l40s protocol decisions

overall interpretation (exp-003 to exp-005)
- local 4070 runs are now serving their intended role well: proving pipeline correctness, validating instrumentation, and mapping feasibility boundaries.
- they provide useful directional signals (memory/time tradeoffs, failure modes, stability), but only limited evidence for final protocol constant selection on l40s hardware.
