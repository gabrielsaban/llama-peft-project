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