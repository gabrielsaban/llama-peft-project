## 19-11-2025

- initialised repo with structure: src/, data/, configs/, logs/, scripts/
- installed miniconda inside WSL2 (Python 3.11), resolved conda activation issues
- installed PyTorch 2.3.1 CUDA 12.1, transformers, datasets, peft, bitsandbytes
- successfully logged in to HuggingFace with fine-grained token
- verified gpt2 load + lexglue eur-lex load
- designed dataset structure: data/lexglue/, data/domain_corpus/, data/splits/

- wrote and committed scripts:
  - prepare_eurlex.py (freeze official splits)
  - prepare_domain_corpus.py (raw → jsonl)
  - create_intrinsic_splits.py (train/val/test)
- generated eurlex_splits.json and eurlex_sample.json

- confirmed repo ready for domain-corpus ingestion

- wrote initial config file: configs/llama3_1b_lora_eurlex_debug.yaml
  - defined model dtype (bf16), LoRA hyperparams (r=16, α=32, dropout=0.05)
  - defined dataset settings (eurlex_text_lm, seq_len=1024, subset sizes)
  - defined training hyperparams (lr=2e-4, batch=2, grad_accum=4)
  - included hardware settings (no 4bit, gradient_checkpointing initially true)

- implemented src/data_module.py:
  - eur-lex text loader
  - tokenisation with no truncation
  - concatenation + fixed-length chunking for causal LM
  - DataCollatorForLanguageModeling(mkm=False)
  - correctly handles train_subset / val_subset

- added src/train_lora.py:
  - yaml parsing
  - dtype resolution
  - base model loading (bf16 or 4bit path)
  - LoRA adapter attachment via LoraConfig + get_peft_model
  - optional gradient checkpointing support
  - HuggingFace Trainer setup
  - model + tokenizer saving after train

- resolved two training bugs:
  1) `learning_rate` loaded as string → fixed by casting to float inside train_lora
  2) PEFT + gradient checkpointing triggering:
       "None of the inputs have requires_grad=True" and loss has no grad_fn
     → temporarily disabled gradient_checkpointing for 1B debug runs

- launched first prototype LoRA run on local 3070 Ti:
  - model = LLaMA-3.2-1B-Instruct
  - seq_len=1024, batch=2, grad_accum=4
  - 14,783 training chunks produced
  - training initialised correctly
  - training speed ~1 step/sec on 3070 Ti (expected)
  - currently running without divergence

overall: full pipeline (data → tokenizer → model → LoRA → Trainer) is working end-to-end.
 
## 09/12/2025

