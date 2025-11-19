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