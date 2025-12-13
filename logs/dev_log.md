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

- clarified corpus design for **Layer A** (UK Employment Tribunal decisions):
  - decided to target *substantive employment rights* jurisdiction codes only (e.g. unfair dismissal, discrimination, wages, working time, maternity, whistleblowing, redundancy, etc.) and **exclude** procedural / meta-only stuff (pure time limits, practice & procedure, CAC, Certification Officer, tax, national security, etc.)
  - constructed a filtered `gov.uk` search URL that already restricts to post-2020 and the chosen jurisdiction codes, reducing noise before scraping

- added **HTML scraper + PDF downloader**: `scripts/scraper_downloader.py`
  - iterates paginated `employment-tribunal-decisions` search results via a `page` query param helper
  - extracts decision links with a simple heuristic:
    - only `<a>` under `/employment-tribunal-decisions/...` whose text contains `" v "` (to avoid nav/header links)
  - for each decision:
    - fetches the decision page
    - finds the *asset URL* on `assets.publishing.service.gov.uk` ending in `.pdf`
  - introduced a stable naming convention for PDFs:
    - parse case id from the HTML title: `(\d{5,7})/(\d{4})` → `case_id = 6010118_2024`
    - slugify the title prefix: `"Ms P Bannor v 39QGG Management Ltd and Others"` → `Ms_P_Bannor_v_39QGG_Management_Ltd_and_Others`
    - append original pdf filename tail: `...__Ms_P_Bannor_v_39QGG_Management_Ltd___Others_-_6010118-2024.pdf`
    - final file name truncated/sanitised with `safe_filename` to avoid illegal chars but **preserve `.pdf`**
  - added:
    - custom `User-Agent` (“employment-law-research-bot/0.1”)
    - polite delay between downloads
    - PDF magic-header check (`%PDF-`) to avoid writing HTML/garbage as `.pdf`
    - basic stats: downloaded vs skipped count
  - verified against the filtered URL: ~1 pdf/sec, tested on first ~150 decisions; filenames look clean and consistent, no more extensionless junk

- built **PDF → cleaned text** converter: `scripts/pdf_to_txt.py`
  - uses `pdfplumber` to extract page text
  - per-page filtering:
    - heuristic `looks_like_boilerplate` to drop:
      - “the employment tribunals (interest) order 1990”
      - “public access to employment tribunal decisions”
      - “recording and transcription / joint presidential practice direction...”
  - normalisation & quality passes:
    - **hyphenation repair**: `"discrimi-\nnation"` → `"discrimination"` using `(\w+)-\n(\w+) → \1\2`
    - **admin tail trimming**:
      - scan final ~40 lines for admin phrases:
        - “judgment sent to the parties…”
        - “for the tribunal office”
        - “judgment – rule 61 march 2017” and variants
        - “note:” blocks referring to “written reasons” / “rule 61”
      - if found *and* doc has >10 lines, truncate from that point downwards
      - secondary hard cut via substring search on `TRUNCATION_PATTERNS` (`judgment sent to the parties on`, etc.) as a last resort
      - strip any remaining rule-footer lines containing “judgment - rule 61 march 2017”
    - whitespace cleanup:
      - rstrip every line
      - collapse multiple blank lines → max 1 consecutive blank line
      - strip leading/trailing blank lines
  - writes cleaned `.txt` to `data/domain_corpus/raw_txt/` with same stem as PDF
  - warns on very short outputs (`< 200 chars`) so we can inspect weird edge cases (e.g. pure orders, corrupted PDFs)
  - tested on 20+ random cases including short 1–2 page decisions and long multi-page judgments; output looks like “pure judgment + reasons” with admin garbage removed

- current **corpus pipeline state**:
  - **ready**:
    - `scripts/scraper_downloader.py` can pull thousands of filtered tribunal PDFs into `data/domain_corpus/raw_pdfs/`
    - `scripts/pdf_to_txt.py` converts these into reasonably clean plain text in `data/domain_corpus/raw_txt/`
  - **stubbed / to adjust**:
    - `prepare_domain_corpus.py` currently:
      - assumes `RAW = ../data/domain_corpus/raw`
      - simple cleaning → `processed/corpus.jsonl` with `source = "education_law_raw"`
    - this needs to be updated to:
      - point `RAW` to `raw_txt` (employment tribunal txts, not generic “raw”)
      - refine cleaning (or reuse the logic already in `pdf_to_txt` so we don’t double-mangle)
      - set a sensible `source` tag (e.g. `"uk_employment_tribunal"`), and possibly keep metadata hooks for case_id / year if we want it later

overall: today’s work essentially built **Layer A ingestion** skeleton: we can (when ready) scrape ~5k+ tribunal decisions, convert them into cleaned text, and adapt `prepare_domain_corpus.py` to turn them into a domain corpus jsonl for intrinsic LM evaluation and fine-tuning. next concrete step is to refactor `prepare_domain_corpus.py` to target `raw_txt`, preserve basic metadata, and generate `corpus.jsonl` + `corpus_stats.json` on the new employment-law data.

## 13/12/2025

- **completed large-scale ingestion for layer a (tribunal decisions):**
    - ran `scraper_downloader.py` end-to-end on the filtered gov.uk employment tribunal search (post-2020, substantive jurisdictions only)
    - successfully downloaded **~5,000 tribunal decision PDFs** into `data/domain_corpus/raw_pdfs/`
    - no significant scraping failures observed beyond expected duplicates / missing assets
    - download rate stable with polite delay; no blocking encountered
- **bulk conversion of layer a PDFs → cleaned text:**
    - ran `pdf_to_txt.py` across the full tribunal batch
    - applied existing boilerplate removal, hyphen repair, admin footer trimming, and whitespace normalisation
    - **discarded very short outputs** (<300 characters) to remove:
        - pure orders with no reasons
        - corrupted or empty PDFs
        - metadata-only documents
    - retained only substantive judgments with meaningful reasoning
    - resulting corpus is now suitable for intrinsic LM training and evaluation
- **layer a status:**
    - tribunal corpus is now effectively “frozen” pending final pass through `prepare_domain_corpus.py`
    - next step for layer a will be:
        - refactor `prepare_domain_corpus.py` to target `raw_txt/`
        - assign a stable source tag (e.g. `"uk_employment_tribunal"`)
        - generate `corpus.jsonl` + `corpus_stats.json`
        - create intrinsic train/val/test splits at document level
- **layer b corpus curation (doctrine + guidance):**
    - compiled a curated list of **authoritative employment-law doctrine sources**, split into distinct sub-collections:
        - **ACAS guidance PDFs** (practical interpretation used by tribunals)
        - **ACAS Codes of Practice** (statutory relevance)
        - **House of Commons Library briefings** (doctrinal summaries and context)
        - **gov.uk employment law topic pages** (statutory rights explanations and thresholds)
    - these sources were selected because they:
        - reflect how employment law is interpreted and applied in practice
        - are frequently cited or relied upon in tribunal reasoning
        - provide a complementary signal to raw tribunal judgments (cleaner, expository text)
- **layer b ingestion approach:**
    - all layer b documents were **downloaded manually** to ensure:
        - source quality
        - topical relevance
        - avoidance of irrelevant boilerplate or duplicated content
    - documents are currently stored in structured subdirectories (e.g. `acas_guides/`, `codes/`, `doctrine/`, `govuk/`)
    - next step will be to implement:
        - lightweight converters for each format type (PDF-based vs HTML-derived)
        - reuse of the existing text normalisation logic where appropriate
        - integration into `prepare_domain_corpus.py` with distinct source tags per sub-collection
- **overall state at end of day:**
    - layer a (tribunal decisions): **fully ingested and cleaned at scale**
    - layer b (doctrine/guidance): **curated and staged for text conversion**
    - domain corpus is now the main remaining blocker before:
        - domain LM training
        - lora vs qlora comparisons on intrinsic perplexity