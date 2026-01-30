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
  - tribunal corpus is now effectively materially complete as a pipeline validation, pending scope refinement and final corpus selection.
  - next step for layer a will be:
      - refactor `prepare_domain_corpus.py` to target `raw_txt/`
      - assign a stable source tag (e.g. `"uk_employment_tribunal"`)
      - generate `corpus.jsonl` + `corpus_stats.json`
      - create intrinsic train/val/test splits at document level
  - note: this ingestion was completed before finalising the revised experimental scope and served as a full-scale dry run of the tribunal pipeline rather than the final experimental corpus.

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

overall:
- layer a (tribunal decisions): **fully ingested and cleaned at scale**
- layer b (doctrine/guidance): **curated and staged for text conversion**
- domain corpus is now the main remaining blocker before:
    - domain LM training
    - lora vs qlora comparisons on intrinsic perplexity

## 18/12/2025

- **scope pivot: eur-lex → uk employment-law + downstream task**
  
- initial experiments were scaffolded around the eur-lex corpus to validate the peft/qlora training pipeline. however, further review showed that:
  - eur-lex is an eu-level, multi-domain legal corpus and does not reflect the linguistic or structural properties of uk employment law,
  - continued pretraining on eur-lex would confound the study by introducing broad legal variation rather than domain-specific adaptation,
  - and eur-lex benchmarks do not naturally support in-domain downstream evaluation aligned with uk employment practice.

the project scope was therefore refined to:
- focus on uk employment-law text (employment tribunal decisions + guidance/doctrine),
- evaluate domain-adaptive continued pretraining under quantisation,
- and add an in-domain supervised downstream task (employment tribunal section/paragraph classification) to assess whether intrinsic improvements transfer to structured legal understanding.

eur-lex assets were retained only for initial pipeline validation and excluded from all subsequent experiments.

## 20/12/2025

**tribunal corpus reset following scope refinement**

following the scope pivot (18/12), the previously ingested tribunal corpus (~5,000 decisions) was **intentionally discarded**.

rationale:
- the revised study design fixes total training tokens *before* training to isolate quantisation effects,
- layer-b token counts materially constrain how many tribunal decisions can be included,
- retaining the full scraped corpus would require post-hoc truncation or arbitrary downsampling.

the tribunal pipeline (scraper + pdf→txt cleaning) was therefore preserved, but **data collection was reset** to allow:
- pilot-based estimation of tokens per decision,
- controlled sampling to hit a fixed 1.8–2.4M token budget,
- transparent, reproducible corpus selection aligned with the final evaluation design.

## 29/12/2025

- **built layer b automated extraction pipeline:**
  - created `scripts/layer_b_pdf_to_txt.py` to handle Layer B corpus (ACAS guides, codes, doctrine, gov.uk)
  - adapted tribunal PDF cleaning logic with Layer B-specific patterns:
    - deterministic boilerplate removal: ACAS branding, copyright notices, gov.uk standard footers
    - page number detection and removal (standalone numbers, "page X of Y" patterns)
    - header/footer indicators (e.g., "ACAS", "Guidance", "Code of Practice")
    - hyphenation repair across line breaks
    - contents section detection and removal (heuristic-based)
    - whitespace normalization (collapse excessive blank lines, strip trailing spaces)
  - script processes all 4 subdirectories recursively: `acas_guides/`, `codes/`, `doctrine/`, `govuk/`
  - flattened output structure: `{source_type}__{filename}.txt` in `data/domain_corpus/layer_b_raw_extracted/`

- **successfully extracted layer b corpus:**
  - ran extraction pipeline on all 64 manually curated Layer B files
  - **63/64 files extracted successfully** with minimum 500 character threshold
  - output breakdown by source type:
    - acas_guides: 26 files
    - codes: 4 files  
    - doctrine: 7 files
    - govuk: 26 files
  - automated cleaning removed ~90% of deterministic noise:
    - page numbers, headers, footers ✅
    - copyright/branding boilerplate ✅
    - hyphenation artifacts ✅
    - excessive whitespace ✅

- **identified remaining context-dependent noise requiring manual review:**
  - **acas_guides**: navigation phrases embedded in substantive content
    - "Find out more about...", "Read our advice on...", "Get more advice and support"
    - cross-references that sometimes contain legal substance, sometimes pure navigation
    - example: "For more information including how to apply..." (useful) vs "Find out more..." (navigation)
  - **codes**: 
    - forewords (may contain useful legal status info)
    - scattered references: "More comprehensive advice and guidance is contained in..."
  - **doctrine**:
    - academic formatting: footnotes, reference numbers (38, 39, 40...)
    - disclaimer boilerplate at document start
    - contact information sections for external services
    - directory/contents sections at start/end
  - **govuk**:
    - inline path references: "(/employment-status/worker)", "(/call-charges)"
    - embedded navigation breadcrumbs
    - contact info: "Contact Acas... Monday to Friday, 8am to 6pm"

- **decision: proceed with documented manual cleaning pass**
  - automated extraction saved majority of work (deterministic patterns handled)
  - remaining noise is semantically context-dependent and cannot be removed deterministically without unacceptable false positives.
  - estimated effort: 63 files × 5-10 min each = 5-6 hours focused work
  - approach:
    - review each `layer_b_raw_extracted` file
    - manually remove context-dependent navigation/boilerplate while preserving substantive legal content
    - save cleaned versions to `data/domain_corpus/layer_b_cleaned/`
    - maintain `cleaning_notes.md` documenting major deletions per file for dissertation methodology section
  - rationale: ensures corpus quality + provides intimate knowledge of corpus for methodology writeup + creates audit trail for reproducibility discussion. while this introduces a limited non-automated step, all deletions are documented per file and the pre-cleaned corpus is retained to enable auditability and discussion of reproducibility trade-offs.

- **refined extraction script for defensibility:**
  - removed aggressive digit-only page number pattern (prevents false positives on numbered sections)
  - added table-of-contents verification (requires 2+ ToC indicators before removal)
  - implemented audit trail: `extraction_summary.json` tracks per-file cleaning actions (chars removed, lines dropped)
  - improved pattern documentation for dissertation methodology section

- **next steps (in priority order):**
  1. manual cleaning pass on 63 Layer B extracted files → `layer_b_cleaned/`
  2. tokenize Layer B cleaned corpus using LLaMA-3 tokenizer → measure actual token counts by source type
  3. pilot Layer A: convert 150 tribunal PDFs → txt → tokenize → calculate median tokens/decision
  4. calculate Layer A scraping target: if Layer B = X tokens, need 3-4X from tribunals; determine additional scraping required
  5. complete Layer A ingestion to hit target corpus size (1.8-2.4M tokens total)

- **corpus construction principle:**
  - all experiments fix token budgets and selection criteria *before* training; data is re-collected when scope changes rather than retroactively trimmed.

## 29/01/2025

### layer b: manual cleaning completed + token budget established

- completed **manual cleaning pass** over all extracted Layer B text files in  
  `data/domain_corpus/layer_b_raw_extracted/`
- edits were strictly subtractive:
  - removed navigation, cross-link prompts, contact info, and duplicated boilerplate
  - did **not** rewrite, paraphrase, or summarise legal content
  - preserved borderline cases where removal risked deleting legal meaning
- original extracted texts and source PDFs retained for auditability

#### cleaning decisions by source type

- **acas_guides**
  - removed:
    - “Find out more…”, “Read more about…”, “Get advice and support…”
    - embedded navigation lists and repeated section signposting
  - retained:
    - explanatory paragraphs even when informal in tone
    - examples illustrating how rules apply in practice
  - rationale: ACAS guidance reflects *practical interpretation* used by tribunals

- **acas codes of practice**
  - removed:
    - publication metadata and references to other ACAS documents
  - retained:
    - forewords and statutory status explanations
  - rationale: codes have quasi-legal weight and are frequently cited in judgments

- **doctrine (hoc briefings / explanatory notes)**
  - removed:
    - footnote numbering, inline citation markers
    - contents pages, contact sections, disclaimers
  - retained:
    - policy background and legislative history, in-line page numbers
  - rationale: provides doctrinal framing used implicitly in tribunal reasoning. page numbers never broke up words

- **gov.uk guidance**
  - removed:
    - inline hyperlink artifacts (e.g. `(/employment-status/worker)`)
    - breadcrumbs, contact hours, call charges
  - retained:
    - threshold rules, statutory definitions, entitlement explanations
  - rationale: high signal density despite templated structure

- decision: tolerate minor formatting artefacts where removal risked semantic loss

### layer b tokenisation
- tokenised cleaned Layer B corpus using **LLaMA-3 tokenizer** (vocab size 128,256)
- script outputs:
  - `layer_b_token_stats.json`
  - `layer_b_token_stats.csv`

#### tokenisation summary
- total files: **60**
- total tokens: **200,237**
- mean tokens/file: **3,337**
- median tokens/file: **2,461**

**by source type:**
- acas_guides: 95,441 tokens (≈48%)
- codes: 20,164 tokens (≈10%)
- doctrine: 43,081 tokens (≈21%)
- govuk: 41,551 tokens (≈21%)

- largest outlier:
  - `doctrine__keyemploymentrights.txt` (~26k tokens)
  - solution: will be downweighted 

#### implication
- Layer B is **smaller than initially anticipated** (≈200k tokens vs planned 400–700k)
- this is acceptable:
  - signal density is high
  - Layer B is intended as *contextual grounding*, not primary adaptation signal
- Layer B will likely constitute **~10–15%** of total training tokens

### revised corpus weighting (provisional)
- **Layer A (tribunals):** target **1.6–1.8M tokens**
- **Layer B (guidance/doctrine):** fixed at **~200k tokens**
- combined total: **~1.8–2.0M tokens**

rationale:
- keeps tribunal judgments dominant
- preserves doctrinal grounding
- avoids “more data helps” confound in quantisation comparison

### layer a pilot: tribunal token estimation
- implemented stricter short-document filter in `pdf_to_txt.py`
  - discard outputs `< 1000 characters`
- ran pilot on **150 scraped tribunal PDFs**

#### pilot observations
- substantial variance in extracted length:
  - many decisions are legitimately short (orders, summary decisions)
  - others are long, multi-thousand-line judgments
- some PDFs contain:
  - scanned pages → no extractable text
  - boilerplate-only pages incorrectly flagged
- current filter is conservative but imperfect

#### next step (planned)
- manually inspect ~30–50 pilot outputs:
  - short but valid judgments
  - dropped cases that should be retained
  - long cases with extraction artefacts
- refine:
  - boilerplate detection thresholds
  - minimum-length criteria (likely switch from chars → tokens)
- re-run pilot tokenisation to estimate:
  - median tokens per retained decision
  - variance across jurisdictions


### layer a planning logic
once pilot stats are finalised:

1. compute median tokens per tribunal decision
2. derive required number of decisions to reach target Layer A token count
3. rescrape only required number (no post-hoc truncation)
4. fix corpus **before training** to preserve experimental control

### methodological note
- minor residual noise is tolerated in both layers
- noise is **systematic and symmetric** across LoRA and QLoRA conditions
- therefore it does not bias comparative results
- trade-off explicitly documented for dissertation reproducibility discussion