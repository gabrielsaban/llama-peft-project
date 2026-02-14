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

---

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

---

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

---

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

---

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

---

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

---

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

---

## 30/01/2026 - 01/02/2026

### milestone: data → tokenisation pipeline completed (layer a + layer b)

- completed end-to-end pipeline from **raw documents → cleaned text → reflowed text → tokenisation-ready corpus** for both layers
- extensive iterative testing and optimisation focused on reducing pdf artefacts while preserving legal structure
- added verbose, per-file metric logging at key stages for auditability and reproducibility

### layer a: tribunal pdf extraction stabilised

- substantially revised `tribunal-pdf_to_txt.py` to improve robustness across heterogeneous tribunal pdf layouts

#### key changes / decisions

- **margin-cropped extraction**
  - crop top/bottom margins before text extraction to remove headers/footers early

- **within-page boilerplate truncation**
  - truncate footer boilerplate inside pages instead of dropping whole pages
  - avoids deleting mixed-content pages with legitimate reasoning

- **explicit header/footer artefact removal**
  - removes case numbers, page markers, separators, leaked metadata post-join

- **front-matter removal**
  - drop everything before first uppercase judgment-style heading
  - reduces noise from coversheets and publication metadata

- **multi-stage trailing admin trimming**
  - removes “judgment sent to the parties”, rule 61 notes, tribunal office footers
  - final safety trim catches leaked judge/approval lines near document end

- **normalisation**
  - left-strip lines, collapse blank lines, trim edges
  - preserves paragraph boundaries for downstream reflow

#### quality filtering

- hard keep rules:
  - **≥500 words**
  - **must contain paragraph numbering**
- rationale:
  - filters orders, stubs, and extraction failures
  - paragraph numbering used as proxy for structured judicial reasoning

#### logging

- per-file jsonl metrics written to `logs/pdf_to_txt_metrics.jsonl`, including:
  - page character counts
  - which cleaning stages triggered
  - word count
  - paragraph numbering + quality score

### layer a: reflow stage added

- introduced `reflow_tribunal_text.py` as a dedicated transformation:
  - `raw_txt → raw_txt_reflow`
- separation allows extraction logic and formatting reconstruction to evolve independently

#### reflow behaviour (high-level)

- single marker parser for numbered, decimal, alpha, roman, dash, and bullet lists
- removes pdf artefact blank lines inside lists
- aggressive line merging within paragraphs
- conservative around headings and structural boundaries
- preserves nesting and avoids inserting blanks inside sublists

#### logging

- per-file metrics written to `logs/reflow_metrics.jsonl`:
  - marker count
  - input vs output line counts

### layer b: status

- no methodological change
- full pipeline integrated and tokenisation-ready
- layer b remains a fixed, high-signal contextual corpus

### methodological note

- preprocessing introduces systematic transformations but:
  - scripts are deterministic and version-controlled
  - applied identically across all training conditions
  - supported by per-file metrics
- residual noise treated as symmetric preprocessing error, not a comparative confound

### current status

- both layers complete up to tokenisation-ready text
- layer a pipeline now includes:
  - robust extraction
  - explicit quality gating
  - dedicated reflow stage
  - auditable logs at each step

### next steps

- run pilot tokenisation on **post-reflow layer a** to estimate:
  - median tokens per decision
  - variance and outliers
- spot-check a small stratified sample of kept vs dropped decisions
- consider switching minimum-length filter from **words → tokens**
- freeze preprocessing scripts and corpus snapshot before training

---

## 13/02/2026

### layer a: pilot tokenisation and corpus sizing

- switched minimum-length filter in `tribunal-pdf_to_txt.py` from **word count (≥500)** to **LLaMA-3 token count (≥750)**
  - aligns quality filtering with the actual training budget unit
  - tokenizer loaded once at startup and passed through to `convert_pdf()`
- added `--dropped-dir` flag to `tribunal-pdf_to_txt.py`
  - rejected files written to `data/domain_corpus/raw_txt_dropped/` with reason-prefixed filenames (e.g. `short_158tok__case_name.txt`)
  - enables spot-checking of borderline rejections without re-running the pipeline
- created `preprocess/tokenise_pilot.py`
  - tokenises post-reflow `.txt` files using LLaMA-3 tokenizer
  - computes summary statistics (mean, median, std, quartiles, IQR)
  - detects outliers via IQR method
  - outputs budget estimate based on median tokens/decision and target corpus size
  - writes `pilot_token_stats.json` and `pilot_token_stats.csv`

### initial pilot (150 PDFs)

- ran pipeline end-to-end on 150 existing PDFs
- survival rate: 40/150 (26.7%), all rejections were short-token docs
- zero `no_para_numbering` rejections — paragraph numbering filter is not the bottleneck
- cleaning stage trigger rates stable and sensible:
  - chop front matter: 100%
  - final safety trim: 92%
  - header/footer artifact removal: mean 33 lines/doc
- pilot token stats (40 files):
  - median: 7,789 | mean: 11,073 | std: 11,369
  - min: 734 | max: 56,444
  - 3 high outliers (36k–56k tokens)

### scaled run (930 PDFs)

- scraped 930 tribunal PDFs and ran full pipeline
- survival rate: 271/930 (29.1%) — consistent with pilot
- all 678 rejections were `short_*tok` (0 paragraph-numbering drops)
- drop boundary clean: highest dropped file 720 tokens, lowest kept 690

#### post-reflow token distribution (271 files, uncapped)

| metric | value |
|---|---|
| median | 6,925 |
| mean | 11,460 |
| std | 12,743 |
| Q1 | 3,567 |
| Q3 | 15,546 |
| IQR | 11,980 |
| min | 690 |
| max | 102,020 |
| total | 3,105,775 |

- 17 high outliers detected (>33k tokens each)
- right-skewed distribution: mean >> median, consistent with pilot

#### top-document concentration

| group | tokens | % of total |
|---|---|---|
| top 5 | ~358k | ~11.5% |
| top 10 | ~540k | ~17.4% |
| all 17 outliers | ~783k | ~25.2% |
| bottom 50% (136 files) | ~600k | ~19% |

5 mega-decisions hold more tokens than the entire bottom half of the corpus.

#### pipeline health at scale

| stage | trigger rate |
|---|---|
| chop front matter | 98% |
| strip trailing admin | 24% |
| truncate admin tail | 8% |
| final safety trim | 70% |
| header/footer removal | mean 28 lines, max 252 |
| quality score | mean 56, median 53 |
| reflow compression | 0.32 (consistent with pilot 0.33) |

- `tokens_per_1k_chars` stable at ~210 across all files — extraction quality is uniform

### per-document token cap decision

- with 17 outliers holding 25% of total tokens from 6% of documents, uncapped training would allow a handful of mega-decisions to disproportionately dominate gradient updates
- decided to cap at **20,000 tokens per document**, truncating at the nearest preceding paragraph boundary (`\n\n`)
- simple front-truncation (keep start, cut tail) is appropriate because:
  - front matter has already been chopped — text starts at JUDGMENT/REASONS heading
  - legal reasoning builds forward; later paragraphs reference earlier ones, not vice versa
  - only 17/271 documents affected — not enough to introduce meaningful front-bias
  - for DAPT, the model learns domain vocabulary and syntax patterns, not argument structure
- rejected alternatives:
  - multi-span window concatenation: creates synthetic discontinuities within training samples
  - start/middle/end segment selection: introduces dangling cross-references and adds complexity for negligible benefit on 6% of documents
- implemented `--max-tokens` flag in `reflow_tribunal_text.py`
  - truncation applied after reflow to preserve paragraph structure
  - snaps to last `\n\n` before token limit (falls back to last `\n` if paragraph break would lose >50%)
  - logs `truncated`, `original_tokens`, `kept_tokens` per file in reflow JSONL

### pre-cap metrics preserved

- uncapped metrics saved to `logs/pre-cap/` for dissertation justification:
  - `pilot_token_stats.json`, `pilot_token_stats.csv`
  - `pdf_to_txt_metrics.jsonl`, `reflow_metrics.jsonl`
- post-cap metrics (to be generated) serve as the actual corpus specification

### post-cap distribution (271 files, max 20k tokens)

- re-ran `reflow_tribunal_text.py --max-tokens 20000` on all 271 files, then re-tokenised
- post-cap metrics saved to `logs/post-cap/`

| metric | pre-cap | post-cap | change |
|---|---|---|---|
| median | 6,925 | 6,925 | — |
| mean | 11,460 | 9,215 | −19.6% |
| std | 12,743 | 6,738 | −47.1% |
| max | 102,020 | 19,997 | −80.4% |
| total | 3,105,775 | 2,497,335 | −19.6% |
| outliers | 17 | 0 | eliminated |

- cap only affected 17 documents; median and quartiles unchanged
- distribution now well-behaved: mean ≈ median, no extreme concentration

### corpus selection

- 2.5M tokens would dilute Layer B weighting (~200k tokens) and reduce sensitivity for the LoRA vs QLoRA comparison
- created `preprocess/select_corpus.py`:
  - deterministic shuffle (seed=42) of all 271 post-cap Layer A files
  - greedy selection to fill 1.8M total token budget (5k tolerance)
  - backfill pass for skipped docs if under budget
- ran selection: **170 Layer A docs selected** from 271 candidates

#### final corpus composition

| layer | files | tokens | % of total |
|---|---|---|---|
| layer A (tribunal decisions) | 170 | 1,602,190 | 88.9% |
| layer B (ACAS/codes/doctrine/govuk) | 60 | 200,237 | 11.1% |
| **total** | **230** | **1,802,427** | — |

- overshoot vs 1.8M target: +2,427 tokens (+0.13%)
- manifest written to `data/domain_corpus/corpus_manifest.json`

### next steps

- begin training pipeline: tokenise `corpus_final/` into HuggingFace dataset format
- run first LoRA vs QLoRA comparison experiment
- write full methdology section

---

## 14/02/2026

### protocol freeze direction + calibration setup

- moved experimental control from epoch-driven runs to **fixed-budget step-driven runs**:
  - `max_steps` is now first-class in training config and wired into `TrainingArguments`
  - objective is strict comparability across LoRA/QLoRA under identical optimisation-step/token budgets
- confirmed model choice for dissertation runs as **base LLaMA** (not instruct) for DAPT/perplexity evaluation:
  - avoids instruction-tuning alignment confounds when measuring domain adaptation dynamics
- consolidated and reviewed assumptions in `logs/protocol_v1.md` as the formal pre-calibration protocol:
  - fixed context length (`seq_len=1024`) and fixed split seed (`42`)
  - fixed LoRA target modules (`q_proj/k_proj/v_proj/o_proj`), baseline adapter capacity (`r=16, alpha=32`)
  - fixed optimiser/scheduler family (AdamW + cosine, warmup ratio 0.05, clip 1.0)
  - fixed comparative design principle: hold optimisation/data budget constant across LoRA vs QLoRA
  - fixed evaluation intent: report overall val ppl plus stratified layer A / layer B val ppl

### corpus manifest + split reproducibility hardening

- updated `preprocess/select_corpus.py` to emit a fully specified manifest:
  - added `layer_b.selected` per-file `{filename, tokens}` entries (previous blocker for stratified splitting)
  - made `source_dir` values portable repo-relative paths (removed machine-specific absolute paths)
- rebuilt `data/domain_corpus/corpus_manifest.json` and verified internal consistency of file and token totals

- refactored `src/create_intrinsic_splits.py` to match protocol:
  - stratified train/val by **layer (A/B) + token-length quartile within layer**
  - outputs `data/splits/intrinsic_splits.json` with:
    - `splits.{train,val}` rows
    - `by_relpath` mapping (`layer_a/...`, `layer_b/...`) for loader-safe identifiers
    - counts + token totals by split (layer-wise and overall)
    - quartile definition + allocation debug metadata
- rationale captured: quartile stratification stabilises held-out perplexity estimates and reduces short-doc skew in validation

### data module refactor for domain DAPT

- refactored `src/data_module.py` from eur-lex-only path to domain-corpus path:
  - consumes `corpus_final/` + split JSON assignments
  - deterministic row ordering via `relpath` sort for reproducibility
  - robust split-label handling (`val` / `validation`)
  - UTF-8 decoding with replacement fallback for brittle source files
- replaced map-batch-dependent grouping with deterministic python-level token concat+chunk
  - avoids hidden batch-boundary token loss
  - chunks to fixed `seq_len=1024`
- returns:
  - `lm_train`
  - `lm_val_all`
  - `lm_val_layer_a`
  - `lm_val_layer_b`
- `val_subset` semantics corrected to subset at row level first, then derive layer-specific val sets

### trainer/eval wiring for stratified perplexity

- updated `src/train_lora.py` to support `domain_corpus_lm` end-to-end
- added stratified evaluation trainer path:
  - logs overall val loss/perplexity
  - logs layer A val loss/perplexity
  - logs layer B val loss/perplexity
  - perplexity guarded against non-finite losses
- removed duplicate eval logging in custom trainer path
- QLoRA path validated in code:
  - `hardware.use_4bit: true` → 4-bit NF4 + bf16 compute + double quant + `prepare_model_for_kbit_training`

### calibration configs + launch scripts prepared

- added calibration configs:
  - `configs/llama3_3b_lora_domain_calibration.yaml`
  - `configs/llama3_3b_qlora_domain_calibration.yaml`
- both target:
  - domain corpus split
  - seq_len 1024
  - LoRA r=16, alpha=32, dropout=0.05
  - fixed-step short calibration regime
- added executable launch scripts:
  - `scripts/run_3070ti_lora_calibration.sh`
  - `scripts/run_3070ti_qlora_calibration.sh`

### checkpoint/evaluation policy (intended)

- checkpoint rule for main runs remains:
  - keep best checkpoint by **overall val perplexity** + final checkpoint
  - downstream transfer evaluation will use best-by-overall-val-ppl checkpoint
- calibration purpose clarified:
  - validates plumbing/stability and feasible batch/step settings
  - does **not** serve as post-hoc tuning to maximise perplexity

### immediate next checks

- verify base-model HF access/auth on run machine before launch
- run baseline eval before first training step and capture in experiments log
- run 3B LoRA + QLoRA calibration on 3070 Ti
- after calibration, freeze protocol v2 run constants (max_steps, effective batch, eval cadence, lr fallback rule)

### known gaps (not yet implemented)

- checkpoint selection policy is defined in protocol v1, but trainer is not yet configured to enforce:
  - `metric_for_best_model=eval_perplexity`
  - `load_best_model_at_end=True`
- protocol metrics `peak_vram` and `step_time` are not yet logged in training loop outputs
- baseline pre-training evaluation (unadapted model on held-out validation) is not yet automated in run flow
