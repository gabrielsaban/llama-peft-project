# Data provenance

## Corpus summary

The frozen `protocol_v3` corpus contains **1,802,427 Llama-3 tokens across 230
documents**:

| Layer | Documents | Tokens | Content |
|---|---:|---:|---|
| A | 170 | 1,602,190 | UK employment tribunal judgments |
| B | 60 | 200,237 | Employment-law guidance and doctrine |

The fixed document-level split uses seed 42 and an 85/15 train/validation ratio,
stratified by corpus layer and token-length quartile. The authoritative records
are `data/domain_corpus/corpus_manifest.json`,
`data/splits/intrinsic_splits.json`, and `logs/protocol_v3.md`.

## External sources

- [GOV.UK employment tribunal decisions](https://www.gov.uk/employment-tribunal-decisions)
- [Acas advice](https://www.acas.org.uk/advice)
- [GOV.UK employing people guidance](https://www.gov.uk/browse/employing-people)
- [GOV.UK working guidance](https://www.gov.uk/browse/working)
- [House of Commons Library research](https://commonslibrary.parliament.uk/)

Public availability does not imply that the source material is owned by this
project or covered by its MIT licence. Each source retains its own terms.

## Project contribution and responsible use

The project contribution is the source selection, extraction, cleaning,
paragraph-aware token capping, deterministic selection, fixed split, training
protocol, and aggregate analysis. Employment-law documents can describe
sensitive workplace events even when public. The corpus was used for aggregate
intrinsic evaluation, not legal advice or assessment of individuals. Lower
perplexity does not establish fairness, factual accuracy, legal reliability, or
suitability for deployment.
