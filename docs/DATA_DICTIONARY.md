# Data dictionary

One row represents a distinct work or a reviewed edition family consolidated for training.

| Column | Meaning |
|---|---|
| `document_id` | Canonical Open Library work ID selected for the example |
| `source_work_ids` | All source work IDs represented by this example |
| `title`, `subtitle` | Source titles, enriched from a confirmed English edition where available |
| `authors` | Source author names for identity checks; not appended to model input |
| `subject_headings` | Explicit source subjects, retained as an array |
| `genre_tags` | Identifiable genre tags copied from subjects; not independently inferred |
| `first_sentence` | Source-provided field when available; empty when absent |
| `first_sentence_sources` | Provenance links for the extracted first-sentence field |
| `ddc_raw` | Original DDC candidates, including notation and decimal detail; audit only |
| `label_level_1` | The target string: 000, 100, 200, 300, 400, 500, 600, 700, 800 or 900 |
| `label_name` | Human-readable name of the target; never use as an input feature |
| `raw_text` | Prepared title, subtitle, first sentence and subjects; no fabricated summary |
| `edition_ids`, `isbns` | Known edition identifiers and available ISBNs; incomplete coverage |
| `source_urls`, `retrieved_at` | Work provenance and latest contributing search-snapshot timestamp; individual API timestamps are in sources |
| `review_status` | Original sample-review status, or not_reviewed; distinct from duplicate review |
| `label_override` | Evidence and reason for a documented broad-class correction; empty otherwise |
| `previous_split_memberships` | Original split for each constituent work, or not_previously_split |
| `prior_holdout_status` | Whether this whole split group contains a previous test/validation book |
| `description_optional`, `description_source_url` | Optional description with provenance; not included in default raw_text |
| `text_enrichment_sources` | Edition URLs used to extend titles/subtitles |
| `text_availability` | title_only, or title_and_context when subtitle/subjects/first sentence exists |
| `deduplication_status` | retained_distinct_work or consolidated_edition_family |
| `split_group_id` | Conservative group identifier; keep a complete group within one evaluation split |

JSONL stores arrays and objects directly. CSV stores those fields as JSON strings. Empty text means not supplied, not a negative fact about the book.

The raw DDC is not overwritten when a broad class is corrected. A source value such as `128/.2` can provide L1 `100` without making any claim that the detailed notation was normalized or independently validated.
