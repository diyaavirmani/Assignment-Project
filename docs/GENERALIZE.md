# Generalizing the RAG Pipeline to a New Corpus

**Status:** Architecture guide — 3GPP is the implemented corpus. This document describes how to add a second one.

The system is designed so that everything corpus-specific plugs in through one seam: `src/core/corpus_config.py`. The retrieval pipeline, chunking logic, embedding model, vector store, security controls, and API surface are all corpus-agnostic. You provide a `CorpusConfig`; the rest stays the same.

---

## What varies between corpora

| Concern | Where it lives | 3GPP example |
|---|---|---|
| Document catalog | `CorpusConfig.catalog` | 37 specs: `spec_number`, `series`, `domain`, `generation` |
| Text cleaning | `CorpusConfig.text_cleaner` | Strip `\n\d+ 3GPP TS …` running headers |
| Retrieval filter dimensions | `CorpusConfig.filter_dimensions` | `["domain", "generation"]` |
| Doc ID field | `CorpusConfig.doc_id_field` | `"spec_number"` |
| Download/URL builder | corpus-specific helper | `spec_catalog.get_ftp_url()` |

Everything else — chunk size, overlap, embedding model, vector store schema, LLM prompts, rate limiting, SSRF guards — stays unchanged.

---

## The seam in code

`src/core/corpus_config.py` defines:

```python
@dataclass
class CorpusConfig:
    name: str
    description: str
    catalog: List[CatalogEntry]          # list of dicts, at least doc_id + title
    text_cleaner: TextCleaner            # Callable[[str], str]
    filter_dimensions: List[str]         # metadata keys for pre-retrieval filters
    doc_id_field: str = "doc_id"
```

`build_3gpp_corpus_config()` is the factory for the current corpus. It wraps the existing `spec_catalog.CATALOG`, adds a normalized `doc_id` alias, and wires in `threegpp_text_cleaner`.

`DEFAULT_CORPUS` is the module-level instance returned by that factory.

### Where 3GPP plugs into the live pipeline

The indexing pipeline (`scripts/build_index.py`) consumes `DEFAULT_CORPUS` at two points:

**`scripts/build_index.py` line 34 — import:**
```python
from src.core.corpus_config import DEFAULT_CORPUS
```

**`scripts/build_index.py` — startup summary (in `main()`):**
```python
print(f"Corpus : {DEFAULT_CORPUS.summary()}")
print(f"Filters: {DEFAULT_CORPUS.filter_dimensions}")
```
This is where the filter dimensions (currently `["domain", "generation"]`) come from the config, not from hardcoded strings.

**`scripts/build_index.py` — `enrich_chunks()` function:**
```python
corpus_entry = DEFAULT_CORPUS.get_entry(inferred["spec_number"]) if inferred else None
```
This is the production call site where the corpus-specific catalog plugs in. `DEFAULT_CORPUS.get_entry()` routes through the `CorpusConfig` dataclass to retrieve the domain, generation, spec number, and title that get stored as chunk metadata in ChromaDB. Swap `DEFAULT_CORPUS` for a different `CorpusConfig` and `enrich_chunks` picks up that corpus's fields without any other change.

---

## How to add a second corpus: step-by-step

### 1. Build the catalog

Create a list of `CatalogEntry` dicts. Required keys:

- `doc_id` — unique stable identifier (e.g. `"NIST-800-53-rev5"`)
- `title` — human-readable name shown in the UI

Add any extra keys you need for filtering or display. These become chunk metadata stored alongside each vector.

```python
NIST_CATALOG = [
    {
        "doc_id": "NIST-800-53-rev5",
        "title": "Security and Privacy Controls for Information Systems",
        "category": "control-catalog",
        "family": "AC",
    },
    ...
]
```

### 2. Write a text cleaner

Inspect what your corpus's PDFs or DOCXs look like after extraction. Common artifacts:

- SEC filings: XBRL inline tags, Bates numbers
- Legal codes: section number headers (`§ 42.1(b)(3)` repeated as footers)
- Medical guidelines: watermarks, chapter page numbers

Write a cleaner function with the signature `(str) -> str`. Start from `generic_text_cleaner` and layer on your patterns:

```python
def nist_text_cleaner(text: str) -> str:
    import re
    # Remove page footers like "NIST SP 800-53 Rev. 5  PAGE 42"
    text = re.sub(r"\nNIST SP[^\n]+PAGE \d+\n", "\n", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

If there are no special artifacts, pass `generic_text_cleaner` (already imported from `corpus_config`).

### 3. Create the CorpusConfig

```python
from src.core.corpus_config import CorpusConfig

nist_corpus = CorpusConfig(
    name="nist-800",
    description="NIST SP 800-series cybersecurity publications.",
    catalog=NIST_CATALOG,
    text_cleaner=nist_text_cleaner,
    filter_dimensions=["category", "family"],
    doc_id_field="doc_id",
)
```

### 4. Use the config in the processing pipeline

When indexing, pass the config's cleaner and catalog to the document processor:

```python
from src.core.document_processor_impl import UnifiedDocumentProcessor

processor = UnifiedDocumentProcessor(chunk_size=1000, chunk_overlap=200)

for doc_path in your_corpus_files:
    raw_text, file_meta = processor.load_document(doc_path)
    clean_text = nist_corpus.text_cleaner(raw_text)        # corpus-specific cleaning

    # Enrich metadata from catalog before chunking
    entry = nist_corpus.get_entry(doc_id_from_filename(doc_path))
    if entry:
        file_meta.update(entry)

    chunks = processor.chunk_text(clean_text, file_meta)
    # ... embed and index chunks as usual
```

The chunk metadata now carries whatever keys your catalog entry has. The retriever's `_build_where_filter` method will accept any of those keys as filters — you just need to pass them as keyword arguments.

### 5. Update the UI filter dimensions (optional)

The Streamlit app's sidebar filter widgets currently hard-code "Domain" and "Generation" because those are the 3GPP filter dimensions. To surface your corpus's dimensions:

- The canonical place to read filter options is `CorpusConfig.filter_dimensions`.
- The inline `CATALOG` in `streamlit_app.py` is a **cloud mirror** of `spec_catalog.CATALOG` — it was inlined to avoid the full import chain on Streamlit Cloud where the `src/` package may not be on the path. For a new corpus deployed to Streamlit Cloud, you would add an analogous inline catalog block and pass your filter dimensions to the widget-building code.

This is the one place that requires touching the UI layer; everything else is pure Python configuration.

---

## What the 3GPP corpus looks like as the worked example

| CorpusConfig field | 3GPP value |
|---|---|
| `name` | `"3gpp"` |
| `description` | "3GPP Release 16/17 specifications covering 5G NR and LTE. 37 curated specs across RAN and Core domains." |
| `catalog` | 37 entries wrapping `spec_catalog.CATALOG`; `doc_id` aliased to `spec_number` |
| `text_cleaner` | `threegpp_text_cleaner` — strips running headers matching `\n\d+ 3GPP TS …` |
| `filter_dimensions` | `["domain", "generation"]` |
| `doc_id_field` | `"spec_number"` |

Source: `src/core/corpus_config.py`, `build_3gpp_corpus_config()`.

The 3GPP catalog data itself remains in `src/core/spec_catalog.py`. Adding a new corpus does not touch that file.

---

## What is not yet wired

Honest framing of the current state:

- **`UnifiedDocumentProcessor.clean_text` still has an inline 3GPP cleaner.** (`src/core/document_processor_impl.py`, `clean_text` method.) The operation order of that inline cleaner and `threegpp_text_cleaner` differ slightly, so a direct swap would change chunking output on some inputs. The natural next step is to align the two implementations and accept a `text_cleaner` parameter in `UnifiedDocumentProcessor`, defaulting to `corpus_config.DEFAULT_CORPUS.text_cleaner`. For now, the cleaner runs inside the processor; a second corpus would either subclass `UnifiedDocumentProcessor` or call `corpus.text_cleaner(raw_text)` before passing text to `processor.chunk_text`.
- **The retriever does not yet accept a `CorpusConfig`.** The `domain` / `generation` filter parameters in `DocumentRetriever.retrieve()` are currently 3GPP-specific positional args. Generalizing them to accept any dimension from `config.filter_dimensions` is the second step.
- **No second corpus has been indexed.** The architecture is designed to generalize; it has not been validated against a real second corpus. A test using a synthetic NIST-like catalog is in `tests/test_corpus_config.py`.

The claim is: designed to generalize; 3GPP is the implemented corpus. The indexing pipeline already routes catalog lookups through `DEFAULT_CORPUS`; the processor and retriever are the remaining two steps.

---

## Why this matters for regulated domains

Technical and regulated corpora share a pattern: dense, structured documents with stable versioning, specialist vocabulary, and query patterns that mix precise lookup ("what does TS 38.413 say about handover triggers?") with conceptual navigation ("how does the SMF relate to the PCF?"). The same RAG architecture applies to NIST controls, SEC filings, HIPAA guidance, or aviation regulations.

The `CorpusConfig` makes that claim visible in code, not just in prose.
