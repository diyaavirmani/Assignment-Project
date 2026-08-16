# Data Directory

This directory contains 3GPP specifications and processed data for the RAG system.

## Structure

```
data/
├── raw/              # Original 3GPP PDF specifications
├── processed/        # Parsed and chunked documents (JSON)
└── vectordb/         # ChromaDB vector database (generated)
```

## Downloading 3GPP Specifications

### Recommended Starter Specs (5G Core)

1. **TS 38.300** - NR Overall Description
   - URL: https://www.3gpp.org/ftp/Specs/archive/38_series/38.300/
   - Description: High-level 5G NR architecture overview

2. **TS 38.401** - NG-RAN Architecture Description
   - URL: https://www.3gpp.org/ftp/Specs/archive/38_series/38.401/
   - Description: Radio Access Network architecture

3. **TS 23.501** - 5G System Architecture
   - URL: https://www.3gpp.org/ftp/Specs/archive/23_series/23.501/
   - Description: 5G core network architecture

4. **TS 38.104** - Base Station Radio Transmission and Reception
   - URL: https://www.3gpp.org/ftp/Specs/archive/38_series/38.104/
   - Description: gNB radio specifications

### How to Download

**Option 1: Manual Download**
1. Visit the URLs above
2. Download the latest version PDF
3. Place in `data/raw/` directory

**Option 2: Automated Script (Coming Soon)**
```bash
python scripts/download_specs.py
```

### Expanding Your Dataset

Additional useful specs:
- **TS 38.211** - Physical channels and modulation
- **TS 38.214** - Physical layer procedures for data
- **TS 38.331** - Radio Resource Control (RRC)
- **TS 23.502** - 5G Procedures
- **TS 33.501** - Security architecture

## Processing Pipeline

After downloading PDFs:

```bash
# Process all PDFs in data/raw/
python src/core/document_processor.py

# This will:
# 1. Parse PDFs → data/processed/*.json
# 2. Generate embeddings
# 3. Store in data/vectordb/
```

## Storage Requirements

- **Raw PDFs**: ~5-10 MB per spec
- **Processed JSON**: ~2-3x PDF size
- **Vector DB**: ~100-200 MB for 10 specs

Total: ~500 MB for a good starter dataset

## Notes

- The `data/vectordb/` directory is auto-generated
- Don't commit large PDFs to Git (they're in .gitignore)
- Keep raw PDFs as backup for reprocessing
