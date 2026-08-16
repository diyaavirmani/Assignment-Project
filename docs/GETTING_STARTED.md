# Getting Started with 3GPP RAG Assistant

This guide will help you get the 3GPP RAG Assistant up and running on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.9 or higher
- pip (Python package installer)
- Git
- A free Groq API key (get one at https://console.groq.com/keys)

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/3gpp-rag-assistant.git
cd 3gpp-rag-assistant
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Groq API key (cloud mode) or leave blank for Ollama local mode
# You can use any text editor, for example:
nano .env
```

Add your Groq API key:
```
GROQ_API_KEY=<your-groq-api-key-here>
```

### 5. Download 3GPP Specifications

Create a `data/raw` directory and download some 3GPP specs:

**Recommended starter specs:**
- [TS 38.300 - NR Overall Description](https://www.3gpp.org/ftp/Specs/archive/38_series/38.300/)
- [TS 38.401 - NG-RAN Architecture](https://www.3gpp.org/ftp/Specs/archive/38_series/38.401/)
- [TS 23.501 - 5G System Architecture](https://www.3gpp.org/ftp/Specs/archive/23_series/23.501/)

Place the downloaded PDFs in `data/raw/`

### 6. Process Documents

```bash
python scripts/build_index.py
```

This will:
- Parse the PDF files
- Split them into chunks
- Generate embeddings
- Store them in the vector database

### 7. Start the Backend API

```bash
uvicorn src.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### 8. Launch the Frontend (Optional)

In a new terminal window:

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run Streamlit app
streamlit run src/frontend/app.py
```

The web interface will open at `http://localhost:8501`

## Verifying Installation

Try asking a question:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the NG-RAN architecture?"}'
```

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError`**
- Solution: Make sure your virtual environment is activated

**Issue: `Groq API Error`** (cloud mode)
- Solution: Verify `GROQ_API_KEY` is set correctly in `.env`
- Get a free key at https://console.groq.com/keys
- To avoid API keys entirely, switch to Ollama local mode (set `LLM_PROVIDER=ollama` in `.env` and install Ollama from https://ollama.com)

**Issue: `No documents found`**
- Solution: Ensure PDFs are in `data/raw/` and run the document processor

**Issue: `Port already in use`**
- Solution: Change the port number or kill the process using the port

## Next Steps

- Read the [API Documentation](http://localhost:8000/docs)
- Check out example queries in the README
- Explore the code in `src/`
- Add more 3GPP specifications

## Need Help?

- Open an issue on GitHub
- Check existing issues for solutions
- Contact: ericcosta.public@gmail.com
