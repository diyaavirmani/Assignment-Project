# Demo Slide: 3GPP RAG Assistant

---

## Slide Content

---

### HEADLINE
**Stop searching. Start asking.**
*AI-powered Q&A for 3GPP technical specifications*

---

### THE PROBLEM
> Engineers spend hours hunting answers buried in 500-page standards documents.

---

### THE SOLUTION
A local AI assistant that answers your 5G standards questions in plain English — with citations, in seconds.

---

### HOW IT WORKS  *(3-step visual)*

```
[ Ask a question ]  →  [ Searches 44K+ spec chunks ]  →  [ Cited answer in <3s ]
```

---

### WHAT YOU'LL SEE TODAY

1. Live query against TS 38.300 & TS 38.401
2. Streaming answers with source citations
3. Edge case handling — no hallucinations

---

### KEY FACTS  *(bottom bar / callout boxes)*

| 44K+ chunks indexed | 100% context precision | 50ms retrieval | 100% local |
|---|---|---|---|
| TS 38.300 & TS 38.401 | Eval benchmark | p50 latency | Zero data exposure |

---

### STACK  *(small, lower corner)*
Ollama · ChromaDB · sentence-transformers · FastAPI · Streamlit · Docker

---

### ROADMAP TEASER
More specs · Multi-document queries · Hosted team version

---

## Design Notes

- **Background:** Dark navy or black — telecom/tech feel
- **Accent color:** Electric blue or signal green
- **Font:** Clean sans-serif (Inter, DM Sans)
- **Keep it sparse** — this is a backdrop to your live demo, not the main event
- The 3-step flow and the 4 key facts boxes are the visual anchors
- Drop the GitHub URL in the bottom corner: `github.com/ericnc09/3gpp-rag-assistant`
