# Security Policy

## Supported versions

This project is a personal side project with a single maintained branch (`main`). Only the latest version on `main` receives security fixes.

| Version / branch | Supported |
|---|---|
| `main` (latest) | Yes |
| Any earlier tag or fork | No |

## Reporting a vulnerability

**Do not open a public GitHub issue for security reports.** Public disclosure before a patch is ready can put users of the deployed app at risk.

Send a private email to **ericcosta.public@gmail.com** with the subject line `[3gpp-rag-assistant] Security Report`.

Include:
- A description of the vulnerability and the affected component
- Steps to reproduce (or a proof-of-concept)
- Your assessment of impact and exploitability
- Your name/handle for attribution (optional)

**What to expect:**
- Acknowledgement within 72 hours
- An assessment (confirmed, not confirmed, or out of scope) within 5 business days
- A patch or mitigation plan communicated to you before any public disclosure
- Credit in the release notes if desired

## Scope

**In scope:**
- The FastAPI backend (`src/api/`)
- The Streamlit Cloud app (`streamlit_app.py`)
- Secrets / credentials handling
- Dependency vulnerabilities with a realistic exploit path against this app

**Out of scope:**
- Vulnerabilities in upstream LLM providers (Groq, Ollama) — report those to the respective vendors
- Social engineering or phishing attacks
- Denial-of-service attacks purely at the infrastructure/CDN layer (Streamlit Cloud)
- Self-XSS or issues requiring the attacker to have existing admin access to the deployment

## Security architecture overview

The threat model, controls, and residual risks are documented in
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

Key controls in brief:
- Per-IP rate limiting with LRU-bounded store (20 req/min; max 10,000 tracked IPs)
- Session-level query cap (200 queries/session) for cost and DoS protection
- Pydantic input validation with explicit min/max bounds and Literal enums on all API fields
- SSRF prevention: Ollama URL restricted to localhost equivalents; download URLs restricted to a three-domain GitHub allowlist
- Tar path-traversal guard on vector-index extraction
- Security response headers (X-Content-Type-Options, X-Frame-Options, CSP, Referrer-Policy, Permissions-Policy)
- HTML-escaping of all retrieval output before rendering in the UI
- Secrets passed only via environment variables or Streamlit Secrets — never committed to the repo
