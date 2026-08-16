# Demo Q&A Prep — 3GPP RAG Assistant

Grouped by likely source. Answers are concise — expand in the room as needed.

---

## PRODUCT / USE CASE

**Q: What problem does this actually solve day-to-day?**
> A: Telecom engineers constantly cross-reference 3GPP specs — during architecture reviews, RFP responses, code reviews, compliance checks. Today that means Ctrl+F through PDFs or Google searches that return forum posts instead of the actual standard. This gives you a direct, cited answer from the spec itself.

**Q: Who is the target user?**
> A: Primarily telecom engineers, solutions architects, and technical product managers working on 5G NR. Secondarily, anyone who needs to navigate dense standards documentation — regulatory, legal, internal wikis.

**Q: How is this different from just asking ChatGPT?**
> A: Three key differences: (1) ChatGPT answers from training data that may be outdated or wrong — this answers from the actual spec text. (2) Every answer is cited — you can verify the source. (3) It runs fully locally, so sensitive or proprietary content never leaves your machine.

**Q: What specs does it cover today?**
> A: v1.0.0 covers TS 38.300 (NR overall description) and TS 38.401 (NG-RAN architecture). The roadmap includes additional RAN specs, core network specs, and the ability to add your own documents.

---

## TECHNICAL

**Q: How accurate is it?**
> A: In our evaluation suite — 10 representative 5G questions — we hit 100% context precision (retrieved chunks are always from the right source) and 80% context recall (answers cover the key concepts). 60% of answers pass our full quality threshold. Early days, and there's clear room to improve with more spec coverage and prompt tuning.

**Q: What happens when it doesn't know the answer?**
> A: It says so. The retrieval step returns a confidence signal, and if the relevant context isn't in the indexed specs, the system surfaces that rather than fabricating an answer. In telecom, a confident wrong answer is worse than an honest "I don't have that."

**Q: How long does it take to set up?**
> A: `docker compose up` and you're running. First-time setup pulls the Ollama model (~2GB), which takes a few minutes. After that, cold start is under 10 seconds.

**Q: Can it handle proprietary or internal documents?**
> A: Yes — that's a first-class use case. You drop your PDFs into the data folder, run the indexing script, and it works the same way. Everything stays local.

**Q: What hardware does it need?**
> A: Runs on a modern laptop. 8GB RAM minimum, 16GB recommended for comfortable performance. No GPU required — inference runs on CPU via Ollama.

---

## BUSINESS / STRATEGIC

**Q: Is this open source? What's the license?**
> A: Yes, fully open source on GitHub. Anyone can use it, fork it, or contribute. Link: `github.com/ericnc09/3gpp-rag-assistant`

**Q: What's the cost to run?**
> A: Zero ongoing cost. No API fees, no cloud dependency. You pay for the hardware it runs on — which you already own.

**Q: What's on the roadmap?**
> A: Three near-term priorities: (1) broader spec coverage — more 3GPP documents indexed, (2) multi-document reasoning — answers that synthesize across multiple specs, (3) a hosted team version for organizations that want this as a shared internal tool without self-hosting.

**Q: How would a company deploy this internally?**
> A: Two paths: self-hosted on a team server using the Docker Compose setup — takes about an hour. Or wait for the hosted version where we manage the infrastructure. Either way, the data stays within your control.

---

## CURVEBALL / TOUGH QUESTIONS

**Q: Couldn't someone just build this with a RAG template in an afternoon?**
> A: The basic pipeline, yes. The hard parts are the domain-specific chunking strategy for spec documents (tables, section hierarchies, cross-references), the evaluation framework to actually measure quality, and making it production-ready — API, streaming, Docker, test coverage. That's what v1.0.0 represents.

**Q: Why not just use an existing enterprise search tool?**
> A: Enterprise search finds documents. This answers questions. The difference is whether you get "here are 12 PDFs that might be relevant" vs. "here is the answer to your specific question, from section 4.3.2 of TS 38.300." For spec work, the latter saves the actual time.

**Q: What's the failure mode — when does it go wrong?**
> A: Two main cases: (1) the answer is in a spec we haven't indexed yet — it'll tell you it doesn't have enough context. (2) Very nuanced questions that require synthesizing across multiple distant sections — recall drops. Both are known gaps on the roadmap.

---

## CLOSING MOVE

If someone asks a question you can't answer cleanly:

> *"Great question — let me actually show you rather than tell you."*

Then type their question directly into the demo. Let the system answer it live.
