# Demo GIF Capture Instructions

**Target file:** `docs/assets/demo.gif`  
**Purpose:** Embedded in README.md as the first visual a visitor sees.  
**Recording:** Requires the live app and a screen-capture tool — the instructions below are the script.

---

## What to show (30–45 seconds, optimized for a 6-second recruiter scan on GitHub)

The GIF should convey three things in order:

1. **Real product, not a toy.** The Streamlit app loads cleanly with the domain/generation filter panel visible.
2. **It works.** A realistic engineering question gets a cited answer with source spec numbers.
3. **The answer is grounded.** The source expander opens to show an actual chunk from a 3GPP spec with the spec number visible.

### Suggested query sequence

**Query 1 (architecture — high visual impact):**
> "Describe the F1 interface between gNB-CU and gNB-DU"

Expected: answer references TS 38.401 and TS 38.470/38.473. Source cards show spec numbers and generation badges. F1 is a multi-spec architecture question — a good showcase for retrieval that spans more than one specification.

**Query 2 (optional — shows cross-domain breadth if time allows):**
> "What is the role of the AMF in the 5G Core?"

Expected: domain filter switches to CORE; answer references TS 23.501. Shows the domain filtering UI in action.

---

## Step-by-step capture guide

### Prerequisites
- The live app is running at [https://3gpp-rag-assistant.streamlit.app/](https://3gpp-rag-assistant.streamlit.app/) — use the public URL, not localhost, so the GIF reflects real deployment
- Browser: Chrome or Firefox, zoom at 90% so the full app is visible without scrolling
- Window: 1280×800 or 1440×900 recommended

### Before recording
- [ ] Open the app and wait for the vector DB to load (spinner disappears)
- [ ] Clear any existing chat history if visible
- [ ] Make sure the sidebar shows "All" for both Generation and Domain filters

### Recording steps

1. **Show the app loaded and idle** (2 seconds). Sidebar visible. Chat input empty.
2. **Type Query 1** slowly enough that the text is readable: `Describe the F1 interface between gNB-CU and gNB-DU`
3. **Press Enter** and let the answer stream in (do not cut this off — the streaming effect is a feature worth showing).
4. **Scroll down slightly** if needed to show the source expander(s) below the answer.
5. **Click one source expander** to reveal the retrieved chunk text and spec number (e.g., "38401-h11.docx · TS 38.401 · 5G RAN").
6. **Pause 1 second** on the expanded source.
7. End recording.

Total: 30–45 seconds is ideal. Over 60 seconds will loop awkwardly on GitHub.

---

## Capture tools

**macOS:**
- [Kap](https://getkap.co/) — free, exports GIF directly. Set FPS to 10 (lower = smaller file size).
- [GIPHY Capture](https://giphy.com/apps/giphycapture) — free, simple.
- QuickTime screen recording → convert with `ffmpeg`:
  ```bash
  ffmpeg -i demo.mov -vf "fps=10,scale=900:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 demo.gif
  ```

**Linux:**
- [Peek](https://github.com/phw/peek) — drag to select region, exports GIF.

**Windows:**
- [ScreenToGif](https://www.screentogif.com/) — free, exports GIF directly.

**File size target:** Under 5 MB. GitHub renders GIFs inline up to ~10 MB, but large GIFs slow page loads. Use 10 FPS and 900px width.

---

## After recording

1. Save as `docs/assets/demo.gif`
2. Verify it loops cleanly (no visible jump at the loop point)
3. Embed it in README.md: `![Demo](docs/assets/demo.gif)`

---

## Placeholder note

`docs/assets/demo.gif` does not exist yet — the recording requires the live app. The README references this path with a fallback note until the recording is done:

```markdown
<!-- demo.gif placeholder — see docs/assets/DEMO_CAPTURE.md for recording instructions -->
```

Once the recording is committed at `docs/assets/demo.gif`, replace the fallback note in README with the image tag and the embed will appear automatically.
