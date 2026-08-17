# TTB Label Compliance POC

A proof-of-concept tool that checks alcohol beverage label images against
their COLA application data, built around a routing architecture that
keeps the common case fast without sacrificing accuracy on messy images.

## Design principle: safe automation, not full automation

This tool is deliberately conservative in one specific way, worth stating
up front rather than leaving implicit: **it will never auto-approve a
label unless every check can be resolved with real confidence.** Wherever
a check can't be -- an ambiguous read, a formatting judgment the system
can't fully verify, an unusual layout -- the result is `flag_for_review`,
not `approve`. There is no code path that guesses in the direction of
approval.

In practice, this means the review team's queue will contain a mix of
genuine violations and labels that are probably fine but couldn't be
automatically confirmed -- never labels the system silently pushed
through despite being unsure. The tradeoff is deliberate: some compliant
labels get flagged unnecessarily (added reviewer workload, low stakes)
rather than risk a non-compliant label being silently approved
(unacceptable, regardless of how rarely it might occur). This isn't just
a stated intention -- several proposed improvements during development
were tested and rejected specifically because they measurably violated
this principle, even when they otherwise improved overall accuracy. See
"Known open items" for a concrete example.

## How it works

1. **Triage** — a fast, local quality check (blur, brightness, contrast,
   resolution) decides whether a submission is "clean" or "messy."
2. **Extraction** — both paths now follow the same OCR-then-parse
   pattern, differing only in which OCR engine reads the text: clean
   labels use Tesseract (fast path); messy labels use GLM-OCR, a small
   OCR-specialist model confirmed more reliable than Tesseract on
   degraded images (careful path, local/Ollama backend only -- falls
   back to a direct vision read on a hosted backend, since GLM-OCR has
   no hosted equivalent). Both paths also make a second, narrow vision
   call to read the government warning's header text verbatim -- OCR
   text alone can't answer case-sensitivity reliably (case gets
   normalized away in plain OCR text). Header bold-ness is judged
   separately and deterministically (see weight_detection.py below),
   not by any model. Extraction runs deterministically (fixed
   temperature and seed), so repeated runs on the same image produce
   identical results.
3. **Validation** — both paths feed into a shared, SQLite-backed rule set
   covering the 7 required TTB label fields, plus two dedicated checks on
   the government warning header: its wording/casing (compared as literal
   extracted text, case-sensitive, against "GOVERNMENT WARNING:" -- not a
   model self-report) and whether it's bold (the one piece that genuinely
   can't be reduced to text, so it stays a model judgment).
4. **Reporting** — results come back as a plain-language explanation per
   field, not a bare pass/fail, and are persisted to `results/results.db`
   by default so a run's output survives after the terminal closes (see
   "Results history" below).

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
python scripts/init_db.py
ollama pull qwen2.5:14b
ollama pull qwen2.5vl:7b
ollama pull glm-ocr
pytest
```

**Windows:** `pytesseract` requires the Tesseract binary separately —
`winget install --id UB-Mannheim.TesseractOCR`. If `tesseract --version`
isn't recognized after installing, the binary's folder likely didn't get
added to PATH automatically — add it manually via System Properties →
Environment Variables → Path, then fully restart your terminal (not just
a new tab) before retrying.

## Usage

Generate synthetic test labels (see `sample_labels/README.md`), then run:

```bash
python scripts/generate_test_labels.py
python main.py --labels sample_labels/ --applications sample_labels/applications.json
```

Options:
- `--labels` — a single image path, or a directory of images
- `--applications` — JSON file mapping filenames to expected fields + context
- `--output` — write results to a JSON file instead of printing to the terminal
- `--max-workers` — concurrent batch workers (default: 8); tune for your GPU/CPU

## Results history

Every run of `main.py` saves its results by default (pass `--no-persist`
to skip). View past results without re-running anything:

```bash
python scripts/view_results.py --limit 20
```

## AI models

Both extraction steps run through a swappable backend
(`app/llm_backend.py`, selected via the `LLM_BACKEND` env var — defaults
to `ollama`), so the same codebase supports this local dev setup today
and a hosted deployment later without touching extraction logic. The
default, local backend runs through [Ollama](https://ollama.com) — no
external API, no API key:

- `qwen2.5:14b` — parses OCR text into structured fields, for both the
  fast path (Tesseract-read text) and the careful path (GLM-OCR-read
  text)
- `glm-ocr` — a 0.9B-parameter OCR specialist; reads raw text from
  messy/degraded label images on the careful path (local/Ollama backend
  only). Confirmed reliable across all 5 tracked messy test cases for
  the 7 core fields, at a fraction of the size of the vision model
  previously used for this step (0.9B params vs. 6GB) -- a genuine
  efficiency win, though not a confirmed accuracy improvement, since the
  vision model's direct read was already largely reliable on these same
  fields; the header/bold mechanism this doesn't touch was always the
  actual weak point.
- `qwen2.5vl:7b` — used narrowly now: a small vision call on every
  submission (fast and careful alike) to read the government warning
  header's exact wording/casing, compared with a deterministic,
  case-sensitive text match. It is no longer used for full-image field
  extraction on the careful path. Header bold-ness is NOT a model
  judgment at all -- it's measured deterministically (see
  weight_detection.py). This was tested directly with GLM-OCR too (a
  markdown-based bold-annotation prompt) and found not to work: the
  model doesn't apply formatting annotations regardless of image
  quality, confirmed on a sharp, unambiguous control image.

Ollama must be running locally (`ollama serve`, or the desktop app)
before the app runs. Once the models are pulled, inference happens
entirely on-device — no outbound network calls at runtime.

## Running the Streamlit app

```bash
streamlit run app.py
```

By default this uses the local Ollama backend (see "AI models" and
"Setup" above) — no data leaves your machine. To use the hosted Gemini
backend instead, set `LLM_BACKEND=gemini` and `GEMINI_API_KEY` in `.env`
(see `.env.example`).

**Privacy note:** Google's Gemini free tier may use submitted content to
improve its products; a paid tier does not. If that's a concern —
testing with real or sensitive label data, for example — use the local
Ollama backend instead, which never sends anything outside your machine.
That's the default with no extra configuration: just complete the
"Setup" steps above (Ollama install + model pulls) and run the app —
no API key, no `.env` changes required.

## Deployment considerations for a TTB-internal version

This version runs as a public-facing app, but because inference is
already local by design (see above), adapting it for TTB's internal
network isn't really a firewall workaround anymore — the remaining
questions are about setup and hosting. Extraction now runs through a
swappable backend (`app/llm_backend.py`) rather than calling Ollama
directly, so switching between local and hosted inference for different
deployment targets is a config change, not a code change:

1. **Model weight distribution** — pulling model weights from Ollama's
   public registry needs internet access once, at setup time. Behind a
   locked-down firewall, that likely means either a one-time approved
   exception for that registry endpoint, or mirroring the model files
   through an internal artifact repository instead.
2. **Centralized vs. per-machine hosting** — running Ollama on every
   agent's workstation requires GPU-capable hardware on each machine.
   Alternatively, host Ollama on a single internal GPU server and have
   the app reach it only over the internal network.
3. **On-prem/Azure-native deployment** — containerize the full stack
   (app + Ollama) and run it inside the existing FedRAMP-certified Azure
   environment on approved GPU compute, keeping everything inside TTB's
   existing security boundary rather than depending on any external
   registry at runtime.

## Status

**Implemented and tested:**
- Triage (`triage.py`) — thresholds tuned against the synthetic
  `sample_labels/` set; covered by `tests/test_triage.py`, which pins
  expected routing per file so a threshold regression can't ship
  silently again. Caveat: these are synthetic renders, not real
  photographs — worth revisiting once real or more realistic label
  photos are available.
- Validation (`validation.py`) — the 7-field ruleset, plus two dedicated
  checks on the government warning header: an exact, case-sensitive text
  match on its wording/casing, and a separate boolean check on whether
  it's bold. Wording match on the warning *body* (not the header) is
  case/whitespace-normalized rather than requiring byte-exact model
  output. Covered by `tests/test_validation.py`.
- Extraction (`extraction.py`) — deterministic sampling. Both paths use
  the same OCR-then-parse pattern (Tesseract for clean images, GLM-OCR
  for messy ones — confirmed more reliable than Tesseract there), with
  field-parsing going through the swappable backend
  (`app/llm_backend.py`, `LLM_BACKEND` env var). GLM-OCR is local/Ollama
  only; the careful path falls back to a direct vision read on a hosted
  backend. The warning header's verbatim text is read via a dedicated
  vision call; its bold-ness is measured deterministically (see
  weight_detection.py below) — neither is reliably derivable from plain
  OCR text. Covered by `tests/test_extraction.py` and
  `tests/test_llm_backend.py`.
- Batch orchestration (`batch.py`) — concurrency is configurable via
  `--max-workers` rather than hardcoded, so a future hosted deployment
  can retune it for its actual hardware.
- Reporting (`reporting.py`) — plain-language per-field output,
  including route and quality signals.

**Known open items:**
- No frontend yet — deferred to the planned Streamlit app (see
  "Deployment considerations" above), which will serve as the actual
  end-user interface; `main.py` remains a CLI for development/testing.
- Public deployment (Streamlit + a free hosted LLM API, since the
  current local-Ollama design can't run on Streamlit Community Cloud's
  free tier) is planned but not started.
- Every submission now makes at least one vision-model call (for the
  header check), including fast-path ones that previously never touched
  it. This makes the earlier VRAM-swap concern for mixed-route batches
  unconditional rather than occasional -- worth re-timing a full batch
  and watching `ollama ps` before treating current latency numbers as
  representative.
- Government warning header bold-detection is deterministic (OCR +
  ink-density measurement, not a model judgment) and reliable on the
  fast path: every clean-image test case, bold and non-bold alike,
  resolves correctly, every run. On the careful path (blurred images),
  it correctly identifies a genuinely NON-bold header, but currently
  fails to confirm a genuinely bold one most of the time -- root cause
  is a measurement bias, not a localization failure: OCR reliably finds
  the header even under blur, but blurred real ink measures less dense
  than the pristine, unblurred calibration references, regardless of
  true font weight. 4 known cases are marked `xfail` in
  `tests/test_weight_detection.py`.

  This does not compromise the tool's core safety guarantee: across
  every variant tested during development -- the shipped method, three
  alternative bounding-box refinements, blur-matched calibration,
  same-image body-text comparison, and same-image brand-name comparison
  -- every single observed failure has been in the safe direction (a
  compliant label sent to review) with zero exceptions in the shipped
  code. The two variants that did produce a dangerous-direction result
  (a non-compliant label reading as compliant) were tested and rejected
  before ever being adopted. Revisit only with an approach verified,
  the same way, to never produce a false-bold result.