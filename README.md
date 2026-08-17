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
2. **Extraction** — clean labels go through OCR followed by local LLM
   parsing into structured fields (fast path); messy labels skip OCR and
   are read directly by a local vision-capable model (careful path). Both
   paths also make a second, narrow vision call to read the government
   warning's header text verbatim and judge whether it's bold -- OCR text
   alone can't answer either question reliably (case gets normalized
   away, and font weight isn't a character at all), so this runs on every
   submission, not just messy ones. Extraction runs deterministically
   (fixed temperature and seed), so repeated runs on the same image
   produce identical results.
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

Both extraction steps run locally through [Ollama](https://ollama.com) —
no external API, no API key:

- `qwen2.5:14b` — parses OCR text into structured fields (fast path)
- `qwen2.5vl:7b` — reads label images directly: full extraction on the
  careful path, plus a second, narrow header check on every fast-path
  submission too (the header's exact wording/casing and its bold-ness --
  every submission now uses this model at least once, not only messy
  ones). The header's wording and casing are transcribed as literal text
  and compared with a deterministic, case-sensitive exact match -- only
  bold-ness remains a model judgment, since font weight can't be
  represented as extracted text.

Ollama must be running locally (`ollama serve`, or the desktop app)
before the app runs. Once the models are pulled, inference happens
entirely on-device — no outbound network calls at runtime.

## Deployment considerations for a TTB-internal version

This version runs as a public-facing app, but because inference is
already local by design (see above), adapting it for TTB's internal
network isn't really a firewall workaround anymore — the remaining
questions are about setup and hosting:

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
- Extraction (`extraction.py`) — deterministic sampling. The warning
  body and its header are extracted separately: the body is parsed
  normally by whichever path handles the submission, while the header's
  verbatim text/casing and bold-ness are always answered by a dedicated
  vision call, since neither is reliably derivable from OCR text.
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
  ink-density measurement, not a model judgment), correct on 5 of 7
  tracked test cases, and -- more importantly than the raw count --
  has never been observed to fail in the dangerous direction (a
  non-compliant label read as compliant). The 2 known failures
  (tests/test_weight_detection.py, marked xfail) both fail safe: a
  compliant label gets sent to human review rather than silently
  approved. Several fixes were tried and deliberately rejected because
  they traded a safe-direction fix for a dangerous-direction regression
  on a different case: three bounding-box-tightening strategies (naive
  min/max, connected-components), and JPEG-compression calibration
  (tested, ruled out -- no effect on any verdict). Revisit only with an
  approach verified not to introduce any false-bold result, not just
  one with a better pass rate.