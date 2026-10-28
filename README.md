# d1 list extraction

Fast, experimental list extraction around the frozen **LiquidAI d1-2B** model.
The original model selects one text span per question. This project captures its
start/end scores once, then extracts additional non-overlapping mentions using
boundary checks, an empirical stopping cutoff, and comparisons between entity
questions. No token-by-token generation or additional transformer pass is needed
for the later selections.

**The weights are unchanged.** This is an inference-time model adapter and decoder,
not a fine-tuned checkpoint. It currently targets **named people, software
products/platforms, organizations, and their union**. Complete events and arbitrary
list schemas have not been validated.

## Quick start

Use Python 3.12 in a clean environment. Install CPU PyTorch first to avoid downloading
CUDA libraries that this CPU reference runner does not use:

```bash
git clone https://github.com/JMoroniBello/d1-list-extraction.git
cd d1-list-extraction
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.14.1 torchvision==0.29.1
python -m pip install -e '.[inference]'

d1-list download
d1-list extract --input examples/document.txt --kinds products
```

The pinned q8 decoder weights are about 2.13 GB, plus tokenizer and host files.
The reference runner uses CPU ONNX Runtime with an observed process footprint
around 3-4 GB in the development environment. Leave additional RAM for the OS and
other applications. GPU inference is not implemented or benchmarked by this wrapper.

`download` fetches the exact official checkpoint revision; it does not fetch changing
`main` weights. The checkpoint includes Python host code. Its pinned file hashes
are checked before loading. To use a pre-existing copy, pass `--model-dir /path/to/model`
or set `D1_LIST_MODEL_DIR`. `d1-list verify-model --full-weights` also hashes the large
weight file against its official LFS SHA-256.

## Python API

```python
from d1_list_extraction import ListExtractor, decode

with ListExtractor(threads=8) as model:
    scores = model.capture(
        "Dana Ruiz at Cedar Labs uses Slack and Notion. "
        "Dana Ruiz also manages Slack."
    )

# The model can be unloaded before list decoding.
result = decode(scores, kinds=["products"], policy="atomic")
for item in result["answers"]["products"]["items"]:
    print(item["text"], item["span"])

scores.save("/tmp/document-scores.npz")
```

The snapshot contains the source text, question definitions, original native
answers, character offsets, and raw pointer scores. Treat it as private source
data. Loading a snapshot uses no pickle and requires only the core NumPy package.

```bash
# Run the model once and save its reusable evidence.
d1-list capture --input examples/document.txt --scores /tmp/document-scores.npz

# Run different decoders without loading weights or calling a model.
d1-list decode --scores /tmp/document-scores.npz --policy atomic --trace
d1-list decode --scores /tmp/document-scores.npz --policy soft
```

Repeated names at different source offsets remain separate occurrences. Returned
spans use Python character offsets `[start, end)`, not byte offsets. Results are in
score order; sort by `span[0]` if source order is required. `joint_score` is an
uncalibrated start-probability times end-probability, not confidence that the entity
is correct. The default maximum is ten items per question; `at_capacity` means
the cap was reached, not that every mention has been found.

## What the model actually processes

Each capture asks all four reference questions in the same native packed group:
people, products, organizations, and combined. Even `--kinds products` still
computes all four questions; it only filters the returned answer kinds. This
preserves the tested context and provides rival type scores. The wrapper verifies
exactly one decoder forward for each capture and does not load the unused vision
encoder.

The default input limit is 8,192 tokens including questions. It can be changed with
`--max-input-tokens`; requests that exceed it are rejected rather than truncated.
The underlying checkpoint has a larger context limit, but large-context memory
and quality have not been validated here.

The original d1 question API is also exposed for decision tasks:

```python
with ListExtractor() as model:
    answer = model.decide(
        "Please refund the duplicate charge.",
        {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    )
```

`decide` retains upstream question semantics and does not apply list filters.
Questions must fit one packed group; text is the supported input modality in this
wrapper. This method is separate from the evaluated entity-list profile.

## Decoder profiles

| Profile | Behavior |
|---|---|
| `baseline` | Repeated non-overlapping span selection, without boundary or type filters |
| `atomic` | Tested default: whole-word endpoints, punctuation/connector rejection, cross-question type comparison |
| `soft` | Whole-word endpoints, modest length penalty, cross-question type comparison; no hard punctuation/connector rejection |

All use the same original score cutoff of `0.003` and preserve the initial native
presence gate. Filters and reranking happen **before** overlap suppression. That
allows the components of a rejected merged span to compete separately. An
organization candidate is rejected if another cached typed question supports the
same candidate more strongly. Those question distributions are compatibility
signals, not mutually exclusive calibrated type probabilities.

Atomic rules can reject legitimate names containing commas or words such as
"and". Soft penalties can disadvantage genuine long names. A rejected native
presence decision still yields an empty list. Neither profile guarantees
exhaustive extraction; validate thresholds and policies on your own data.

## Experimental evidence

The development experiment used 24 real customer-feedback records: eight for
decoder selection and 16 for evaluation. Labels were source-grounded but
agent-authored, not independently human-verified. The evaluation cases had already
been inspected during earlier experiments, so these are exploratory follow-up
results rather than new blind validation.

| Same evaluation scores | Correct targets / 28 | False positives |
|---|---:|---:|
| Baseline repeated selection | 24 | 2 |
| Boundary + type filtering | 26 | 0 |

Every previously correct target was retained. The combined question overlaps the
specific-type questions: these 28 question-level targets represent 14 physical
source mentions, not 28 independent mentions. Pooling outputs covered all 14
source mentions after refinement. The development set still had one false
positive. Only four source records named software, so product-list evidence is
particularly limited.

An isolated three-repeat CPU measurement found a median **16.5 ms per record for
the extra selected list decoding across all four questions**, after scores were
already in memory. The recorded original model capture averaged 7.65 seconds per
record on that machine. This was not a standalone one-question versus four-question
latency comparison. Timings vary with hardware and workload.

Private warehouse text, original captures, credentials, and local paths are not
distributed. Aggregate measurements are under `benchmarks/`; the public tests use
synthetic score fixtures. The portable decoder was separately checked against
576 saved question/profile outputs from the original and clarified experiments.

## Tests and project layout

```bash
# Core tests need no model download, Torch, GPU, or private data.
python -m pip install -e .
python -m unittest discover -s tests -v
```

* `decoder.py`: model-free list decoding and filtering.
* `snapshot.py`: immutable score arrays and portable snapshot format.
* `backend.py`: text-only one-pass capture and native decision API.
* `questions.py`: exact reference question profile.
* `model.lock.json`: pinned checkpoint, runtime revision, and file hashes.
* `_vendor/liquidonnx/`: unmodified upstream runtime source.

The large weights are downloaded from Hugging Face, not stored in GitHub. The
model and bundled runtime remain subject to the **LFM Open License v1.0** in
`LICENSE`; see `NOTICE` and `THIRD_PARTY_NOTICES.md`. This is an independent project,
not an official Liquid AI release.
