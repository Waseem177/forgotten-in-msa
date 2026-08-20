# A False Positive in Cross-Lingual Unlearning Evaluation

Code, adapters and evaluation data for our audit of cross-lingual unlearning in
`Qwen2.5-0.5B-Instruct`.

Cross-lingual audits of LLM unlearning ask whether a fact forgotten in one
language is still recoverable in another. The usual design compares a model
before and after unlearning, sees a smaller loss increase in the second
language than in the first, and reads that residual as a fact surviving
translation. We show the inference is unsafe without a control that is
routinely left out.

Unlearning TOFU fictional-persona facts with GradDiff and auditing across
English, Modern Standard Arabic, Egyptian Arabic and Hindi reproduces exactly
that signature: forget-set loss rises by +1.818 nats in English but only +0.550
in MSA, 30% as much. Measuring the **base** model as well shows the signature to
be an artefact. English-only fine-tuning left no measurable trace of those facts
in the other three varieties — it raised forget-set loss there, and raised
retain-set loss by an indistinguishable amount. There is no cross-lingual memory
for unlearning to have left behind. Once the never-learned and
uniform-degradation components are removed, the MSA effect is +0.034 nats, 3.5%
of English rather than 30%.

The methodological claim: a base-model transfer control and a retain-set control
are jointly necessary for any cross-lingual unlearning result, and both come at
no additional training cost.

The LaTeX source is in `paper/`; `make -C paper` builds the PDF. Compiled PDFs
are not tracked.

## Layout

```
scripts/          pipeline entry points
src/data/         TOFU loading and the multi-variety probe sets
src/train/        QLoRA collator (answer-only loss masking)
src/audit/        teacher-forced recall probe
src/utils/        per-step training trace
data/personas/    machine-translated probes: msa, egy, hi
runs/             one directory per training run (weights gitignored)
results/          audit output, written by run_audit.py
paper/            LaTeX source
```

## Setup

```
pip install -r requirements.txt
```

The scale is deliberate: it fits on a single consumer GPU, so the whole
experiment reproduces without cloud compute. The base model is frozen in 4-bit
NF4 and only LoRA adapters (r=16, alpha=32) are trained. The anchor run is the
expensive step at 23.3 GPU-hours; unlearning and auditing are cheap beside it.

## Pipeline

**1. Anchor.** Fine-tune the base model on TOFU so it actually knows the facts
that will later be unlearned. This is the model every later comparison is made
against.

```
python scripts/train_anchor.py --run_id anchor_v1 --epochs 10
```

**2. Unlearn.** GradDiff ascends on the forget set while descending on a retain
set, `-forget_loss + retain_weight * retain_loss`. It passes through the useful
region and then diverges, so each epoch is written to its own run directory and
you pick from among them rather than taking the final state.

```
python scripts/run_ga_unlearn.py --anchor_run anchor_v1 --run_id ga_v4 --epochs 4
```

Watch it live in a second terminal — the trace is written per step, and the
watcher warns when retain loss climbs above where it opened, which is the
signature of a run destroying the adapter rather than forgetting facts:

```
python scripts/watch_train.py --run ga_v4
```

**3. Translate the probes.** NLLB-200-distilled-600M, into MSA, Egyptian Arabic
and Hindi. Already committed under `data/personas/`, so this only needs
rerunning if you change varieties.

```
python scripts/translate_tofu.py --split forget10 --varieties msa egy hi
```

**4. Audit.** Teacher-forced cross-entropy on answer tokens only, summed rather
than averaged so the score stays comparable across languages that tokenize at
different rates. The two flags are the point of the paper: `--include_base`
adds the transfer control, `--probe_retain` adds the retain control.

```
python scripts/run_audit.py --anchor_run anchor_v1 --unlearn_run ga_v4_ep2 \
    --varieties en msa egy hi --include_base --probe_retain

python scripts/analyze_audit.py --audit_file results/ga_v4_ep2_audit.jsonl
```

The audit writes one row per probe to `results/<unlearn_run>_audit.jsonl`;
`analyze_audit.py` turns that into the deltas, bootstrap intervals and figures
reported in the paper.

## Adapters

Weights are too large for git and ship as a release instead:

```
gh release download checkpoints-v4
tar xzf checkpoints.tar.gz
```

That archive contains `anchor_v1`, `ga_v4_ep2` (the checkpoint the paper
audits) and `ga_v4_ep3` (a stronger-forgetting alternative). The base model is
pulled from HuggingFace separately.

## Data

TOFU is from Maini et al. (2024), distributed under the MIT licence, which
permits redistribution of our translated derivative. The non-English probes are
machine translations and are treated as such throughout: the paper reports a
robustness check restricting MSA to the 337 answers containing no Latin-script
residue, which moves the result negligibly.

## Authors

Mohamed Waseem and Mohammed Fawaz, Sathyabama Institute of Science and
Technology, Chennai. Waseem wrote the paper and ran the anchor and unlearning
training; Fawaz owns the audit and analysis code and the runs it produces.
