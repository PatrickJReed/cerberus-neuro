# v0 Phase 1 — Baseline Disease Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train `BaselineDiseaseClassifier` on the v0 16k-crop subset to establish the disease-classification ceiling that gates whether the multi-task Cerberus run (Phase 2) is worth pursuing.

**Architecture:** Single-task training run on validated infrastructure. The only code change is making `notebooks/02_train.ipynb` § 4 (the baseline cell) self-contained — it currently inherits constants (`LR`, `WARMUP_STEPS`, `BATCH_SIZE`, `NUM_WORKERS`, `CROPS_PER_SITE`, `N_EPOCHS`, `steps_per_epoch`, `train_loader`, `val_loader`) from § 3 (Cerberus cell). For Phase 1 we want to skip § 3 entirely, so § 4 needs to define those itself.

**Tech Stack:** PyTorch + AMP, AdamW + cosine + warmup scheduler, `cerberus_neuro.model.BaselineDiseaseClassifier`, `cerberus_neuro.training.train`. All validated infrastructure; no new dependencies.

**Success criteria** (from spec, § Phase 1):

- val_acc ≥ 0.65 → proceed to Phase 2 (multi-task Cerberus)
- val_acc 0.55–0.65 → marginal; decide between scope-up or accept narrower gap
- val_acc ≤ 0.55 → scope-up mandatory before continuing v0

---

## File Structure

Files modified by this plan:

- `notebooks/02_train.ipynb`: cell 9 (the § 4 baseline training cell) gets all dependencies inlined so it can run without § 3 having executed first

No new files. No `src/cerberus_neuro/` package code changes. No version bump.

---

## Task 1: Make § 4 baseline cell self-contained

**Files:**
- Modify: `notebooks/02_train.ipynb` (cell index 9 — the baseline training cell)

**What changes:** The cell's content is replaced. All variables it currently inherits from cell 7 (`LR`, `WARMUP_STEPS`, `BATCH_SIZE`, `NUM_WORKERS`, `CROPS_PER_SITE`, `N_EPOCHS`, `steps_per_epoch`, `train_loader`, `val_loader`) are defined locally. The TF32 matmul setting is also called locally.

- [ ] **Step 1: Update cell 9 source via JSON edit**

```bash
python3 - <<'PY'
import json
path = '/Users/patrickreed/Sandbox/cerberus-neuro/notebooks/02_train.ipynb'
nb = json.load(open(path))

new_src = '''# Self-contained baseline-only training cell. Can run after § 1 setup and
# § 2 manifest construction without § 3 (Cerberus) having executed first.
import torch
from torch.utils.data import DataLoader

from cerberus_neuro.model import BaselineDiseaseClassifier
from cerberus_neuro.training import TrainConfig, train

# TF32 matmul on Ampere/Ada GPUs (L4, A100). Free ~5-10% speedup.
torch.set_float32_matmul_precision('high')

# Config matches the cell-type-only sanity-check recipe that achieved val acc
# 0.96 in 1 epoch — the same training infrastructure should converge for
# disease classification at the same data scale.
BATCH_SIZE = 64
CROPS_PER_SITE = 12
NUM_WORKERS = 8
LR = 3e-4
N_EPOCHS = 15

train_ds = NeuroPaintingDataset(
    train_manifest, CACHE_DIR,
    crop_size=256, crops_per_site=CROPS_PER_SITE,
    min_cells_per_crop=1, augment=True,
)
val_ds = NeuroPaintingDataset(
    val_manifest, CACHE_DIR,
    crop_size=256, crops_per_site=CROPS_PER_SITE,
    min_cells_per_crop=1, augment=False, shuffle=False,
)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
    pin_memory=torch.cuda.is_available(),
    prefetch_factor=4,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
    pin_memory=torch.cuda.is_available(),
    prefetch_factor=4,
)

n_train_sites = len(train_manifest)
steps_per_epoch = max(1, (n_train_sites * CROPS_PER_SITE) // BATCH_SIZE)
WARMUP_STEPS = max(1, int(0.05 * N_EPOCHS * steps_per_epoch))
print(f'steps_per_epoch={steps_per_epoch}  n_epochs={N_EPOCHS}  '
      f'warmup_steps={WARMUP_STEPS}  total_steps={N_EPOCHS * steps_per_epoch}')

baseline_cfg = TrainConfig(
    n_epochs=N_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    lr=LR,
    encoder_lr_ratio=0.1,         # pretrained encoder at 0.1*LR; head at LR
    weight_decay=1e-4,
    warmup_steps=WARMUP_STEPS,
    grad_clip_norm=1.0,
    amp=torch.cuda.is_available(),
    log_every_steps=20,
    ckpt_every_steps=steps_per_epoch,  # one mid-epoch latest.pt write
    seed=0,
)

baseline_dir = CKPT_BASE / 'baseline'
baseline_summary = train(
    model=BaselineDiseaseClassifier(in_channels=6, n_classes=2),
    train_loader=train_loader,
    val_loader=val_loader,
    cfg=baseline_cfg,
    checkpoint_dir=baseline_dir,
    hf_repo='patrickjreed/cerberus-neuro-v0-baseline',
    resume_from=baseline_dir / 'latest.pt' if (baseline_dir / 'latest.pt').exists() else None,
)
print('\\nbaseline summary:', baseline_summary)
'''

# Cell 9 is the baseline training code cell. Verify its identity before
# overwriting (must contain BaselineDiseaseClassifier and baseline_cfg).
src9 = ''.join(nb['cells'][9]['source']) if isinstance(nb['cells'][9]['source'], list) else nb['cells'][9]['source']
assert 'BaselineDiseaseClassifier' in src9 and 'baseline_cfg' in src9, \\
    f'cell 9 is not the baseline training cell (got first 200 chars: {src9[:200]!r})'

nb['cells'][9]['source'] = new_src
nb['cells'][9]['outputs'] = []
nb['cells'][9]['execution_count'] = None

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
print('cell 9 patched')
PY
```

Expected output:
```
cell 9 patched
```

- [ ] **Step 2: Verify the notebook is still valid JSON and the patched cell is parseable Python**

Run:
```bash
python3 -c "
import json, ast
nb = json.load(open('/Users/patrickreed/Sandbox/cerberus-neuro/notebooks/02_train.ipynb'))
src = ''.join(nb['cells'][9]['source']) if isinstance(nb['cells'][9]['source'], list) else nb['cells'][9]['source']
ast.parse(src)
print(f'OK: {len(nb[\"cells\"])} cells, cell 9 parses ({len(src)} chars)')
"
```

Expected output (cell count may differ slightly; the assertion that matters is cell 9 parses):
```
OK: 12 cells, cell 9 parses (XXXX chars)
```

- [ ] **Step 3: Verify cell 9 declares all the names it now needs locally (no inheritance from cell 7)**

Run:
```bash
python3 - <<'PY'
import json
nb = json.load(open('/Users/patrickreed/Sandbox/cerberus-neuro/notebooks/02_train.ipynb'))
src = ''.join(nb['cells'][9]['source']) if isinstance(nb['cells'][9]['source'], list) else nb['cells'][9]['source']

# These names are referenced by the cell. They must either be assigned in
# the cell itself or inherited from § 1 setup (CACHE_DIR, CKPT_BASE,
# train_manifest, val_manifest, NeuroPaintingDataset).
must_be_local = ['BATCH_SIZE', 'CROPS_PER_SITE', 'NUM_WORKERS', 'LR',
                 'N_EPOCHS', 'WARMUP_STEPS', 'steps_per_epoch',
                 'train_ds', 'val_ds', 'train_loader', 'val_loader',
                 'baseline_cfg', 'baseline_dir', 'baseline_summary']
must_be_inherited = ['CACHE_DIR', 'CKPT_BASE', 'train_manifest', 'val_manifest',
                     'NeuroPaintingDataset']

for name in must_be_local:
    if name + ' =' not in src and name + '=' not in src and f'\\n{name}' not in src.replace('\\n', '\\n'):
        # rough check: the name should appear on the LHS of an assignment
        # somewhere in the cell. The full `=` form covers `BATCH_SIZE = 64`.
        if f'{name} =' not in src:
            print(f'FAIL: {name} not assigned locally in cell 9')
            raise SystemExit(1)

for name in must_be_inherited:
    if name not in src:
        print(f'WARN: {name} not referenced (may be unused; OK if so)')

print('OK: all local names assigned, no missed inherited names')
PY
```

Expected output:
```
OK: all local names assigned, no missed inherited names
```

- [ ] **Step 4: Commit the change**

```bash
cd /Users/patrickreed/Sandbox/cerberus-neuro && git add notebooks/02_train.ipynb && git commit -m "$(cat <<'EOF'
make: 02_train.ipynb § 4 baseline cell self-contained

Per docs/superpowers/specs/2026-05-08-v0-baseline-first-paired-experiment-design.md.

The baseline cell was inheriting BATCH_SIZE / CROPS_PER_SITE / NUM_WORKERS /
LR / N_EPOCHS / WARMUP_STEPS / steps_per_epoch / train_loader / val_loader
from § 3 (Cerberus cell), so running § 4 alone required first running § 3.
Phase 1 of the v0 plan is baseline-only (Cerberus is gated on Phase 1
outcome), so § 4 needs to be runnable without § 3.

All inherited constants now defined locally in § 4 with values matching
the cell-type-only sanity-check recipe that achieved val acc 0.96 in
1 epoch (BATCH_SIZE=64, CROPS_PER_SITE=12, NUM_WORKERS=8, LR=3e-4,
N_EPOCHS=15). DataLoaders constructed locally as well.

§ 4 still inherits CACHE_DIR / CKPT_BASE (defined in § 1) and
train_manifest / val_manifest (defined in § 2). Those are the
intentional cross-cell dependencies.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" && git push origin main
```

Expected: commit lands and pushes successfully. If push fails due to a Colab autosave on `origin/main`, the recovery is `git pull --rebase origin main && git push origin main` (no merge conflict expected since this commit only touches cell 9 and Colab autosaves don't typically modify just that cell during a non-running session).

---

## Task 2: Run Phase 1 in Colab Pro

**Files:** none modified. This is a manual execution step performed in Colab.

- [ ] **Step 1: Open the notebook in Colab via the badge**

URL: `https://colab.research.google.com/github/PatrickJReed/cerberus-neuro/blob/main/notebooks/02_train.ipynb`

If a tab is already open, do **File → Revert** to pull the latest committed version (otherwise Colab's in-memory copy is stale and an autosave will overwrite the new cell).

- [ ] **Step 2: Confirm runtime is L4 GPU**

In Colab toolbar: Runtime → Change runtime type → Hardware accelerator: GPU (L4). Click Save.

If a runtime is already attached and is something else (T4, A100), restart with L4 to match the wall-clock budget assumed in the spec (~2 hours).

- [ ] **Step 3: Run § 1 setup**

Click into cell 2 (the install + sys.modules purge cell) and run. Wait for `install + cache purge done` print.

Click into cell 3 (the HF login + Drive mount + cache setup). Run. Wait for the GPU print line confirming L4 is attached.

Expected: cell 3 prints something like:
```
HF login: OK (Colab secret)
Mounted at /content/drive
image cache target: /content/cerberus-cache/cpg0038-tegtmeyer-neuropainting
  populated: True (or False on a fresh session)
CACHE_DIR (fast):  /content/cerberus-cache
CKPT_BASE (drive): /content/drive/MyDrive/cerberus-neuro/cache/v0
GPU: True, device: NVIDIA L4
```

- [ ] **Step 4: Run § 2 manifest + prefetch**

Run cell 5. The parallel S3 prefetch handles cache population:
- If `/content/cerberus-cache/` is empty (fresh session), this takes ~5–10 min.
- If `/content/cerberus-cache/` is already populated from a prior session, all keys skip and the prefetch finishes in seconds.

Expected final line: `prefetch done: N downloaded, M cached, 0 failed`. If `failed > 0`, investigate (intermittent S3 failures are usually fine on retry; sustained failures indicate a config problem).

- [ ] **Step 5: SKIP § 3 (Cerberus cell). Do not run cell 7.**

Per the spec, Phase 1 is baseline-only. Cerberus training is Phase 2 (separate spec).

- [ ] **Step 6: Run § 4 (baseline training, cell 9)**

Click into cell 9 and run. The cell will print `steps_per_epoch=...  n_epochs=15  warmup_steps=...` and start training.

Watch for at least the first epoch's val printout (~10–15 min in) to confirm it's converging:
- Step ~20–80 (warmup): loss should be in 0.4–0.7 range, training acc visible per batch.
- Epoch 0 val: `acc_line_condition` will print. If it's already > 0.6, that's a strong early sign. If it's at 0.5 (chance), wait — disease classification often takes 3–5 epochs to commit.

Total wall-clock: ~2 hours for all 15 epochs.

- [ ] **Step 7: Run § 5 (final summary, cell 11)**

After cell 9 completes (`baseline summary: {...}` printed), run cell 11. It reads `baseline/train_log.jsonl` and prints per-epoch val records.

Expected output format:
```
Cerberus (brightfield-only multi-task)
  (no records — § 3 was not run)

Baseline (6-channel single-task disease)
  epoch 0: acc_cond=0.5XXX
  epoch 1: acc_cond=0.5XXX
  ...
  epoch 14: acc_cond=0.XXXX

(no Cerberus records → no gap line)
```

The headline number is the **best val_acc across the 15 epochs**, not necessarily the final epoch (model may peak mid-training and slightly drift down).

---

## Task 3: Apply the decision gate from the spec

**Files:** none modified. This is interpretation + planning.

- [ ] **Step 1: Identify best-epoch baseline val_acc from the train_log.jsonl**

Either visually from the § 5 summary output, or programmatically:

```bash
# Run locally after pulling the autosaved notebook back from Colab via
# git pull (Colab autosave keeps the executed val records embedded).
python3 - <<'PY'
import json
nb = json.load(open('/Users/patrickreed/Sandbox/cerberus-neuro/notebooks/02_train.ipynb'))
# Find cell 11 (final summary). Its outputs include the per-epoch val printout.
c = nb['cells'][11]
for o in c.get('outputs', []):
    if 'text' in o:
        print(''.join(o['text']) if isinstance(o['text'], list) else o['text'])
PY
```

Or pull the JSONL directly from Drive:
```bash
# In Colab:
!cat /content/drive/MyDrive/cerberus-neuro/cache/v0/baseline/train_log.jsonl | python3 -c "
import json, sys
recs = [json.loads(l) for l in sys.stdin if json.loads(l).get('split') == 'val']
best = max(recs, key=lambda r: r['acc_line_condition'])
print(f'best val acc: {best[\"acc_line_condition\"]:.4f} at epoch {best[\"epoch\"]}')
print(f'final epoch acc: {recs[-1][\"acc_line_condition\"]:.4f} at epoch {recs[-1][\"epoch\"]}')
"
```

Record the best-epoch acc as `phase_1_baseline_val_acc`.

- [ ] **Step 2: Apply the decision-gate table**

| `phase_1_baseline_val_acc` | Action |
|---|---|
| ≥ 0.65 | Phase 2 unblocked. Brainstorm + spec the 2-head Cerberus + paired-experiment training plan. |
| 0.55–0.65 | Pause. Decide with the user: accept narrower gap and proceed to Phase 2 anyway, or write a Phase 1.5 spec for data scope-up first. |
| ≤ 0.55 | Phase 2 blocked at current scope. Write a Phase 1.5 spec for data scope-up (full cohort, or 63× resolution, or per-cell crops) before Phase 2 is meaningful. |

The action for the next phase depends entirely on which row applies. Do not pre-commit to Phase 2 work until this gate is evaluated.

- [ ] **Step 3: Document the result**

Append a short results note to the spec or write a separate `docs/superpowers/results/2026-05-08-v0-phase-1-baseline-result.md` file capturing:
- best-epoch val acc
- final-epoch val acc
- val acc trajectory across epochs (one line per epoch)
- which decision-gate row applies
- chosen next action

```bash
# Template for the results doc:
cat > docs/superpowers/results/2026-05-08-v0-phase-1-baseline-result.md <<'EOF'
# v0 Phase 1 Result — Baseline Disease Classifier

**Spec:** docs/superpowers/specs/2026-05-08-v0-baseline-first-paired-experiment-design.md
**Plan:** docs/superpowers/plans/2026-05-08-v0-phase-1-baseline-disease.md
**Run date:** YYYY-MM-DD
**Wall-clock:** ~N min

## Headline number

- best-epoch val_acc_line_condition: 0.XXXX (epoch N)
- final-epoch val_acc_line_condition: 0.XXXX (epoch 14)

## Per-epoch trajectory

| epoch | acc_line_condition |
|---|---|
| 0 | 0.XXXX |
| 1 | 0.XXXX |
| ... | ... |
| 14 | 0.XXXX |

## Decision gate applied

- Threshold matched: [≥ 0.65 / 0.55–0.65 / ≤ 0.55]
- Next action: [Phase 2 unblocked / scope-up needed / Phase 2 blocked]

## Notes

(Anything about training stability, surprising patterns, etc.)
EOF
```

Fill in the real numbers, then commit:
```bash
mkdir -p docs/superpowers/results
git add docs/superpowers/results/2026-05-08-v0-phase-1-baseline-result.md
git commit -m "docs: v0 Phase 1 result — baseline val_acc=0.XXXX (decision: ...)"
git push origin main
```

---

## Self-review

**Spec coverage check:**

- ✅ "Train BaselineDiseaseClassifier on the v0 16k-crop subset" → Task 2
- ✅ "Make § 4 self-contained" → Task 1
- ✅ "Use the validated cell-type-only training recipe" → Task 1 step 1 (BATCH_SIZE=64, LR=3e-4, encoder_lr_ratio=0.1, warmup, AMP, grad clip)
- ✅ "Apply the three-regime decision gate after Phase 1" → Task 3 step 2
- ✅ "Reuse populated /content/cerberus-cache" → Task 2 step 4 (cell 5 prefetch handles this)
- ✅ Out of scope items (Cerberus § 3, Phase 0.5, segmentation head changes, hyperparameter sweeps, v1 stretch) — none of these tasks touch them

**Placeholder scan:** No "TODO", "TBD", or unspecified content. Step 3 of Task 3 has a template with `0.XXXX` placeholders that the engineer fills with the actual run results — these are intentionally placeholders because the values aren't known until the run completes.

**Type/name consistency:** `BATCH_SIZE`, `CROPS_PER_SITE`, `NUM_WORKERS`, `LR`, `N_EPOCHS`, `WARMUP_STEPS`, `steps_per_epoch`, `baseline_cfg`, `baseline_dir`, `baseline_summary` are spelled identically across the patched cell, the verification script, and the commit message.

**Scope:** single notebook edit + one training run + one decision interpretation. Genuinely small. No subprojects.
