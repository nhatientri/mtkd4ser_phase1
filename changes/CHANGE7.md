# Change Summary (CHANGE 7): Fixed time budget, Colab from VS Code, two bug fixes

Results don't change unless you pass the new flag. Existing checkpoints, teacher caches, results JSON files, run names and W&B artifact names all keep working. **No retraining needed.** One behaviour change: KD/MTKD now refuses a teacher trained on a different fold from the run (section 3), because such a teacher has seen the run's test speakers.

## 1. `--TIME_BUDGET_MIN` (fixed-compute training)

This flag prepares the repo for an autoresearch loop, where an agent edits the training recipe and keeps a change only if dev improves. For those comparisons to be fair, every run has to get the same compute. With `N_EPOCHS` alone that doesn't hold: a larger batch, a smaller student or a different backbone changes how much training a run gets.

When `TIME_BUDGET_MIN > 0` (`mtkd4ser/pipeline.py`, `Trainer`):

- **Schedule by time.** The LR schedule is a `LambdaLR` driven by *elapsed training seconds / budget*: linear warmup for the first `WARMUP_RATIO` of the budget, then linear decay to 0. Every run gets a complete schedule, whatever its speed.
- **What the budget counts.** Only training-epoch time counts (`Trainer.train_sec`). Dev evaluation, teacher-logit caching, model loading and test don't.
- **Stopping.** Training stops at the first optimizer step after the budget is used up (never mid-accumulation). The partial epoch is then evaluated on dev as usual, best-checkpoint selection works as before, and `test()` runs on the best checkpoint.
- **Other limits still apply.** `N_EPOCHS` stays an upper bound, so pass a large value (e.g. 100) to let the clock decide. `PATIENCE` early stopping still applies.
- **Resume.** Resume is not budget-aware: the clock restarts from zero, and a warning is printed.

With the default `TIME_BUDGET_MIN=0` the step-based `get_linear_schedule_with_warmup` is used exactly as before. The flag is not part of `run_name()`, so pass `--RUN_NAME` when budgeted runs must not overwrite a normal run with the same settings.

`tests/test_smoke.py` has a budgeted run that checks it stops on the clock before `N_EPOCHS`.

## 2. Colab from VS Code

**Colab VS Code extension.** The extension (`googlecolab/colab-vscode`) runs the notebook on a Colab T4 from the IDE, but `google.colab.userdata.get()` (Colab Secrets) isn't supported there; see colab-vscode issue #215. The notebook now:

- reads `HF_TOKEN` / `WANDB_API_KEY` through `get_secret()`: Colab Secrets, then the environment, then a hidden `getpass` prompt. Nothing is stored in the notebook.
- puts `WANDB_API_KEY` in `os.environ`, so `!python main.py` subprocesses inherit it.
- clones `--branch REPO_BRANCH` (default `refactor/change6`), because `main` doesn't have the CHANGE5/CHANGE6 code yet.
- names the private dataset `ngunghoc/mtkd4ser-phase1` consistently.

**Install pin.** The notebook and `requirements.txt`/`environment.yml` now pin `datasets>=2.16,<4`. `datasets` 4.x decodes `Audio` columns with `torchcodec`, which must be installed and matched to the installed torch, or `prepare_features` fails before training. Notebook cell 15 also decodes one clip, so a broken audio backend shows up in seconds.

## 3. Bug fixes

- **W&B refs with a project or entity prefix.** `checkpoints._candidates` treated any ref containing `/` as a file path, so `project/name:alias` and `entity/project/name:alias` were mangled into names like `name:alias-best:best` and never resolved. Refs whose basename contains `:` (and that aren't `.pth`) are now passed through unchanged. Bare `name:alias` refs and `.pth` paths behave as before.
- **Teacher/student fold guard.** KD/MTKD now refuses a teacher trained on a different fold from the run: for example, the S1 EN teacher in a `--SESSION 2` run, whose training data contains session 2's test speakers.
  - Only the folds of the data the run actually uses are compared: `SESSION` for EN data and `CAFE_FOLD` for FR data. A Multilingual teacher's training covered both, so both are checked.
  - The check reads the teacher checkpoint's stored `config` through `checkpoints.checkpoint_config`, which memory-maps the file and doesn't load the weights.
  - It runs before any audio is decoded. Checkpoints without a `config` only get a warning.
  - `tests/test_smoke.py` checks that a mismatched run fails.

## 4. `colab` CLI

The `colab` CLI (`google-colab-cli`). Scripted runs from a local terminal (`colab new --gpu T4`, `colab exec -f`, `colab upload/download`) are the base for the planned autoresearch harness. The CLI authenticates with Application Default Credentials; the scopes it needs are listed in `colab skill`.
