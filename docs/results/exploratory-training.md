# Exploratory training status

2026-10-31 · Marchel Andrian Shevchenko · Data Sorcerers.

**Preparation code and synthetic verification are complete. Acquired-data execution is blocked; no new frozen manifest or model-performance result is claimed.**

## Observed execution

Colab refused a T4 allocation because of GPU usage limits. No compute units were purchased. An editable private notebook copy was saved in the active Colab account. CPU allocation initially encountered the active-session limit; the idle historical OcuWeave runtime was released without deleting its saved Drive artifacts.

The new CPU runtime connected, but its initial Drive mount failed with `ValueError: mount failed`. One recovery using `force_remount=True` and a 180-second timeout also failed. Setup did not reach source checkout, acquired-data freezing or smoke checks. The saved notebook retains both attempted executions and their errors. A fallback browser attempt to read the historical manifest also timed out. No source data were edited by these attempts.

The connector-visible project folder did not resolve to the same folder as the active browser account; the working notebook was therefore copied through Colab into the active account, without changing sharing permissions.

| Requested stage | Observed status |
|---|---|
| New development manifest | Grouping implementation tested; actual-data freeze pending |
| Nine locked configurations | Matrix and locking code implemented; data-bound receipt pending |
| GPU smoke | Blocked by GPU quota |
| CPU smoke on acquired images | Not executed: Drive mount failed |
| Nine GPU research runs | **0/9 completed** |
| Original-grid validation | Evaluator tested on synthetic fixtures; no new acquired-data scores |
| Historical test and EMR | No new test evaluation or EMR training |

Do not report projected split counts as observed counts. The old manifest, candidate report and quality queue must be read and verified before recording revised counts and hashes. No post-run protected-hash verification occurred because setup did not reach that stage.

## Verified software

The grouping/training implementation at `0718ce3358cb8b61608efe8c7e28781868d554b8` passed CI. The subsequent source-reference and evidence-archive correction at `52ea538268bfcf142ea22c30c6df406c9fdfe168` also passed [GitHub Actions](https://github.com/chenko-the-sorcerers/ocuweave/actions/runs/41179434875).

- 43 Python tests: 37 existing plus six grouping and execution-gate tests; CMake/CTest, native parity and synthetic manifest audit passed in CI.
- Three dependency-free grouping tests and Python syntax compilation passed locally.
- Notebook code syntax checked locally. End-to-end Colab execution and result-figure inspection remain incomplete.

## Resume

After the existing project Drive mounts and CUDA is available, run the release-pinned notebook from the beginning. It creates a fresh evidence directory, checks historical artifacts, freezes the provisional family, locks the common budget, runs three full smoke batches per variant, then executes all nine runs in seed order. Never reuse smoke weights, silently lower the budget, or overwrite historical outputs. Preserve failed/partial attempts and record actual counts, hashes, GPU telemetry and completion receipts when execution succeeds. Clinical annotation and patient linkage remain unverified. FairVLMed/EMR stay outside this experiment.
