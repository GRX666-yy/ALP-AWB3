# ALP-AWB3

English | [中文](README.md)

ALP-AWB3 is a low-latency matheuristic for the **0–1 multidimensional knapsack problem**. Given item profits, resource consumptions, and resource capacities, it returns an item subset satisfying every capacity constraint.

The method constructs a feasible solution using adaptive linear programming and jointly reselects items in at most three restricted cores, each containing at most 20 items. Additional windows alternate removal and insertion candidates and roll forward by retaining 10 items and introducing 10 new ones. Price-assisted bitmaps accelerate resource-feasible pairing of half-table states; half-table and complete-subproblem caches reuse overlapping computation. The three-core cap includes the initial core.

This repository provides the solver, a Python API, and a JSON command-line interface.

## Installation

The validated Python version is **3.12**. The solver uses NumPy and the HiGHS Python interface (`highspy`), with versions pinned in `requirements.txt`.

```bash
git clone https://github.com/GRX666-yy/ALP-AWB3.git
cd ALP-AWB3
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Replace `python` with `python3` if that is the command used by your system.

## Python API

The following six-item, two-resource example demonstrates the Python API.

```python
from solve import solve

result = solve(
    values=[9, 8, 12, 6, 10, 7],
    weights=[[5, 2], [2, 5], [6, 6], [3, 1], [1, 4], [4, 3]],
    capacity=[10, 10],
    budget_s=0.05,
    seed=0,
    name="example",
)
print("profit:", result["value"])
print("selected item indices:", result["selected_ids"])
print("algorithm return time (s):", result["elapsed_s"])
```

`values` is a length-N profit array; `weights` is an **N×M matrix, with items in rows and resources in columns**; `capacity` is a length-M capacity array. N and M must be positive. Profits and capacities must be positive and finite; resource consumptions must be nonnegative and finite. The budget must be finite and nonnegative. An individually oversized item cannot enter a feasible returned subset.

## Command-line interface

Save the following as UTF-8 `instance.json`:

```json
{
  "name": "example",
  "values": [9, 8, 12, 6, 10, 7],
  "weights": [[5, 2], [2, 5], [6, 6], [3, 1], [1, 4], [4, 3]],
  "capacity": [10, 10]
}
```

```bash
python -B solve.py --input instance.json --budget-ms 50 --seed 0 --output solution.json
```

Omit `--output` to print the result to the terminal. CLI budgets are in milliseconds; the Python `budget_s` argument is in seconds. Run `python solve.py --help` for available arguments. The output file's parent directory must already exist.

Main result fields:

|Field|Meaning|
|---|---|
|`value`|Total profit of the admitted feasible subset|
|`selected_ids`|**Zero-based** indices of selected items|
|`elapsed_s`|Actual return time of the internal solving procedure, in seconds|
|`wrapper_elapsed_s`|Array conversion, input checks, solving, and post-solve validation time|
|`budget_s`|Improvement-admission budget, in seconds|
|`events`|Admitted improvement trajectory, including time, profit, and packed subsets|
|`stats`|Construction, window, cache, and bitmap-pairing statistics|
|`model_version`|Configuration version identifier, `88`|

`selected` is an internal hexadecimal packed bit vector; use `selected_ids` for ordinary applications. Neither timing field includes Python startup, module imports, or JSON file I/O.

## Budget and execution semantics

The budget controls the **deadline for admitting feasible improvements**; actual return can occur later. If an extremely short budget permits no positive-profit admission, the solver returns the empty feasible subset with profit 0.

The implementation uses temporary module patching. Calls in the same process must therefore be **sequential**. Use separate independent processes for parallel instances. Numerical-library thread counts are set to one when the solver is imported.

## Code layout and version

```text
solve.py                 Array API and JSON command-line entry point
requirements.txt         Dependencies of the current solver
runtime/
  problem.py             Instances, feasibility, admission, greedy fallback
  lp.py                  HiGHS and adaptive LP construction
  candidates.py          LP features and candidate ordering
  prefix.py, clock.py    E100 prefix and unified timing
  repair.py, ordering.py Initial-core interface and window ordering
  engine.py, core.py     Three-core rolling and restricted optimization
  bitmap.py, common.py   Bitmap pairing and dependency assembly
```

Configuration identifier: `mdkp-bitmap-three-core-v88`.
