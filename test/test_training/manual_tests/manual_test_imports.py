# find_numpy_warn.py
import warnings
import importlib
import traceback
# warnings.filterwarnings("error", category=RuntimeWarning)
import netCDF4
with (
    warnings.catch_warnings()
):  # To catch annoying pydantic x wandb warning - looks like it should be adressed soon: https://github.com/wandb/wandb/issues/10662
    warnings.filterwarnings("ignore")
    import wandb

# Convert RuntimeWarning -> Exception so we get tracebacks

# candidates = [
#     "pandas",
#     "xarray",
#     "netCDF4",
#     "h5py",
#     "torch",
#     "tensorflow",
#     "matplotlib",
#     "pytorch_lightning",
#     "data_preparation"
#     # add any other compiled deps you use
# ]

# failed = []
# for name in candidates:
#     try:
#         print(f"IMPORTING {name} ...", flush=True)
#         importlib.import_module(name)
#         print(f"  OK: {name}")
#     except Exception as e:
#         print(f"  FAIL: {name} -> {type(e).__name__}: {e}")
#         traceback.print_exc()
#         failed.append((name, e))
#         # stop on first failure so you get the full traceback
#         break

# if not failed:
#     print("\nNo RuntimeWarning->Error triggered for the candidate list.")
#     print("If nothing found, try importing project-specific compiled modules (or run a full import of your package).")
#     # Try importing your package last
#     try:
#         print("\nNow importing your package: data_preparation ...")
#         importlib.import_module("data_preparation")
#         print("  OK: data_preparation")
#     except Exception as e:
#         print("  FAIL: data_preparation ->", type(e).__name__, e)
#         traceback.print_exc()
