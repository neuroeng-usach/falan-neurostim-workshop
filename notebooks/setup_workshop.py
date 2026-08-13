"""
setup_workshop.py — everything the notebooks' first cell used to contain.

Students run five lines and see three lines of output. All of the environment
plumbing lives here instead, where it belongs: nobody learns anything about
neurostimulation from reading a dependency pin.

Run with `%run -i setup_workshop.py` so the names it defines (ws, np, lf, tp, ip,
weiss) land in the notebook's namespace.

WHAT IT DOES, IN ORDER
----------------------
1. On Colab: clone or update the repository, then install NEURON and PyFibers.
2. If pip changed numpy, restart the kernel (see below) and ask for one re-run.
3. Put `src/` on the path and drop any cached copies of the workshop modules.
4. Run the physics self-tests quietly.
5. Print one line saying whether the axon simulations will run.

WHY A RESTART IS SOMETIMES NEEDED
---------------------------------
PyFibers pins numpy>=2.2,<2.4 and imports scipy. Colab ships a newer numpy, so
pip downgrades it -- and every compiled package already loaded in the kernel,
scipy included, is then linked against a numpy that is no longer there. It shows
up as `AttributeError: ... '_blas_supports_fpe'`, which looks like a NEURON
problem and is not. No amount of module reloading fixes it; only a new process
does. So this restarts the kernel once, guarded by a sentinel file so it can
never loop, and asks the student to run the cell again.
"""

import importlib.metadata as _md
import os
import pathlib
import subprocess
import sys

REPO_URL = "https://github.com/neuroeng-usach/falan-neurostim-workshop.git"

# The set the workshop is tested against. Installed in ONE pip call so the
# resolver cannot leave numpy and scipy inconsistent with each other.
PINS = ["numpy>=2.2,<2.4", "scipy>=1.15,<1.16", "neuron", "pyfibers"]

IN_COLAB = "google.colab" in sys.modules
_MODULES = ("workshop_setup", "layered_field", "interferential_field",
            "neurostim_model", "weiss", "transcutaneous_plots", "interferential_plots")

_numpy_before = None
try:
    import numpy as _np_probe

    _numpy_before = _np_probe.__version__
except Exception:
    pass

_compile_log = ""

# --- 1. Colab: get the repository and the dependencies ----------------------
if IN_COLAB:
    # An ABSOLUTE target. Cloning to a relative path nests a fresh copy inside
    # the previous clone on every re-run, and Python then serves whichever stale
    # copy it imported first.
    _target = pathlib.Path("/content") / REPO_URL.rsplit("/", 1)[-1][:-4]
    if (_target / ".git").is_dir():
        _p = subprocess.run(["git", "-C", str(_target), "pull", "--quiet", "--ff-only"],
                            capture_output=True, text=True)
        if _p.returncode:
            print("Using the existing clone (could not fast-forward).")
    else:
        print("Fetching the workshop repository...")
        subprocess.run(["git", "clone", "--quiet", REPO_URL, str(_target)], check=True)
    os.chdir(_target)

    print("Installing NEURON and PyFibers (about a minute, first time only)...")
    _pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS],
                          capture_output=True, text=True)
    if _pip.returncode:
        print((_pip.stderr or _pip.stdout)[-2000:])
        raise SystemExit("Install failed -- see the pip output above.")

    # PyFibers ships .mod mechanisms that must be compiled once. Its own
    # post-install check is architecture-specific and can report failure on a
    # non-x86 host even when compilation succeeded, so a non-zero exit is not
    # fatal -- but keep the log: a real failure here is what makes an axon cell
    # die several cells later with a confusing error.
    _c = subprocess.run(["pyfibers_compile"], capture_output=True, text=True)
    _compile_log = (_c.stdout or "") + (_c.stderr or "")

    # --- 2. restart once if numpy moved underneath us -----------------------
    _sentinel = pathlib.Path("/content/.workshop_restarted")
    _numpy_disk = _md.version("numpy")
    if _numpy_before and _numpy_before != _numpy_disk and not _sentinel.exists():
        _sentinel.write_text(_numpy_disk)
        print()
        print("=" * 70)
        print(f"numpy changed {_numpy_before} -> {_numpy_disk}, so the kernel has to")
        print("restart. RUN THIS CELL AGAIN when it comes back -- it will be quick.")
        print("This is normal and happens once per session.")
        print("=" * 70)
        try:
            get_ipython().kernel.do_shutdown(True)   # noqa: F821
        except Exception:
            os.kill(os.getpid(), 9)
        raise SystemExit("Restarting -- please re-run this cell.")

# --- 3. path, and a genuinely fresh import ----------------------------------
ROOT = pathlib.Path.cwd().resolve()
while not (ROOT / "src").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

_src = str(ROOT / "src")
while _src in sys.path:
    sys.path.remove(_src)
sys.path.insert(0, _src)

# Re-running must pick up the files on disk, not the module objects Python
# cached the first time. Without this, updating the repo and re-running the cell
# silently keeps the old code -- which surfaces as an AttributeError for a
# function you can plainly see in the source.
for _m in _MODULES:
    sys.modules.pop(_m, None)

import numpy as np                      # noqa: E402
import interferential_field as ifc      # noqa: E402
import interferential_plots as ip       # noqa: E402
import layered_field as lf              # noqa: E402
import transcutaneous_plots as tp       # noqa: E402
import weiss                            # noqa: E402
import workshop_setup as ws             # noqa: E402

ws.apply_style()
ws.COMPILE_LOG = _compile_log

# --- 4 and 5. report ---------------------------------------------------------
print("repo root:", ROOT)
if ROOT.name in ROOT.parent.parts:
    print("WARNING: nested path. Runtime -> Restart session, then re-run this cell.")
ws.self_test(interferential=True)       # quiet unless something fails
ws.check_environment()
