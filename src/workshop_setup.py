"""
workshop_setup.py

Shared plumbing for both student notebooks of the *Physics of Electrical
Neurostimulation* workshop (transcutaneous stimulation and interferential
current). Nothing in here is physics -- the physics lives in
`layered_field.py`, `interferential_field.py` and `neurostim_model.py`. This
module exists so that the two notebooks share ONE copy of everything that is
not physics:

  * `apply_style()`      -- the figure style, previously copy-pasted into both
                            notebooks' setup cells.
  * `self_test()`        -- runs the validated self-tests of whichever physics
                            modules a notebook actually uses.
  * `have_neuron()`      -- NEURON/PyFibers is optional. Modules that only need
                            the volume-conductor field must still work without
                            it (e.g. on a machine where NEURON was not
                            installed), so both notebooks degrade gracefully
                            instead of dying in the setup cell.
  * `Panel`              -- one slider-panel implementation used by every
                            interactive cell in both notebooks. It replaces
                            ~30 lines of per-cell ipywidgets boilerplate AND
                            removes the need for a separate "fallback"
                            notebook: if ipywidgets is unavailable, the same
                            cell simply runs once with the default values and
                            tells the student how to change them by hand.

WHY A SHARED PARAMETER STORE
----------------------------
The original transcutaneous notebook deliberately let Module 1 and Module 2
reuse the tissue parameters the student had set with Module 0's sliders (they
literally read `sigma1_sl.value` from an earlier cell). That coupling is
pedagogically the point -- "the SAME field you just explored now drives a real
axon" -- but it made the cells order-dependent in a way that was invisible.
`Panel` makes it explicit: every control writes its value into the module-level
dict `STATE`, and a panel can declare `inherit=[...]` to pull parameters that an
earlier module established. `print(ws.STATE)` at any time shows the full
parameter set the student is currently working with.

HEADLESS / CI USE
-----------------
Set the environment variable `WORKSHOP_AUTORUN=1` and every `Panel.show()` also
executes once immediately with its current values. That is what
`scripts/selftest.py` and the notebook-execution check use to exercise the
plotting code without a human clicking buttons.
"""

from __future__ import annotations

import os
from collections import namedtuple

# --- Shared parameter store -------------------------------------------------
STATE: dict = {}

def _flag(name):
    return os.environ.get(name, "") not in ("", "0", "false", "False")


# Set WORKSHOP_AUTORUN=1 to make every Panel.show() also run once immediately
# (used by the notebook-execution check -- nobody is there to click a button).
AUTORUN = _flag("WORKSHOP_AUTORUN")

# Set WORKSHOP_FAST=1 to shorten the deliberately slow cells (the strength-
# duration sweep) so the notebooks can be executed end to end as a smoke test.
# Never set this during a workshop: the point of that cell is that it takes a
# real minute of real computation.
FAST = _flag("WORKSHOP_FAST")

# --- Optional-dependency probes ---------------------------------------------
try:  # ipywidgets + a live IPython kernel
    import ipywidgets as _widgets
    from ipywidgets import Layout as _Layout
    from IPython.display import display as _display, clear_output as _clear_output

    HAVE_WIDGETS = True
except Exception:  # pragma: no cover - exercised on plain-python runs
    _widgets = None
    HAVE_WIDGETS = False

_NEURON = "unchecked"
NEURON_ERROR: str = ""      # why the import failed, for diagnose()

# Set WORKSHOP_NO_NEURON=1 to pretend NEURON is absent. Lets an instructor
# rehearse the degraded path (field-only) on a machine that has it installed.
_FORCE_NO_NEURON = _flag("WORKSHOP_NO_NEURON")

INSTALL_HINT = "pip install neuron pyfibers && pyfibers_compile"

# The bootstrap cell stores the output of `pyfibers_compile` here, so that a real
# compilation failure can be shown at the moment it matters instead of being
# swallowed by the installer's architecture-specific self-check.
COMPILE_LOG: str = ""


def have_neuron(verbose: bool = True):
    """Return the `neurostim_model` module, or None if NEURON/PyFibers is missing.

    Cached, so calling this in several cells costs nothing. NEURON is only
    needed for the cells that simulate an actual axon; the volume-conductor
    field modules are pure NumPy and always available.
    """
    global _NEURON, NEURON_ERROR
    if _NEURON == "unchecked":
        if _FORCE_NO_NEURON:
            _NEURON = None
            NEURON_ERROR = "disabled by WORKSHOP_NO_NEURON=1"
        else:
            try:
                import neurostim_model as nm

                _NEURON = nm
            except Exception as exc:
                _NEURON = None
                NEURON_ERROR = f"{type(exc).__name__}: {exc}"
                if verbose:
                    print(f"NEURON/PyFibers not available here ({NEURON_ERROR}).\n"
                          f"Field-only cells still work. To enable the axon simulations:\n"
                          f"    {INSTALL_HINT}")
    return _NEURON


def check_environment(interferential: bool = False):
    """Print one unmissable line saying which half of the notebook will run.

    Importing `pyfibers` succeeds even when its NEURON mechanisms have not been
    compiled -- the failure only appears later, when a cell actually builds an
    axon. So this does not merely import: it builds a tiny fibre. That is the
    only check that distinguishes "installed" from "actually working", and it is
    the difference between finding out now and finding out in front of a room.
    """
    nm = have_neuron(verbose=False)
    detail = NEURON_ERROR
    ok = nm is not None
    if ok:
        try:
            nm.build_axon(diameter=10.0, n_nodes=11)
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"

    if ok:
        # The happy path should be one calm line, not a wall of "=".
        print("environment ready: field figures and axon simulations will both run.")
        return True

    bar = "=" * 72
    print(bar)
    if True:
        modules = "Part 4" if interferential else "Modules 1 and 2"
        print(f"FIELD FIGURES ONLY — {modules} will NOT run in this kernel.")
        print(f"  reason: {detail or 'neurostim_model could not be imported'}")
        print(f"  fix:    {INSTALL_HINT}")
        print("  then restart the kernel and run this cell again.")
        low = (detail or "").lower()
        if "_multiarray_umath" in low or "_blas_supports_fpe" in low or "numpy" in low:
            print()
            print("  This is a numpy/scipy binary mismatch, not a NEURON problem.")
            print("  PyFibers pins numpy>=2.2,<2.4 and imports scipy, so pip downgrades")
            print("  numpy; a scipy built against a newer numpy then fails on import.")
            print("  Install the tested set and RESTART the kernel:")
            print('      pip install -q "numpy>=2.2,<2.4" "scipy>=1.15,<1.16" neuron pyfibers')
            print("  then Runtime -> Restart session and run the setup cell again.")
        else:
            print("  If pyfibers imported but building a fibre failed, you are missing the")
            print("  compiled NEURON mechanisms: run  pyfibers_compile  and restart.")
        if COMPILE_LOG.strip():
            tail = [ln for ln in COMPILE_LOG.strip().splitlines() if ln.strip()][-6:]
            print("  last lines of pyfibers_compile:")
            for ln in tail:
                print("   ", ln[:150])
    print(bar)
    return False


def where_am_i():
    """Show which files this kernel is ACTUALLY running.

    On Colab, re-running a bootstrap that clones to a relative path nests a new
    copy of the repository inside the previous one, and Python keeps serving
    whichever copy it imported first. The symptom is an AttributeError for a
    function that is plainly present in the source you are looking at. This
    prints the resolved file path of every workshop module so that mismatch is
    visible immediately.
    """
    import sys as _sys

    print("cwd :", os.getcwd())
    for name in ("workshop_setup", "layered_field", "interferential_field",
                 "neurostim_model", "weiss", "transcutaneous_plots",
                 "interferential_plots"):
        mod = _sys.modules.get(name)
        print(f"  {name:22s}", getattr(mod, "__file__", "not imported"))
    nested = [p for p in os.getcwd().split(os.sep) if p]
    if len(nested) != len(set(nested)):
        print("\nA directory name repeats in the path above: you are almost certainly")
        print("running a nested, stale clone. Runtime -> Restart session, then re-run")
        print("the setup cell (the current bootstrap clones to an absolute path and")
        print("cannot nest).")


def diagnose():
    """Print everything needed to work out why the axon simulations will not run.

    Paste the output when asking for help; it distinguishes the three failure
    modes that look identical from the notebook: NEURON absent, PyFibers absent,
    and PyFibers present but its .mod mechanisms never compiled.
    """
    import platform
    import sys as _sys

    print("python     :", _sys.version.split()[0], "|", platform.platform())
    print("executable :", _sys.executable)
    print("ipywidgets :", "yes" if HAVE_WIDGETS else "no (panels fall back to one run)")
    for name in ("numpy", "matplotlib", "neuron", "pyfibers"):
        try:
            mod = __import__(name)
            print(f"{name:11s}: {getattr(mod, '__version__', 'installed')}")
        except Exception as exc:
            print(f"{name:11s}: NOT IMPORTABLE — {type(exc).__name__}: {exc}")
    try:
        import pathlib

        import pyfibers

        mod_dir = pathlib.Path(pyfibers.__file__).parent / "MOD"
        built = [d.name for d in mod_dir.iterdir()
                 if d.is_dir() and (d / "special").exists()] if mod_dir.is_dir() else []
        print("pyfibers mechanisms compiled for:", built or "NONE — run pyfibers_compile")
    except Exception as exc:
        print("pyfibers mechanisms: cannot check —", type(exc).__name__, exc)
    nm = have_neuron(verbose=False)
    print("neurostim_model:", "importable" if nm else f"NO — {NEURON_ERROR}")
    if nm:
        try:
            nm.build_axon(diameter=10.0, n_nodes=11)
            print("build a test axon: OK")
        except Exception as exc:
            print(f"build a test axon: FAILED — {type(exc).__name__}: {exc}")
            print("  -> mechanisms missing or mismatched; run pyfibers_compile, restart kernel")
    if COMPILE_LOG.strip():
        print("\npyfibers_compile output (last 12 lines):")
        for ln in [x for x in COMPILE_LOG.strip().splitlines() if x.strip()][-12:]:
            print("   ", ln[:150])


# --- Figure style (single source of truth for both notebooks) ---------------
STYLE = {
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "grid.color": "#e1e0d9",
    "font.size": 10,
    "legend.framealpha": 0.9,
}


def apply_style():
    """Apply the workshop figure style. Colours for the physics itself (field
    colormap, depolarizing/hyperpolarizing pair, channel A/B) live in
    `layered_field.COLOR_*` / `interferential_field.COLOR_*` so that the same
    quantity is always the same colour in every figure of both notebooks."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)


def self_test(interferential: bool = False, verbose: bool = False):
    """Run the validated physics self-tests. Quiet unless something fails.

    These are not decoration: `layered_field._self_test()` includes a
    current-conservation check that once caught a genuinely wrong (truncated)
    image series, and `interferential_field._self_test()` checks that the beat
    arithmetic reduces to the two-layer solution and that modulation depth is
    100% for collinear and 0% for perpendicular channel currents.

    They used to print about fifteen lines of numbers into the first cell of a
    teaching notebook. Now they print one line, and the detail appears only when
    a check fails -- which is the only time anyone wants to read it. Pass
    verbose=True to see it all.
    """
    import contextlib
    import io

    import layered_field as lf

    checks = [("two-layer field, disc, bipolar", lf._self_test)]
    if interferential:
        import interferential_field as ifc

        checks.append(("interference and beat arithmetic", ifc._self_test))

    for label, fn in checks:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                fn()
        except Exception:
            print(f"SELF-TEST FAILED: {label}")
            print(buf.getvalue())
            raise
        if verbose:
            print(buf.getvalue(), end="")

    names = " + ".join(label for label, _ in checks)
    print(f"physics self-tests passed ({names}).")


# --- Control specs ----------------------------------------------------------
Num = namedtuple("Num", "key label lo hi step value fmt")
Choice = namedtuple("Choice", "key label options value")


def num(key, label, lo, hi, step, value, fmt=None):
    """A continuous control. `key` must be the keyword name of the plotting
    function's parameter, so panels stay self-documenting."""
    return Num(key, label, lo, hi, step, value, fmt)


def choice(key, label, options, value):
    """A discrete control. `options` may be values, or (label, value) pairs."""
    return Choice(key, label, list(options), value)


class Panel:
    """One slider panel + Run button for one plotting function.

    Parameters
    ----------
    func : callable
        Plotting function, called with keyword arguments only.
    controls : list of Num/Choice
        The parameters this panel exposes. Initial values come from `STATE` if
        an earlier panel already set that key, otherwise from the spec -- so
        tissue parameters chosen in Module 0 carry into Modules 1 and 2.
    inherit : sequence of str
        Parameter names taken from `STATE` at run time without showing a
        control for them here (they belong to an earlier module).
    button : str
        Button label.
    note : str
        One line printed above the controls; use it for "this takes ~1 min".
    """

    def __init__(self, func, controls, inherit=(), button="Run", note=None):
        self.func = func
        self.controls = list(controls)
        self.inherit = list(inherit)
        self.button = button
        self.note = note
        for c in self.controls:
            STATE.setdefault(c.key, c.value)

    # -- value handling ----------------------------------------------------
    def _kwargs(self):
        missing = [k for k in self.inherit if k not in STATE]
        if missing:
            raise RuntimeError(
                f"{self.func.__name__} inherits {missing} from an earlier module, but "
                f"those are not set yet. Run the earlier module's cell first."
            )
        kw = {k: STATE[k] for k in self.inherit}
        kw.update({c.key: STATE[c.key] for c in self.controls})
        return kw

    def run(self, **overrides):
        """Run once. Any keyword given here is written into STATE first, so
        `panel.run(sigma1=0.05)` is the no-widget equivalent of moving a
        slider and clicking the button."""
        STATE.update(overrides)
        kw = self._kwargs()
        if self.note:
            print(self.note)
        return self.func(**kw)

    # -- display -----------------------------------------------------------
    def _build_widgets(self):
        style = {"description_width": "190px"}
        layout = _Layout(width="460px")
        ws = {}
        for c in self.controls:
            v = STATE[c.key]
            if isinstance(c, Num):
                kw = dict(value=v, min=c.lo, max=c.hi, step=c.step,
                          description=c.label, style=style, layout=layout)
                if c.fmt:
                    kw["readout_format"] = c.fmt
                ws[c.key] = _widgets.FloatSlider(**kw)
            else:
                ws[c.key] = _widgets.Dropdown(options=c.options, value=v,
                                              description=c.label, style=style,
                                              layout=layout)
        return ws

    def show(self):
        """Display the panel (or, without ipywidgets, run once and explain)."""
        if not HAVE_WIDGETS:
            print("ipywidgets is not available here, so this cell runs once with "
                  "the default values below.")
            print("To explore, re-run with overrides, e.g.:")
            ex = self.controls[0]
            print(f"    panel.run({ex.key}={STATE[ex.key]!r})")
            for c in self.controls:
                print(f"    {c.key:24s} = {STATE[c.key]!r}   # {c.label}")
            self.run()
            return self

        ws = self._build_widgets()
        btn = _widgets.Button(description=self.button, button_style="primary",
                             layout=_Layout(width="280px"))
        out = _widgets.Output()

        def on_click(_):
            with out:
                _clear_output(wait=True)
                STATE.update({k: w.value for k, w in ws.items()})
                try:
                    self.run()
                except Exception as exc:  # keep the workshop moving
                    print(f"{type(exc).__name__}: {exc}")

        btn.on_click(on_click)
        children = list(ws.values()) + [btn, out]
        if self.note:
            children.insert(0, _widgets.HTML(f"<i>{self.note}</i>"))
        _display(_widgets.VBox(children))
        if AUTORUN:
            on_click(None)
        return self


def parse_pairs(text):
    """Parse a two-column paste-in of measurements into [(x, y), ...].

    Accepts commas, whitespace or tabs, ignores blank lines and anything that
    is not a pair of numbers -- so students can paste straight from the handout
    table, comments and all.
    """
    pts = []
    for line in (text or "").strip().splitlines():
        parts = line.replace(",", " ").replace("\t", " ").split()
        if len(parts) < 2:
            continue
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return pts
