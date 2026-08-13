# Physics of Electrical Neurostimulation

A four-hour hands-on workshop in which students compute the electric field of a surface
electrode from first principles, drive a real myelinated axon with that field, measure a
strength-duration curve on their own forearm, and then do the whole thing again with two
kilohertz carriers instead of one pulse.

**1st FALAN Latin American Training Program in Neuroscience — Santiago, Chile, August 2026**

Instructors: **Leonel Medina**, **Rodrigo Osorio**, **Cristian Morales**
[NeuroEng@USACH](https://www.neuroeng-usach.cl) · Departamento de Ingeniería Biomédica,
Universidad de Santiago de Chile

| Notebook | Open in Colab |
|---|---|
| 01 · Transcutaneous stimulation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuroeng-usach/falan-neurostim-workshop/blob/main/notebooks/01_transcutaneous_stimulation.ipynb) |
| 02 · Interferential current | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuroeng-usach/falan-neurostim-workshop/blob/main/notebooks/02_interferential_stimulation.ipynb) |

---

## Three ways to run it

**1 · Google Colab — nothing to install.** Click a badge and run the first cell. It clones this
repository and installs NEURON and PyFibers, which takes about a minute.

On a fresh runtime you will run that cell **twice**. Installing PyFibers forces numpy to a version
Colab does not ship by default, and a kernel that has already imported numpy cannot pick up a new
one — so the cell prints what it is doing, restarts the kernel itself, and asks you to run it again.
The second run takes a few seconds. Then you should see `ENVIRONMENT READY`. This is a property of
how Colab preloads numpy, not a fault in the notebook, and it only happens once per runtime.

**2 · Locally.**

```bash
git clone https://github.com/neuroeng-usach/falan-neurostim-workshop.git
cd falan-neurostim-workshop
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate neurostim
python scripts/selftest.py               # ~2 min, verifies the physics and the plots
jupyter lab notebooks/
```

NEURON ships Linux and macOS wheels. On Windows, use WSL. **A local install is the most reliable
route** — no restart dance, and the environment stays put between sessions. If you are running the
workshop rather than attending it, install locally.

**If a cell says a figure "needs NEURON/PyFibers".** The setup cell at the top of each notebook
prints one of two banners. `ENVIRONMENT READY` means everything will run; `FIELD FIGURES ONLY`
means the axon simulations will not, and it tells you why. It checks by building a test axon rather
than merely importing, because `import pyfibers` succeeds even when its NEURON mechanisms were
never compiled — the failure then surfaces several cells later as a confusing error. If you see the
second banner, run `ws.diagnose()` in a new cell: it distinguishes NEURON missing, PyFibers missing,
and PyFibers installed but not compiled. The usual fix is

```bash
pip install -q "numpy>=2.2,<2.4" "scipy>=1.15,<1.16" neuron pyfibers && pyfibers_compile
```

The numpy and scipy bounds matter. PyFibers pins `numpy>=2.2,<2.4` and imports scipy, so on a
machine with a newer numpy — Colab, for instance — pip downgrades numpy, and the scipy that was
built against the newer one then fails to import with
`AttributeError: module 'numpy._core._multiarray_umath' has no attribute '_blas_supports_fpe'`.
That surfaces as "NEURON is missing" even though NEURON is fine. Installing the two together fixes
it, and the setup cell now names this cause specifically when it sees that error.

followed by restarting the kernel. Module 0 and Parts 1–3 work either way — they are pure NumPy.

**3 · No Python at all.** Open [`explorer/index.html`](explorer/index.html) in any browser. It runs
the two core figures — the activating function, and a strength-duration curve you can fit to your
own measurements — entirely client-side. Useful as a backup if the room's wifi fails, and as the
tool students use at the bench to fit the numbers they just measured.

---

## The four hours

| Time | Block | Where |
|---|---|---|
| 0:00–0:10 | Welcome, safety briefing, consent, form groups of 3–4 | [`SAFETY.md`](SAFETY.md) |
| 0:10–1:30 | **A · Transcutaneous stimulation** — field, axon, strength-duration curve, forearm measurement | `notebooks/01`, `handouts/01` |
| 1:30–1:45 | Break (electrodes off, skin checked) | |
| 1:45–2:55 | **B · Interferential current** — beat, crossed montage, motor threshold, afferent block | `notebooks/02`, `handouts/02` |
| 2:55–3:00 | Synthesis: one volume conductor, two ways to deliver current | |
| 3:00–4:00 | **C · Transcranial magnetic stimulation** | (separate material) |

Blocks A and B each interleave simulation with measurement rather than separating them: the
strength-duration sweep takes about a minute of real computation, so groups launch it and go put
electrodes on a forearm while it runs. 
**Read [`SAFETY.md`](SAFETY.md) before the session, and before volunteering.** It has the
exclusion criteria, the consent script, the kit list and the rules that apply while any
current is flowing.

---

## What is in here

```
notebooks/
  01_transcutaneous_stimulation.ipynb   Field -> axon -> strength-duration curve
  02_interferential_stimulation.ipynb   Two carriers -> beat -> deep activation
src/
  layered_field.py          Two-layer volume conductor, method of images, self-tested
  interferential_field.py   Four-pad placement + beat/modulation arithmetic
  neurostim_model.py        MRG myelinated axon via PyFibers/NEURON, threshold searches
  weiss.py                  Strength-duration law + published reference values (no NEURON)
  workshop_setup.py         Bootstrap, figure style, and the widget-optional Panel helper
  transcutaneous_plots.py   Every figure of notebook 01
  interferential_plots.py   Every figure of notebook 02
handouts/
  01_transcutaneous_handout.md / .docx  Forearm protocol + recording tables
  02_interferential_handout.md / .docx  Crossed-montage protocol + recording tables
explorer/
  index.html                Zero-install browser version of the two core figures
scripts/
  selftest.py               Runs every module self-test and renders every figure headlessly
SAFETY.md                   Exclusion criteria, consent script, kit list, rules while stimulating
```

Facilitator notes, the minute-by-minute run of show and the device-specific bench procedures are
kept separately and deliberately not published: they contain the answers to the predict-then-run
questions the notebooks are built around. Instructors reusing this material can request them from
the authors.

---

## Why you can trust the numbers

The point of the workshop is that nothing is a black box, so the physics is checked rather than
asserted. `python scripts/selftest.py` runs all of it:

- **Current conservation.** The two-layer field integrates to the injected current within 0.4%.
  This check earned its place: an earlier version of `layered_field.py` used a single image
  instead of the full multiple-reflection series and failed this test by ~60%.
- **Analytic limits.** Potential is continuous across the layer boundary; the homogeneous case
  reproduces the textbook point-source formula exactly; insulating and conducting backings move
  the potential in the right directions.
- **Disc electrode.** A vanishingly small disc converges to the point source (relative difference
  3.5 × 10⁻⁶); peak potential falls monotonically with pad radius; current conservation still
  holds at a realistic 5 mm radius.
- **Bipolar pair.** Antisymmetric about the pad-pair midpoint; converges to the single-source
  field near one pad when the pair is far apart.
- **Interferential arithmetic.** Reduces exactly to the two-layer solution for one channel;
  modulation depth is 100% for collinear and 0% for perpendicular channel currents; the
  four-lobe "clover" of deep modulation and the central null are both verified numerically.
- **Against the literature.** With the default 10 µm fibre at 8 mm depth, the simulated cathodic
  threshold is 1.73 mA and the anodic 6.57 mA — a **3.8× asymmetry**, matching the classic result
  for point-source nerve stimulation. Fitting Weiss to the simulated curve gives chronaxie
  **0.40 ms** and rheobase **0.72 mA**, against a published near-nerve human ulnar chronaxie of
  0.32 ms (SD 0.17) and rheobase 0.91 mA (SD 0.37), measured awake with a near-nerve needle
  (Tsui et al. 2014, *Anaesthesia* 69:678–682).
- **Independently, with a different method.** The same bipolar pad montage was also solved on a
  meshed, anisotropic forearm with a finite-element model, as a cross-check on the analytical
  two-layer solution students use. That comparison is instructor material rather than a teaching
  module — it needs a mesher and careful unit bookkeeping, which is not a good use of the session —
  but it is why we are comfortable teaching the simple model.

---

## Equipment

Full kit list in [`SAFETY.md`](SAFETY.md). In short: a two-channel TENS/EMS unit with a monophasic
rectangular mode and an independently programmable kHz mode (these sessions were built around a
**TENS Ultima Neo**), self-adhesive electrodes, gel, alcohol wipes, a blunt plastic probe, a ruler,
a drinking straw and tape, and a 1 kΩ resistor with a multimeter for the device checks.

One warning worth taking seriously if you reuse this material with a consumer stimulator: several of
its ordinary behaviours produce a smooth, plausible-looking and completely meaningless
strength-duration curve. Trains delivered where you expected single pulses; an output clipping
against its voltage ceiling; a displayed pulse width that is per phase rather than total; a
displayed current that was never measured in the first place. The handout gives students the
diagnostic — plot charge against pulse width and ask whether the fitted chronaxie is physically
possible — and the bench tests that rule each cause out are part of the instructor material.

---

## Handouts

Both handouts are Markdown so they live in version control and render on GitHub; print them from
the browser or an editor preview. The interferential handout is also kept as the original `.docx`
for anyone who would rather edit it in Word. Keep the two in sync if you change either.

---

## Citing and reusing

Code in `src/`, `explorer/` and `scripts/` is MIT licensed. The teaching materials — notebooks,
handouts and instructor documents — are CC BY 4.0. See `LICENSE` and `CITATION.cff`.

The axon model comes from [PyFibers](https://github.com/wmglab-duke/pyfibers) (Duke WMGLab), a
NEURON implementation of the McIntyre–Richardson–Grill double-cable model. If you use this
material, please cite PyFibers and NEURON as well as this repository.

## Acknowledgements

Built for the FALAN School at Universidad de Santiago de Chile. Thanks to the students who
volunteered their forearms, and who will discover that the crossing point of a crossed montage is
not where the stimulation is strongest.
