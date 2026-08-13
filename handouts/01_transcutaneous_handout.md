# Transcutaneous Electrical Stimulation

**Student handout — Block A** · Physics of Electrical Neurostimulation
Taller Escuela de Neurociencia (FALAN School) — Santiago, August 2026
NeuroEng@USACH · Leonel Medina, Rodrigo Osorio, Cristián Morales

Name: _______________________________   Partner(s): _______________________________

Volunteer initials: __________   Date: __________   Consent confirmed by: __________

Companion notebook: `notebooks/01_transcutaneous_stimulation.ipynb`
Safety, exclusion criteria and consent script: `SAFETY.md` — read before volunteering.

---

## By the end of this session you will be able to

1. Read the electric field of a surface electrode pair over layered tissue, and name the feature of
   that field that actually excites a nerve.
2. Predict where an axon fires from the activating function, and check that prediction against a
   nonlinear membrane simulation.
3. Measure a motor strength-duration curve on a forearm and fit rheobase and chronaxie to it.
4. Say why cathodic and anodic thresholds differ — and identify the condition under which they
   become identical.
5. Recognise, from the shape of your own data, when a stimulator is lying to you.

## Session map (80 min)

| | |
|---|---|
| Module 0 — the electric field itself (notebook) | ~20 min |
| Module 1 — one stimulus: does the fibre fire? (notebook) | ~20 min |
| Module 2 — strength-duration curve: model vs. your forearm | ~30 min |
| Wrap-up and comparison | ~10 min |

**Work in parallel.** The simulated sweep takes about a minute of real computation. Start it, then
go and set up the forearm measurement while it runs.

## What you need

Two-channel TENS/EMS unit with a **monophasic rectangular** mode · two self-adhesive electrodes
(3 cm circular, or 5 × 5 cm) · conductive gel · alcohol wipes · skin marker · ruler · drinking
straw and tape · 1 kΩ resistor and multimeter (for the device checks) · a volunteer.

---

## Safety — read before switching anything on

- Proceed only with informed verbal consent, and only with a facilitator present.
- **Do not stimulate** anyone with a pacemaker, ICD or other implanted electronic device, a seizure
  disorder, an arrhythmia, broken or inflamed skin at the electrode site, or who is or may be
  pregnant.
- Never place electrodes across the chest, the neck or the head, and never stimulate across the
  heart. This session uses **one forearm only**.
- Always start at 0 mA and ramp slowly. Stop as soon as you reach the endpoint you are looking for.
  Do not keep increasing to see how strong it gets.
- Change amplitude only while both electrodes are firmly attached. Never peel an electrode off with
  current flowing; return the amplitude to 0 first.
- In monophasic mode there is a net direct current. At 1–2 Hz this is negligible — about 4 µA
  average for a 20 mA, 200 µs pulse — but **do not leave a monophasic train running at high
  frequency**, and inspect the skin under the cathode between blocks. Redness appears there first.
- Fresh electrodes per volunteer. Stop immediately on any report of pain or burning.

---

## Module 0 — The electric field of an electrode pair

Work through Module 0 in the notebook before touching the hardware, and record your answers.

**Predict first, then run.** With the cathode on and the muscle layer *more* conductive than the
shallow skin+fat layer, will the deep layer concentrate current toward itself or spread it out?

My prediction: ______________________________________________

What happened: ______________________________________________

**Pad radius.** Raise the electrode radius from 0 (a point) to 5–10 mm.

| | goes up / goes down / no change |
|---|---|
| Peak current density under the pad | |
| Sharpness of the activating function at the fibre | |

In one sentence, the trade-off an electrode designer is making: ____________________________

______________________________________________________________________________

**Pad separation.** What happens to the field at the fibre as the two pads move closer together?
Why does this matter for where you place electrodes on a real arm?

______________________________________________________________________________

______________________________________________________________________________

---

## Module 1 — Cathodic vs. anodic

Run Module 1 near threshold, then well above it.

| | cathodic | anodic | ratio |
|---|---|---|---|
| Threshold amplitude, monopolar (mA) | | | |
| Threshold amplitude, bipolar, symmetric (mA) | | | |

**Before you run the bipolar case, predict the ratio:** _______

Explain the bipolar result. Is it a bug?

______________________________________________________________________________

______________________________________________________________________________

**Initiation site.** Does the actual initiation site stay locked to the peak of the activating
function as you push the amplitude far above threshold, or does it drift? Why would you expect
the linear prediction to be *most* accurate exactly at threshold?

______________________________________________________________________________

---

## Module 2 — Strength-duration curve on a forearm

### Set up the stimulator

| Parameter | Set to | What the device displays |
|---|---|---|
| Waveform | monophasic rectangular | |
| Frequency | **lowest available (1–2 Hz)** | |
| Pulse width | see the data table below | |
| Amplitude | start at 0 | |

Two things to settle before any data is worth recording:

**Is the pulse width per phase or total?** Check the manual or the display. Note which:
______________

**Which lead is the cathode?** Do not trust a red/black convention. Keep both pads stuck down and
still, measure the threshold, then **swap only the cables at the device** and measure again. The
configuration with the *lower* threshold has the cathode over the nerve. Expect roughly 2–3×.

Threshold, configuration 1: ______ mA   Configuration 2: ______ mA   Cathode is the ______ lead.

### Place the electrodes

1. Volunteer seated, forearm supported, palm up, wrist and fingers relaxed and free to move.
   Clean the skin with alcohol and let it dry.
2. **Active** electrode over the median nerve at the wrist crease. **Return** 6–8 cm proximal on
   the *dorsal* surface, over tendon or bone, so it produces no contraction of its own.
3. Keep at least 5 cm between electrode centres. Close spacing sends most of the current through
   superficial skin: you get pain before contraction.
4. Find the spot first. At 200 µs and clearly suprathreshold amplitude, move the active pad in
   ~5 mm steps until thumb abduction is cleanest at the lowest current. **Mark the outline with
   the skin marker and do not move it again** — a 1 cm shift can change threshold by 30%, which is
   the size of the effect you are trying to measure.
5. Tape a drinking straw to the distal phalanx of the thumb, pointing at a ruler stood on edge.
   A 2 mm twitch becomes a 2 cm pointer excursion, and you get a graded reading instead of
   yes/no.

### Define the endpoint before you collect data

**Motor threshold = the lowest amplitude that produces a visible response in 5 of 10 pulses.**

Hunt it: raise the amplitude until the response is unmistakable, then step down until you lose
5/10, then confirm with a further 10 pulses. The person scoring twitches must not be able to see
the device settings, and the pulse-width order must be randomised. Record the pointer excursion in
millimetres as well as the yes/no — it makes the threshold reproducible between observers.

### Data

Randomised pulse-width order for your group (from the facilitator, or shuffle it yourselves):
______  ______  ______  ______  ______

| Pulse width (µs) | ms | Hits / 10 | Threshold (mA) | Pointer (mm) | Notes |
|---|---|---|---|---|---|
| 50 | 0.05 | | | | |
| 100 | 0.10 | | | | |
| 150 | 0.15 | | | | |
| 200 | 0.20 | | | | |
| 250 | 0.25 | | | | |
| **50 (repeat, last)** | 0.05 | | | | drift check |

Drift: first and last measurement of 50 µs differ by ______ mA (______ %).

If that difference is comparable to the change you see across pulse widths, your curve is measuring
skin hydration, not membrane kinetics. Say so in your write-up.

### Fit

Enter the pulse width and threshold columns into `MY_DATA` in Module 2 of the notebook, or into
`explorer/index.html` if you would rather not wait for Python.

Weiss / Lapicque law:  **I(t) = I_rheobase · (1 + chronaxie / t)**

| | Your forearm | This model | Published (near-nerve ulnar) |
|---|---|---|---|
| Rheobase (mA) | | | 0.91 (SD 0.37) |
| Chronaxie (ms) | | | 0.32 (SD 0.17) |

---

## Questions

**1.** Your rheobase is almost certainly higher than the published value, and your chronaxie
probably is not far off. Why does the distance from electrode to nerve affect one of these two
parameters far more than the other?

______________________________________________________________________________

______________________________________________________________________________

**2.** Suppose your five thresholds came out nearly identical, around 10 mA. Compute the charge
Q = I·t for each and plot Q against t. The Weiss law predicts a straight line with slope = rheobase
and intercept = rheobase × chronaxie. What chronaxie does a line through the origin imply, and why
is that impossible for a myelinated axon? Name two things about the apparatus that would produce
this result.

______________________________________________________________________________

______________________________________________________________________________

**3.** In Module 1 you found cathodic and anodic thresholds identical for a symmetric bipolar pair.
Your real forearm montage is bipolar. Does polarity matter on a real arm, then? What would have to
be true for it to matter, and does your swap-the-cables measurement support it?

______________________________________________________________________________

______________________________________________________________________________

**4.** You adjusted σ₁, σ₂ and fibre depth in Module 0 until the model's rheobase
matched yours. What did the values you needed imply about where the median nerve actually sits
under your pads, and is that anatomically plausible?

______________________________________________________________________________

---

## Quick reference

- Weiss / Lapicque: I(t) = I_rh · (1 + τ_ch / t). Rheobase = threshold for an infinitely long
  pulse; chronaxie = the pulse width at which threshold is exactly twice rheobase.
- Linearised for fitting: I = I_rh + (I_rh·τ_ch)·(1/t) — regress I against 1/t.
- Activating function f(x) = ∂²V_e/∂x². Positive → depolarizing, negative → hyperpolarizing. The
  *curvature* of the extracellular potential drives the membrane, not its value.
- Two-layer reflection coefficient k = (σ₁ − σ₂)/(σ₁ + σ₂). Negative k (muscle more conductive
  than skin+fat) pulls current toward the deep layer.
- Typical chronaxie by fibre type: Aα large myelinated motor 50–100 µs · Aδ ~170 µs ·
  C ≥ 400 µs. Percutaneous measurements sit at the long end of these ranges.
- Expected motor threshold, single monophasic pulses, median nerve at wrist, 3 cm circular pads:
  ~17–33 mA at 50 µs · 10–20 mA at 100 µs · 7–13 mA at 200 µs. With 5 × 5 cm pads, roughly 1.5×
  higher. If you measure far below this at 50 µs, suspect the device, not the nerve.
- Membrane time constant τ_m ≈ 0.1–1 ms — the reason the same tissue behaves completely
  differently under the kHz carriers of Block B.
