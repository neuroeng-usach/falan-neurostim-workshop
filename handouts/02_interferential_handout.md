# Interferential Current Stimulation

**Student handout — Block B** · Physics of Electrical Neurostimulation
Taller Escuela de Neurociencia (FALAN School) — Santiago, August 2026
NeuroEng@USACH · Leonel Medina, Rodrigo Osorio, Cristián Morales

Name: _______________________________   Partner(s): _______________________________

Volunteer initials: __________   Date: __________   Consent confirmed by: __________

Companion notebook: `notebooks/02_interferential_stimulation.ipynb`
Safety, exclusion criteria and consent script: `SAFETY.md` — the same rules as Block A apply.

---

## By the end of this session you will be able to

1. Explain how two medium-frequency (kHz) currents interfere to produce a low-frequency
   amplitude-modulated beat inside the tissue.
2. Identify where the beat amplitude is largest for a given montage, and set up a cross
   (quadripolar) configuration on the forearm.
3. Elicit pulsatile finger-muscle contractions with a carrier that is itself too fast for the axon
   to follow pulse by pulse.
4. Demonstrate the block of cutaneous afferents under a kHz field, using a plastic probe on the skin
   between the electrodes.
5. Contrast interferential stimulation with the single-pulse stimulation of Block A.

## Session map (70 min)

| | |
|---|---|
| Module 0 — how interferential current works (notebook Parts 1–2) | ~15 min |
| Module 1 — the field of four electrodes; where the beat is deepest (Part 3) | ~20 min |
| Module 2 — cross montage, motor threshold and pulsatile contraction | ~20 min |
| Module 3 — cutaneous afferent block with the plastic probe | ~10 min |
| Wrap-up and discussion | ~5 min |

## What you need

A two-channel interferential stimulator (CH1 and CH2 independently programmable in the kHz range) ·
four self-adhesive electrodes · conductive gel · alcohol wipes · a blunt plastic probe · skin marker
· stopwatch · a volunteer.

---

## Module 0 — How interferential current works

### Background

Two sinusoidal currents at slightly different medium frequencies — here **f1 = 4000 Hz on CH1** and
**f2 = 4002 Hz on CH2** — are injected through separate electrode pairs. Where the two fields
overlap inside the tissue they add, and the sum is a ~4 kHz carrier whose **amplitude** swells and
fades at the difference frequency Δf = |f2 − f1|. That slow envelope is the stimulus the nerve
responds to, and it is generated *inside* the tissue rather than applied at the surface.

The axon membrane behaves as a low-pass filter with a time constant of roughly 0.1–1 ms. A 4 kHz
carrier is far too fast to be followed cycle by cycle, so each individual cycle contributes almost
no net depolarisation — but the envelope is slow enough to drive the membrane.

Because the carrier is medium-frequency, skin impedance is low (Z falls with frequency), so the
stimulus is comfortable and reaches deeper than a low-frequency pulse of the same perceived
strength.

Continuous kHz current also depolarises and then desensitises the fine cutaneous afferents directly
under the electrodes — carrier-induced conduction block. That is what Module 3 tests.

### From the notebook, before you touch hardware

Run Part 2 at Δf = 100 Hz and at Δf = 2 Hz. Then run Part 3 and record the number that matters:

Modulation depth at the **geometric centre** of the montage: ______ %

Peak modulation depth on the plane, and roughly where it sits: ______ % at ______________

**Predict before running Part 3:** where will the beat be deepest? _____________________

Were you right? If not, what is wrong with the usual picture of the crossing point as the "focus"?

______________________________________________________________________________

______________________________________________________________________________

---

## Module 1 — The cross (quadripolar) montage

### Procedure — electrode placement

1. Seat the volunteer with the forearm supported, palm up, wrist and fingers relaxed and free to
   move. Clean the skin with alcohol and let it dry.
2. Identify the target: the muscle belly of the finger flexors on the volar (anterior) forearm,
   about one third of the way down from the elbow crease.
3. **CH1, parallel to the nerve:** place its two electrodes along the proximal–distal axis, one
   proximal and one distal, so its current runs along the nerve/muscle direction.
4. **CH2, perpendicular to the nerve:** place its two electrodes on the medial and lateral sides of
   the same region, so its current crosses CH1 at roughly 90°.
5. The four electrodes form a square or diamond with the two current paths crossing near the centre,
   over the finger flexors.
6. Keep at least 3–4 cm between the two electrodes *within* a channel, and press each electrode flat
   with no air gaps. Mark the centre of the cross lightly with the skin marker.
7. Have a facilitator confirm the montage before switching on.

### Task 1 — Set the stimulator

Program the following and record what your device actually displays.

| Parameter | Set value | Displayed / notes |
|---|---|---|
| CH1 carrier frequency | 4000 Hz | |
| CH2 carrier frequency | 4002 Hz | |
| Beat (difference) frequency | = ______ Hz | |
| Waveform | sinusoidal, continuous | |
| CH1 amplitude at motor threshold | ______ mA | |
| CH2 amplitude at motor threshold | ______ mA | |
| Electrode size / type | | |

**Both channels must be raised together.** The beat exists only where the two fields overlap with
comparable amplitude. If one channel is much stronger than the other, the montage degenerates into
ordinary bipolar stimulation from the stronger pair and there is no interference to speak of.

---

## Module 2 — Motor threshold and pulsatile contraction

**Goal:** bring the volunteer from no sensation, through tingling, to a visible pulsatile
finger-muscle contraction driven by the 2 Hz beat.

### Procedure

1. Start both channels at 0 mA. Ramp CH1 and CH2 together in ~1 mA steps, pausing 2–3 s at each step
   and asking the volunteer what they feel.
2. Record three landmarks: **(a) sensory threshold** — first tingling; **(b) motor threshold** —
   first visible twitch of the fingers; **(c) comfortable pulsatile contraction**.
3. Count the contractions over 10 s and divide by 10. Compare with Δf = f2 − f1.
4. Note which fingers move. The finger flexors are innervated by the median nerve, and the ulnar
   nerve for the 4th–5th digits, so the pattern tells you where the beat actually landed.
5. Detune the beat: set CH2 to 4050 Hz (Δf = 50 Hz) and try to reach motor threshold again. Record
   what happens to the *character* of the contraction.
6. Finally set CH2 to 4000 Hz (Δf = 0). Record whether any contraction persists.

| Condition | CH1 / CH2 (Hz) | Δf (Hz) | Amplitude (mA) | What you observed |
|---|---|---|---|---|
| Sensory threshold | 4000 / 4002 | 2 | | |
| Motor threshold | 4000 / 4002 | 2 | | |
| Comfortable pulsatile contraction | 4000 / 4002 | 2 | | |
| Detuned beat | 4000 / 4050 | 50 | | |
| No beat | 4000 / 4000 | 0 | | |
| Single channel only (CH1 on, CH2 off) | 4000 / — | — | | |

Measured contractions per second at Δf = 2 Hz: ______

### Question

The carrier delivers 4000 cycles per second, yet the muscle contracts only about twice per second.
Explain this in terms of membrane time constant and low-pass filtering.

______________________________________________________________________________

______________________________________________________________________________

---

## Module 3 — Blocking the cutaneous afferents

**Goal:** while the muscle is still contracting to the beat, show that light-touch sensation on the
skin inside the cross is reduced compared with skin outside it.

### Procedure

1. **Control, before switching on.** With the stimulator at 0 mA, touch the volunteer with the blunt
   plastic probe at four points and take a 0–10 rating at each.
2. The volunteer keeps their **eyes closed** for all probe testing, and the tester varies the order
   of sites so ratings are not anticipated.
3. Ramp both channels back up to the comfortable pulsatile contraction from Module 2 and hold it
   steady for about 60 s.
4. With stimulation running, repeat the probe test at A, B, C and D **in a different order**.
5. Take the current to 0 and retest all four sites immediately, then again after 2 min, to see how
   quickly sensation recovers.
6. Swap roles and repeat with a second volunteer if time allows.

| Probe site | Before (0–10) | During (0–10) | Just after (0–10) | +2 min (0–10) |
|---|---|---|---|---|
| A — centre of the cross (between electrodes) | | | | |
| B — 2 cm outside the montage, same side | | | | |
| C — dorsal forearm, same arm | | | | |
| D — opposite forearm (control) | | | | |

Was the muscle still contracting while site A felt blunted?  **Yes / No**

Block index = (rating at B during) − (rating at A during) = ______

### Question

Site A is closer to the electrodes than site B, yet the motor axons under A are being driven
vigorously. Why can the same field block small superficial sensory fibres while continuing to drive
deeper large motor axons?

______________________________________________________________________________

______________________________________________________________________________

---

## Comparison with Block A

| | Single-pulse stimulation (Block A) | Interferential stimulation |
|---|---|---|
| Frequency content | | |
| Current at motor threshold (mA) | | |
| Reported comfort (0–10) | | |
| Contraction rate (per second) | | |
| Touch sensation between electrodes | | |

## Discussion

1. Did the contraction appear at the crossing point, where the notebook says the beat is *weakest*?
   If it did, what does that tell you about the real current paths — think about the anisotropy of
   muscle, the symmetry of your montage, and the depth of the nerve relative to the plane the
   notebook plotted.
2. Rotating one channel moves where the crossing point falls. If you moved CH2 by 2 cm proximally,
   which fingers would you expect to respond, and why?
3. How would you separate a genuine kHz block of the afferents from simple habituation, or from
   distraction by the ongoing contraction? Design one extra control you could run in five minutes.

## Wrap-up questions

1. Write the beat frequency in terms of f1 and f2, and state what changes physically inside the
   tissue when Δf goes from 2 Hz to 50 Hz.
2. Why does a 4 kHz carrier feel more comfortable than a 100 Hz pulse train at the same perceived
   strength?
3. The crossing point of the two channels is often described as the place where stimulation is
   "focused". Based on what you observed and on the field model, how would you correct that
   statement?

---

## Quick reference

- Beat (difference) frequency: Δf = f2 − f1 → 4002 − 4000 = **2 Hz**.
- Effective carrier: (f1 + f2)/2 ≈ 4001 Hz; envelope amplitude ∝ 2·A·cos(π·Δf·t).
- Modulation depth m = ( |a + b| − |a − b| ) / ( |a + b| + |a − b| ), where **a** and **b** are the
  two channels' current-density vectors. **m = 1** where they are collinear and equal; **m = 0**
  where they are perpendicular and equal — which is exactly the geometric centre of a symmetric
  cross montage. The deep modulation sits in four off-centre lobes.
- Membrane low-pass filtering: τ_m ≈ 0.1–1 ms. The 125 µs half-cycle at 4 kHz is comparable to the
  chronaxie of Aα motor fibres (~50–100 µs) but delivers little net depolarisation per cycle.
- Skin impedance falls with frequency, so kHz carriers are more comfortable and penetrate deeper
  than low-frequency pulses.
- kHz conduction block: sustained medium-frequency current desensitises small superficial cutaneous
  afferents while deeper large motor axons still follow the envelope.
- Today's setting: CH1 4000 Hz, CH2 4002 Hz, sinusoidal continuous, cross (quadripolar) montage over
  the volar forearm finger flexors.
