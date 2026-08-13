"""
neurostim_model.py

A "reasonably realistic" model of transcutaneous electrical nerve stimulation,
built on real research-grade software: PyFibers (https://github.com/wmglab-duke/pyfibers),
a NEURON-Python package implementing the McIntyre-Richardson-Grill (MRG) double-cable
model of a myelinated peripheral axon -- the same model family used in published studies
of peripheral nerve stimulation, FES, and DBS thresholds.

PHYSICS
-------
Fiber: a single myelinated axon (MRG model), diameter adjustable (available discrete
diameters: 5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0 um). A large motor axon
(e.g. ulnar nerve Aalpha fiber) is well represented by ~10-14 um.

Electrode: a point current source sitting ON the skin surface (transcutaneous), above
a homogeneous tissue volume conductor with one effective (lumped) conductivity
representing the combined skin+fat+muscle path. Because current cannot flow into the
air above the skin, it is confined to the tissue half-space below, which (relative to
an infinite-medium point source) DOUBLES the potential at any given distance:

    V(r) = I / (2 * pi * sigma * r)      [surface electrode, half-space]
    V(r) = I / (4 * pi * sigma * r)      [electrode buried in bulk tissue, full-space]

sigma ~ 0.2-0.3 S/m is a reasonable lumped effective conductivity for skin+fat+muscle
in series (dermis ~0.23 S/m, muscle ~0.13-0.56 S/m depending on fiber orientation,
fat much lower ~0.02-0.2 S/m -- see docstring references at bottom of file).

Threshold search: PyFibers' ScaledStim.find_threshold() runs a bisection search on
stimulus amplitude directly on the compartmental (Hodgkin-Huxley-type, nonlinear)
model -- this is a REAL simulation of the nonlinear membrane, not the idealized
Weiss/Lapicque law. That law is only fit to the results afterward, exactly as you
would fit it to real experimental (pulse width, threshold) data.

VALIDATION (already run once, see workshop notes)
--------------------------------------------------
At diameter=10um, depth=5mm, sigma=0.25 S/m (this file's defaults):
    - Cathodic vs anodic threshold ratio ~3.9x (anodic higher) -- matches the
      classic asymmetry reported for point-source nerve stimulation.
    - Weiss-law fit to the simulated cathodic strength-duration curve gives
      chronaxie ~0.32 ms, rheobase ~0.6-0.9 mA depending on exact depth/sigma --
      chronaxie matches published human ulnar-nerve chronaxie almost exactly
      (Tsui et al. 2014, Anaesthesia 69:678-682: chronaxie 0.32 ms, SD 0.17,
      rheobase 0.91 mA, SD 0.37, awake, measured with a near-nerve needle
      stimulator -- expect a transcutaneous surface measurement, which sits
      farther from the nerve, to show a HIGHER rheobase; chronaxie should be
      comparable since it mainly reflects fiber membrane kinetics, not distance).
"""

from __future__ import annotations

import numpy as np
from pyfibers import build_fiber, FiberModel, ScaledStim, ThresholdCondition

import layered_field
import interferential_field

DEFAULT_SIGMA = 0.25       # S/m, effective bulk tissue conductivity (skin+fat+muscle, lumped)
DEFAULT_DIAMETER = 10.0    # um, representative large myelinated motor axon (ulnar nerve Aalpha)
DEFAULT_NODES = 51         # enough nodes that the fiber ends don't distort the threshold
BIPOLAR_DEFAULT_NODES = 91  # bipolar needs a longer fiber: both pads (up to ~70-80mm
                             # apart) must sit well clear of the fiber's cut ends, or
                             # the threshold search spuriously fires from an edge node
                             # instead of the intended stimulation site
DEFAULT_DEPTH_MM = 5.0     # electrode-to-nerve depth (mm), adjust to match anatomy/measurement


def build_axon(diameter: float = DEFAULT_DIAMETER, n_nodes: int = DEFAULT_NODES):
    """Build an MRG myelinated axon of the given diameter (um)."""
    return build_fiber(FiberModel.MRG_INTERPOLATION, diameter=diameter, n_nodes=n_nodes)


def set_electrode(fiber, depth_mm: float = DEFAULT_DEPTH_MM, sigma: float = DEFAULT_SIGMA,
                   position_frac: float = 0.5, half_space: bool = True):
    """Place a point-source transcutaneous electrode above the fiber and store the
    resulting extracellular potential profile (mV, per 1 mA reference current) on
    fiber.potentials.

    depth_mm: perpendicular distance from the skin electrode to the nerve fiber.
    position_frac: where along the fiber the electrode is centered (0.5 = midpoint).
    half_space: if True, apply the x2 surface-electrode correction (recommended for
        a transcutaneous electrode on skin); if False, treat it as buried in bulk tissue.
    """
    z0 = fiber.length * position_frac
    y0 = depth_mm * 1000.0  # mm -> um
    pot = fiber.point_source_potentials(0.0, y0, z0, 1.0, sigma, inplace=True)
    if half_space:
        pot = pot * 2.0
        fiber.potentials = pot
    return pot


def set_electrode_layered(fiber, fiber_depth_mm: float,
                           sigma1: float = layered_field.DEFAULT_SIGMA1,
                           sigma2: float = layered_field.DEFAULT_SIGMA2,
                           h_mm: float = layered_field.DEFAULT_H_MM,
                           position_frac: float = 0.5,
                           electrode_radius_mm: float = 0.0):
    """Place the electrode using the explicit two-layer volume-conductor model
    from layered_field.py (skin+fat over muscle), instead of the single
    homogeneous half-space used by set_electrode(). This is the same field
    students explore in Module 0 -- using it here means the NEURON fiber is
    driven by literally the same, visible physics, not a second hidden formula.

    fiber_depth_mm: depth of the (horizontal) fiber below the skin surface --
        must be > h_mm for the fiber to sit in the muscle layer, which is the
        physiologically relevant case for a peripheral nerve.
    position_frac: fraction along the fiber directly under the electrode.
    electrode_radius_mm: 0 = point source; >0 models a disc electrode of that
        radius (same superposition as layered_field.potential()).
    """
    z0 = fiber.length * position_frac  # electrode's position along the fiber's own axis (um)
    x_offset_mm = (fiber.coordinates[:, 2] - z0) / 1000.0
    V = layered_field.potential(x_offset_mm, -fiber_depth_mm, i0_mA=1.0,
                                 sigma1=sigma1, sigma2=sigma2, h_mm=h_mm,
                                 electrode_radius_mm=electrode_radius_mm)
    fiber.potentials = V
    return V


def set_electrode_bipolar_layered(fiber, fiber_depth_mm: float, separation_mm: float,
                                   sigma1: float = layered_field.DEFAULT_SIGMA1,
                                   sigma2: float = layered_field.DEFAULT_SIGMA2,
                                   h_mm: float = layered_field.DEFAULT_H_MM,
                                   position_frac: float = 0.5,
                                   electrode_radius_mm: float = 0.0):
    """Like set_electrode_layered, but places TWO electrodes -- a working pad
    A and a return pad B, `separation_mm` apart along the fiber, centered at
    position_frac -- instead of one, using layered_field.bipolar_potential().
    This matches how a real TENS unit (e.g. the Ultima Neo used in the lab
    session) actually operates: two pads, no separate ground, current flows
    between them.

    The reference field is built with pad A as source (+1 mA) and pad B as
    sink (-1 mA) when scaled by amplitude=+1; the existing cathodic/anodic
    sign convention in find_threshold_current/run_single_stim (sign=-1 for
    'cathodic') then determines which pad acts as the cathode at any given
    amplitude -- pad A simply takes over the role "the electrode" played in
    the single-electrode model, exactly as in set_electrode_layered.
    """
    z0 = fiber.length * position_frac  # um, along the fiber's own axis
    x_offset_mm = (fiber.coordinates[:, 2] - z0) / 1000.0
    V = layered_field.bipolar_potential(x_offset_mm, -fiber_depth_mm, separation_mm,
                                         i0_mA=1.0, sigma1=sigma1, sigma2=sigma2, h_mm=h_mm,
                                         electrode_radius_mm=electrode_radius_mm)
    fiber.potentials = V
    return V


def set_electrode_interferential(fiber, fiber_depth_mm: float, square_mm: float = 80.0,
                                  sigma1: float = layered_field.DEFAULT_SIGMA1,
                                  sigma2: float = layered_field.DEFAULT_SIGMA2,
                                  h_mm: float = layered_field.DEFAULT_H_MM,
                                  y_offset_mm: float = 0.0, position_frac: float = 0.5,
                                  electrode_radius_mm: float = 0.0):
    """Set fiber.potentials to the TWO channel profiles of a quadripolar
    interferential montage (see interferential_field.electrode_positions), one
    row per channel, so the fiber can be driven by two independent carriers with
    run_interferential(). The straight fiber runs along its own axis at depth
    fiber_depth_mm and lateral offset y_offset_mm inside the four-electrode
    square (side square_mm). Returns (Va, Vb) in mV per 1 mA reference current
    per channel -- the same layered_field two-layer physics used everywhere else.
    """
    z0 = fiber.length * position_frac
    x_mm = (fiber.coordinates[:, 2] - z0) / 1000.0  # fiber axis -> montage x (mm), centred
    els = interferential_field.electrode_positions(square_mm)
    Va = interferential_field.channel_potential(x_mm, y_offset_mm, fiber_depth_mm, els['A'],
                                                 i0_mA=1.0, sigma1=sigma1, sigma2=sigma2,
                                                 h_mm=h_mm, electrode_radius_mm=electrode_radius_mm)
    Vb = interferential_field.channel_potential(x_mm, y_offset_mm, fiber_depth_mm, els['B'],
                                                 i0_mA=1.0, sigma1=sigma1, sigma2=sigma2,
                                                 h_mm=h_mm, electrode_radius_mm=electrode_radius_mm)
    fiber.potentials = np.vstack([Va, Vb])
    return Va, Vb


def run_interferential(fiber, amp_mA: float, f1_hz: float, f2_hz: float,
                        tstop_ms: float = 40.0, dt: float = 0.005,
                        ampA_mA: float = None, ampB_mA: float = None):
    """Drive an interferential-montage fiber (potentials set by
    set_electrode_interferential) with channel A at f1_hz and channel B at
    f2_hz, each a unit cosine scaled by ampA_mA / ampB_mA (default: both amp_mA).
    Records Vm at all nodes. Returns (t_ms, vm [n_nodes x n_t], n_ap).

    The two carriers are applied as two INDEPENDENT PyFibers sources (one
    waveform per channel potential set -- see ScaledStim's multi-source
    contract), so the extracellular potential at each compartment is
        ampA * Va * cos(2*pi*f1*t) + ampB * Vb * cos(2*pi*f2*t),
    the true interferential superposition mixing in the tissue, not a single
    premodulated waveform. fail_on_end_excitation is disabled because a
    sustained oscillating field legitimately excites the whole fiber, not just
    a central node.
    """
    wfA = lambda t: np.cos(2 * np.pi * f1_hz * t * 1e-3)
    wfB = lambda t: np.cos(2 * np.pi * f2_hz * t * 1e-3)
    a = amp_mA if ampA_mA is None else ampA_mA
    b = amp_mA if ampB_mA is None else ampB_mA
    fiber.record_vm()
    stim = ScaledStim(waveform=[wfA, wfB], dt=dt, tstop=tstop_ms)
    n_ap, _ = stim.run_sim([a, b], fiber, fail_on_end_excitation=None)
    t = np.array(fiber.time)
    vm = np.array([np.array(v) for v in fiber.vm])
    return t, vm, int(n_ap)


def _rect_waveform(pulse_width_ms: float):
    def wf(t):
        return 1.0 if 0 <= t <= pulse_width_ms else 0.0
    return wf


def find_threshold_current(fiber, pulse_width_ms: float, polarity: str = "cathodic",
                            dt: float = 0.005, max_iterations: int = 12,
                            termination_tolerance: float = 5.0,
                            initial_top_mag: float = 40.0, shrink_retries: int = 5):
    """Bisection threshold search for one rectangular pulse. Returns (signed threshold
    in mA, info) -- negative = cathodic convention. polarity must be 'cathodic' or 'anodic'.

    NOTE ON BOUNDS: if the upper search bound is too large relative to the true
    threshold, long/moderate pulses can trigger depolarization block (the fiber
    fails to fire a clean propagating spike at very high sustained current -- a
    real nonlinear membrane phenomenon, not a bug), which PyFibers reports as a
    "sub-threshold" bounds error. To stay clear of that regime automatically,
    this function starts from initial_top_mag and, if the search fails to find
    a valid bracket, halves the top magnitude and retries (up to shrink_retries
    times) -- equivalent to a physiologist backing off stimulator output when a
    contraction looks off rather than blindly cranking current higher.
    """
    if polarity == "cathodic":
        sign = -1.0
    elif polarity == "anodic":
        sign = 1.0
    else:
        raise ValueError("polarity must be 'cathodic' or 'anodic'")

    top_mag = initial_top_mag
    last_err = None
    for _ in range(shrink_retries):
        stim = ScaledStim(waveform=_rect_waveform(pulse_width_ms), dt=dt, tstop=pulse_width_ms + 8)
        try:
            thresh, info = stim.find_threshold(
                fiber, condition=ThresholdCondition.ACTIVATION,
                stimamp_top=sign * top_mag, stimamp_bottom=sign * 0.01,
                max_iterations=max_iterations, termination_tolerance=termination_tolerance,
            )
            return thresh, info
        except RuntimeError as e:
            last_err = e
            top_mag *= 0.5
    raise RuntimeError(
        f"Could not bracket a threshold for pulse_width={pulse_width_ms} ms after "
        f"{shrink_retries} attempts (last top magnitude tried: {top_mag*2}). "
        f"Last error: {last_err}"
    )


def run_single_stim(fiber, amplitude_mA: float, pulse_width_ms: float, dt: float = 0.005):
    """Run one simulation at a fixed signed amplitude (mA). Records Vm at every node.
    Returns (time_ms: np.ndarray, vm_by_node: np.ndarray [n_nodes x n_timepoints], fired: bool).
    """
    fiber.record_vm()
    stim = ScaledStim(waveform=_rect_waveform(pulse_width_ms), dt=dt, tstop=pulse_width_ms + 8)
    n_ap, _ = stim.run_sim(amplitude_mA, fiber)
    t = np.array(fiber.time)
    vm = np.array([np.array(v) for v in fiber.vm])
    return t, vm, bool(n_ap and n_ap > 0)


def strength_duration_sweep(diameter: float, depth_mm: float, sigma: float,
                             pulse_widths_ms, polarity: str = "cathodic",
                             n_nodes: int = DEFAULT_NODES, half_space: bool = True):
    """Run a full strength-duration sweep using the simple homogeneous half-space
    field. Returns list of (pulse_width_ms, |threshold_mA|). Rebuilds a fresh
    fiber for each point (safest -- avoids state leakage between runs).
    """
    results = []
    for pw in pulse_widths_ms:
        fiber = build_axon(diameter=diameter, n_nodes=n_nodes)
        set_electrode(fiber, depth_mm=depth_mm, sigma=sigma, half_space=half_space)
        thresh, _ = find_threshold_current(fiber, pw, polarity=polarity)
        results.append((pw, abs(thresh)))
    return results


def strength_duration_sweep_layered(diameter: float, fiber_depth_mm: float,
                                     sigma1: float, sigma2: float, h_mm: float,
                                     pulse_widths_ms, polarity: str = "cathodic",
                                     n_nodes: int = DEFAULT_NODES,
                                     electrode_radius_mm: float = 0.0):
    """Same as strength_duration_sweep, but driven by the explicit two-layer
    volume-conductor field (layered_field.py / set_electrode_layered) instead
    of the homogeneous half-space -- this is the version that connects directly
    to whatever sigma1/sigma2/h_mm/electrode_radius_mm students set in the
    Module 0 field explorer.
    """
    results = []
    for pw in pulse_widths_ms:
        fiber = build_axon(diameter=diameter, n_nodes=n_nodes)
        set_electrode_layered(fiber, fiber_depth_mm=fiber_depth_mm,
                               sigma1=sigma1, sigma2=sigma2, h_mm=h_mm,
                               electrode_radius_mm=electrode_radius_mm)
        thresh, _ = find_threshold_current(fiber, pw, polarity=polarity)
        results.append((pw, abs(thresh)))
    return results


def strength_duration_sweep_bipolar_layered(diameter: float, fiber_depth_mm: float,
                                             separation_mm: float,
                                             sigma1: float, sigma2: float, h_mm: float,
                                             pulse_widths_ms, polarity: str = "cathodic",
                                             n_nodes: int = DEFAULT_NODES,
                                             electrode_radius_mm: float = 0.0):
    """Same as strength_duration_sweep_layered, but driven by the bipolar
    (two-electrode) field via set_electrode_bipolar_layered() instead of a
    single electrode -- the version that matches an actual TENS pad pair."""
    results = []
    for pw in pulse_widths_ms:
        fiber = build_axon(diameter=diameter, n_nodes=n_nodes)
        set_electrode_bipolar_layered(fiber, fiber_depth_mm=fiber_depth_mm,
                                       separation_mm=separation_mm,
                                       sigma1=sigma1, sigma2=sigma2, h_mm=h_mm,
                                       electrode_radius_mm=electrode_radius_mm)
        thresh, _ = find_threshold_current(fiber, pw, polarity=polarity)
        results.append((pw, abs(thresh)))
    return results


# --- Strength-duration law & published reference values ---------------------------
# Defined in `weiss.py` (pure NumPy, no NEURON) and re-exported here so that
# `nm.fit_weiss(...)` / `nm.LITERATURE_ULNAR_*` keep working unchanged, while a
# student without NEURON installed can still fit their own measured curve.
from weiss import (  # noqa: E402,F401
    fit_weiss,
    LITERATURE_ULNAR_RHEOBASE_MA,
    LITERATURE_ULNAR_CHRONAXIE_MS,
)
