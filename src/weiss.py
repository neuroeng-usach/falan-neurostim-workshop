"""
weiss.py

The Weiss/Lapicque strength-duration law and the published reference values the
workshop compares against. Kept separate from `neurostim_model.py` on purpose:
fitting a strength-duration curve is arithmetic, not a NEURON simulation, so a
student who has measured thresholds on their own forearm can fit and plot them
even on a machine where NEURON/PyFibers was never installed.

`neurostim_model` re-exports everything here, so `nm.fit_weiss(...)` and
`nm.LITERATURE_ULNAR_RHEOBASE_MA` keep working unchanged.

    I(t) = I_rheobase * (1 + chronaxie / t)

Rheobase is the asymptotic threshold current for an infinitely long pulse;
chronaxie is the pulse width at which threshold is exactly twice rheobase, and
is a property of the membrane's time constant rather than of how far the
electrode sits from the nerve -- which is why, in this workshop, chronaxie is
the robust prediction and rheobase is the sensitive one.
"""

from __future__ import annotations

import numpy as np

# Tsui et al. 2014, Anaesthesia 69(7):678-682, "The effects of general anaesthesia
# on nerve-motor response characteristics (rheobase and chronaxie) to peripheral
# nerve stimulation" -- ulnar nerve, near-nerve (needle) stimulation. The values
# below are the AWAKE (pre-induction) means; the paper reports standard
# deviations, not confidence intervals.
LITERATURE_ULNAR_RHEOBASE_MA = 0.91     # SD 0.37 (awake); 1.11, SD 0.53 under GA
LITERATURE_ULNAR_CHRONAXIE_MS = 0.32    # SD 0.17 (awake); 0.29, SD 0.13 under GA
# That paper is itself a nice independent confirmation of this workshop's central
# claim: general anaesthesia raised rheobase by ~20% (p = 0.05) while leaving
# chronaxie unchanged (p = 0.39). Rheobase is the sensitive parameter, chronaxie
# the robust one.
# Typical textbook chronaxie ranges by fiber type (order of magnitude, various sources):
#   Aalpha (large myelinated motor):   50-100 us
#   Adelta (small myelinated):         ~170 us
#   C (unmyelinated):                  >=400 us

# A transcutaneous surface measurement sits FARTHER from the nerve than the
# near-nerve needle of Tsui et al., so expect a higher rheobase in the lab;
# chronaxie should be comparable, since it mainly reflects membrane kinetics.


def fit_weiss(points):
    """Fit the Weiss/Lapicque law I(t) = Irheobase * (1 + chronaxie / t) to
    (pulse_width_ms, threshold_mA) points via the standard linearization
    I = Irh + Irh*chronaxie*(1/t), i.e. linear regression of I against 1/t.
    Returns dict(rheobase_mA=..., chronaxie_ms=...) or None if <2 points given.
    """
    pts = [(t, i) for t, i in points if t > 0 and i > 0]
    if len(pts) < 2:
        return None
    x = np.array([1.0 / t for t, _ in pts])
    y = np.array([i for _, i in pts])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    if intercept <= 0:
        return None
    return {"rheobase_mA": float(intercept), "chronaxie_ms": float(slope / intercept)}
