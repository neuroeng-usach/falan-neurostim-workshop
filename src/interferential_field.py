"""
interferential_field.py

The electric field and current distribution of an INTERFERENTIAL CURRENT (IFC)
electrode arrangement -- four surface electrodes wired as two independent
channels ("quadripolar"), each channel a bipolar pad pair driven by a slightly
different medium-frequency carrier.

This module deliberately builds NOTHING new at the physics level: it reuses the
exact, independently-validated two-layer volume-conductor point-source solution
from `layered_field.py` (skin+fat over muscle, method of images, full image
series -- see that file's docstring and `_self_test()`), and only adds

  1. a 3-D electrode placement on the skin surface (electrodes at arbitrary
     (x, y), not just along a single line), so the four pads can be arranged in
     the crossed square that makes IFC "interferential", and
  2. the interference / amplitude-modulation (beat) arithmetic that turns the
     two channels' static current-density fields into the low-frequency beat a
     nerve actually responds to.

So every potential value here is still a sum of `layered_field._point_potential`
calls: the same field students explore in the transcutaneous-stimulation
workshop's Module 0, just with four electrodes instead of two.

WHY A MEDIUM-FREQUENCY CARRIER, AND WHY TWO OF THEM
---------------------------------------------------
Skin impedance falls roughly as 1/f, so a ~4 kHz carrier crosses the skin far
more comfortably (lower voltage, less sensory "bite") than a directly-applied
low-frequency pulse train would. But a nerve's membrane is a low-pass filter --
it barely follows a 4 kHz carrier. IFC's trick: drive channel A at f1 (e.g.
4000 Hz) and channel B at f2 = f1 + df (e.g. 4100 Hz). Where the two channels'
currents overlap inside the tissue, they add, and the sum is a 4 kHz-ish carrier
whose AMPLITUDE swells and fades at the beat frequency df = |f1 - f2| (here
100 Hz). That slow beat is what the nerve demodulates and responds to -- a
low-frequency stimulus delivered through a skin-friendly high-frequency carrier,
generated *inside* the tissue rather than applied at the surface.

THE BEAT / AMPLITUDE-MODULATION MATH
------------------------------------
At any interior point the two channels produce static current-density VECTORS
`a` (channel A, oscillating at f1) and `b` (channel B, at f2). The instantaneous
current density is

    J(t) = a cos(2*pi*f1*t) + b cos(2*pi*f2*t).

Near a "beat peak" the two cosines are momentarily in phase (both sweep -1..1
together), so J ~= (a + b) cos(2*pi*f_carrier*t) and the fast carrier's peak
amplitude is |a + b|. One half-beat later they are in antiphase, J ~= (a - b)
cos(...), and the carrier amplitude is |a - b|. So the fast carrier's amplitude
is itself modulated between the two extremes |a + b| (channels in phase) and
|a - b| (antiphase). Which of the two is the larger depends on the angle between
`a` and `b`, so the beat's peak and trough are their max and min, and the depth
of the beat modulation is

    m = | |a + b| - |a - b| | / ( |a + b| + |a - b| )      (0 .. 1).

The non-obvious, clinically important consequence: m = 1 (100% modulation)
wherever `a` and `b` are COLLINEAR and equal in magnitude (one of |a +- b| is
zero), and m = 0 where they are perpendicular and equal (|a + b| = |a - b|). In the crossed four-electrode
layout the two channel currents are ~perpendicular right at the geometric
centre -- so the modulation there is WEAK, and the four strong-modulation lobes
sit off-axis, giving the classic IFC "four-leaf clover" pattern. (Honest
caveat: the full vector envelope has finer sub-beat structure where the two
currents cross near 90 deg; the |a+b| / |a-b| pair used here is the beat-rate
amplitude modulation, which is the standard IFC description and what the nerve's
low-pass membrane effectively tracks.)
"""

from __future__ import annotations

import numpy as np

import layered_field as lf

# --- Defaults ---------------------------------------------------------------
DEFAULT_CARRIER_HZ = 4000.0   # channel A carrier frequency
DEFAULT_BEAT_HZ = 100.0       # f2 - f1, the interferential beat frequency
DEFAULT_SQUARE_MM = 80.0      # side of the square the four pads sit on
DEFAULT_PLANE_DEPTH_MM = 15.0 # depth of the horizontal plane we visualise

# Reuse the workshop's shared palette so every figure reads as one system.
COLOR_CH_A = lf.COLOR_DEPOLARIZING     # red   -- channel A
COLOR_CH_B = lf.COLOR_HYPERPOLARIZING  # blue  -- channel B
COLOR_INK = lf.COLOR_INK
COLOR_MOD_CMAP = ["#fcfcfb", "#ffe6c7", "#f7b267", "#eb6834", "#c0392b", "#7d1d13"]  # warm, for modulation depth


def electrode_positions(square_mm: float = DEFAULT_SQUARE_MM):
    """Four electrodes at the corners of a square centred on the origin, wired
    as two crossed diagonal channels (standard quadripolar IFC):

        A+ (top-left) ........ B+ (top-right)
             .   channel A on one diagonal (A+ -> A-)
             .   channel B on the other  (B+ -> B-)
        B- (bottom-left) ..... A- (bottom-right)

    Returns dict {'A': [(x, y, sign), ...], 'B': [...]} with sign = +1 for the
    source pad, -1 for the sink pad (mm).
    """
    h = square_mm / 2.0
    return {
        'A': [(-h, +h, +1.0), (+h, -h, -1.0)],   # top-left source  -> bottom-right sink
        'B': [(+h, +h, +1.0), (-h, -h, -1.0)],   # top-right source -> bottom-left sink
    }


def _electrode_potential(x_mm, y_mm, z_depth_mm, xe_mm, ye_mm, i0_mA,
                          sigma1, sigma2, h_mm, electrode_radius_mm=0.0, n_terms=60):
    """Potential (mV) on the horizontal plane at depth z_depth_mm (>0, below the
    skin) from ONE electrode centred on the skin at (xe_mm, ye_mm), using the
    layered_field two-layer point-source solution as the building block.

    electrode_radius_mm > 0 models a disc pad as a superposition of point
    sources spread over the disc (sunflower spiral, same construction as
    layered_field.potential's disc path), done here in the true (x, y) surface
    plane so it works for an electrode placed anywhere, not just at the origin.
    """
    x = np.asarray(x_mm, dtype=float)
    y = np.asarray(y_mm, dtype=float)
    z = -abs(float(z_depth_mm))

    if electrode_radius_mm <= 0.0:
        r = np.sqrt((x - xe_mm) ** 2 + (y - ye_mm) ** 2)
        return lf._point_potential(r, z, i0_mA, sigma1, sigma2, h_mm, n_terms)

    N = 100
    idx = np.arange(0, N, dtype=float) + 0.5
    rr = electrode_radius_mm * np.sqrt(idx / N)
    th = np.pi * (1 + 5 ** 0.5) * idx
    xs = xe_mm + rr * np.cos(th)
    ys = ye_mm + rr * np.sin(th)
    di = i0_mA / N
    V = np.zeros_like(x + y, dtype=float)
    for i in range(N):
        r = np.sqrt((x - xs[i]) ** 2 + (y - ys[i]) ** 2)
        V = V + lf._point_potential(r, z, di, sigma1, sigma2, h_mm, n_terms)
    return V


def channel_potential(x_mm, y_mm, z_depth_mm, channel_electrodes, i0_mA=1.0,
                       sigma1=lf.DEFAULT_SIGMA1, sigma2=lf.DEFAULT_SIGMA2,
                       h_mm=lf.DEFAULT_H_MM, electrode_radius_mm=0.0, n_terms=60):
    """Potential (mV) of one channel = source pad + sink pad, by superposition."""
    V = np.zeros_like(np.asarray(x_mm, dtype=float) + np.asarray(y_mm, dtype=float))
    for (xe, ye, sgn) in channel_electrodes:
        V = V + _electrode_potential(x_mm, y_mm, z_depth_mm, xe, ye, sgn * i0_mA,
                                     sigma1, sigma2, h_mm, electrode_radius_mm, n_terms)
    return V


def field_grid(x_range_mm=(-60, 60), y_range_mm=(-60, 60), n=121,
               z_depth_mm=DEFAULT_PLANE_DEPTH_MM, square_mm=DEFAULT_SQUARE_MM,
               i0_mA=1.0, sigma1=lf.DEFAULT_SIGMA1, sigma2=lf.DEFAULT_SIGMA2,
               h_mm=lf.DEFAULT_H_MM, electrode_radius_mm=0.0):
    """Build both channels' potential and in-plane current-density (Jx, Jy) on a
    horizontal (x, y) plane at depth z_depth_mm below the skin.

    The in-plane current density J = sigma * E, E = -grad(V), uses the local
    conductivity at that depth (sigma1 if the plane is still in the shallow
    layer, sigma2 if it has reached the muscle). We visualise the IN-PLANE
    (tangential) current -- the component that would drive a fibre lying in
    this plane -- exactly as layered_field does for its vertical section.
    """
    x = np.linspace(*x_range_mm, n)
    y = np.linspace(*y_range_mm, n)
    X, Y = np.meshgrid(x, y)
    els = electrode_positions(square_mm)
    sig = float(np.where(abs(z_depth_mm) <= h_mm, sigma1, sigma2))
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    grids = {}
    for ch in ('A', 'B'):
        V = channel_potential(X, Y, z_depth_mm, els[ch], i0_mA=i0_mA,
                              sigma1=sigma1, sigma2=sigma2, h_mm=h_mm,
                              electrode_radius_mm=electrode_radius_mm)
        Ex = -np.gradient(V, dx, axis=1)   # mV/mm ~ V/m
        Ey = -np.gradient(V, dy, axis=0)
        grids[ch] = {"V": V, "Ex": Ex, "Ey": Ey, "Jx": sig * Ex, "Jy": sig * Ey}

    return {"x": x, "y": y, "X": X, "Y": Y, "grids": grids, "sigma": sig,
            "electrodes": els, "square_mm": square_mm, "z_depth_mm": abs(z_depth_mm)}


def vertical_field_grid(s_range_mm=(-65, 65), depth_range_mm=(0.0, 40.0), n_s=141, n_z=90,
                        cut_angle_deg=0.0, square_mm=DEFAULT_SQUARE_MM, i0_mA=1.0,
                        sigma1=lf.DEFAULT_SIGMA1, sigma2=lf.DEFAULT_SIGMA2,
                        h_mm=lf.DEFAULT_H_MM):
    """Both channels' potential and in-plane current density on a VERTICAL plane
    (a depth cross-section) through the montage centre -- the "perpendicular
    view" that shows how far the field and the beat penetrate into the tissue.

    The plane's horizontal axis is the montage direction at ``cut_angle_deg``
    (0 = along the montage x-axis, through the two x-axis modulation lobes);
    depth runs from the skin (z = 0) down to -depth_range_mm[1]. A point at
    signed horizontal distance s and depth z corresponds to montage coordinates
    (s*cos, s*sin, z). The two in-plane current-density components are stored as
    ``Jx`` (horizontal, along the cut) and ``Jy`` (vertical, into depth) so that
    :func:`interference_maps` can be reused unchanged to get the beat modulation
    in this plane.

    The beat modulation on this plane is a genuinely 3-D quantity (the clover
    null is where the two current vectors are perpendicular *in 3-D*), so this
    function also computes the out-of-plane horizontal current component (by a
    finite difference perpendicular to the cut) and stores the full montage-frame
    vector (``Jx``, ``Jy``, ``Jz``) for :func:`interference_maps`, plus the
    in-plane pair (``Jalong`` horizontal, ``Jdepth`` vertical) for streamlines.

    Point-source electrodes are used here (the disc superposition is a
    horizontal-view refinement); the penetration/modulation picture is
    unchanged by pad radius at these depths.
    """
    s = np.linspace(*s_range_mm, n_s)
    z = np.linspace(-abs(depth_range_mm[1]), -abs(depth_range_mm[0]), n_z)  # depth, 0 at top
    S, Z = np.meshgrid(s, z)
    th = np.deg2rad(cut_angle_deg)
    cx, cy = np.cos(th), np.sin(th)     # along-cut horizontal unit vector
    px, py = -np.sin(th), np.cos(th)    # perpendicular horizontal unit vector
    Xp = S * cx
    Yp = S * cy
    els = electrode_positions(square_mm)
    ds = s[1] - s[0]
    dz = z[1] - z[0]
    dp = 0.5  # mm, perpendicular step for the out-of-plane derivative
    sig = np.where(Z >= -h_mm, sigma1, sigma2)

    def channel_V(Xc, Yc, ch):
        V = np.zeros_like(S)
        for (xe, ye, sgn) in els[ch]:
            rr = np.sqrt((Xc - xe) ** 2 + (Yc - ye) ** 2)
            V = V + lf._point_potential(rr, Z, sgn * i0_mA, sigma1, sigma2, h_mm)
        return V

    grids = {}
    for ch in ('A', 'B'):
        V = channel_V(Xp, Yp, ch)
        Vp = channel_V(Xp + px * dp, Yp + py * dp, ch)   # shifted perpendicular +
        Vm = channel_V(Xp - px * dp, Yp - py * dp, ch)   # shifted perpendicular -
        Es = -np.gradient(V, ds, axis=1)     # along-cut horizontal field
        Ez = -np.gradient(V, dz, axis=0)     # vertical (depth) field
        Eperp = -(Vp - Vm) / (2 * dp)        # out-of-plane horizontal field
        Ex = Es * cx + Eperp * px            # montage-frame horizontal components
        Ey = Es * cy + Eperp * py
        grids[ch] = {"V": V, "Jx": sig * Ex, "Jy": sig * Ey, "Jz": sig * Ez,
                     "Jalong": sig * Es, "Jdepth": sig * Ez}

    return {"s": s, "z": z, "S": S, "Z": Z, "grids": grids, "electrodes": els,
            "square_mm": square_mm, "h_mm": h_mm, "cut_angle_deg": cut_angle_deg,
            "sigma1": sigma1, "sigma2": sigma2}


def interference_maps(g):
    """From a field_grid() result, compute the interferential beat quantities at
    every point:

        env_max = |a + b|   (fast-carrier peak when the two channels are in phase)
        env_min = |a - b|   (           ... when they are in antiphase)
        amp     = (env_max - env_min) / 2   -- the size of the low-frequency beat
        depth   = (env_max - env_min) / (env_max + env_min)  -- modulation depth 0..1

    where a = channel A current-density vector, b = channel B's. Also returns
    each channel's current-density magnitude for reference.
    """
    A, B = g["grids"]['A'], g["grids"]['B']
    ax, ay, az = A["Jx"], A["Jy"], A.get("Jz", 0.0)
    bx, by, bz = B["Jx"], B["Jy"], B.get("Jz", 0.0)

    magA = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    magB = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    # The fast carrier's amplitude takes the two extreme values |a+b| (channels
    # in phase) and |a-b| (antiphase) over one beat; which of the two is the
    # larger depends on the angle between a and b, so the beat's peak/trough are
    # their max/min, and the modulation depth uses their absolute difference.
    # (az/bz are 0 for the horizontal field_grid -- a 2-D in-plane clover -- and
    # non-zero for vertical_field_grid, where the true 3-D vectors matter.)
    p = np.sqrt((ax + bx) ** 2 + (ay + by) ** 2 + (az + bz) ** 2)
    q = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
    env_max = np.maximum(p, q)
    env_min = np.minimum(p, q)
    amp = 0.5 * (env_max - env_min)
    denom = env_max + env_min
    depth = np.where(denom > 0, (env_max - env_min) / np.where(denom == 0, 1.0, denom), 0.0)

    return {"magA": magA, "magB": magB, "env_max": env_max, "env_min": env_min,
            "amp": amp, "depth": depth}


def beat_waveform(t_ms, f1_hz=DEFAULT_CARRIER_HZ, f2_hz=None, ampA=1.0, ampB=1.0):
    """The scalar interferential beat at a single point where the two channels
    contribute amplitudes ampA (at f1) and ampB (at f2): returns the summed
    signal and its slow envelope. If f2_hz is None, f2 = f1 + DEFAULT_BEAT_HZ.

    (Scalar/collinear version -- used for the time-domain "what is a beat"
    demonstration; the spatial maps above use the full vector a/b.)
    """
    if f2_hz is None:
        f2_hz = f1_hz + DEFAULT_BEAT_HZ
    t = np.asarray(t_ms, dtype=float) * 1e-3  # ms -> s
    s1 = ampA * np.cos(2 * np.pi * f1_hz * t)
    s2 = ampB * np.cos(2 * np.pi * f2_hz * t)
    s = s1 + s2
    # Envelope of the sum of two cosines: swings between |A+B| and |A-B| at the
    # beat rate. |A cos w1 t + B cos w2 t| upper envelope:
    df = 2 * np.pi * (f2_hz - f1_hz)
    env = np.sqrt(ampA ** 2 + ampB ** 2 + 2 * ampA * ampB * np.cos(df * t))
    return s, env, f2_hz


# ---------------------------------------------------------------------------
def _self_test():
    """Sanity checks. The 2-layer physics itself is already validated in
    layered_field._self_test(); here we only check the pieces this module adds:
    the 3-D placement reduces to layered_field, single-channel antisymmetry, and
    the beat/modulation identities."""

    # 1) Reduction to layered_field: one electrode at the origin, evaluated at
    #    horizontal distance r on a plane at depth d, must equal
    #    layered_field.potential(r, -d) exactly (same underlying call).
    d = 10.0
    r = 7.0
    v_here = _electrode_potential(r, 0.0, d, 0.0, 0.0, 1.0,
                                  lf.DEFAULT_SIGMA1, lf.DEFAULT_SIGMA2, lf.DEFAULT_H_MM)
    v_lf = lf.potential(r, -d, i0_mA=1.0)
    assert abs(v_here - v_lf) < 1e-9, (v_here, v_lf)

    # 2) Single-channel antisymmetry: channel A is a +source / -sink pair placed
    #    antisymmetrically about the origin, so V_A(x, y) = -V_A(-x, -y).
    els = electrode_positions(80.0)
    p = channel_potential(11.0, 6.0, 12.0, els['A'])
    m = channel_potential(-11.0, -6.0, 12.0, els['A'])
    assert abs(p + m) < 1e-6 * max(abs(p), 1.0), (p, m)

    # 3) Beat/modulation identities on synthetic current vectors:
    def _mod(ax, ay, bx, by):
        p = np.hypot(ax + bx, ay + by)
        q = np.hypot(ax - bx, ay - by)
        return abs(p - q) / (p + q)
    #   parallel & equal  -> 100% modulation
    assert abs(_mod(1, 0, 1, 0) - 1.0) < 1e-12
    #   perpendicular & equal -> 0% modulation
    assert abs(_mod(1, 0, 0, 1) - 0.0) < 1e-12
    #   antiparallel & equal -> also 100% (|a+b|=0 vs |a-b|=2)
    assert abs(_mod(1, 0, -1, 0) - 1.0) < 1e-12

    # 4) Beat waveform: envelope must peak at |A+B| and trough at |A-B|.
    t = np.linspace(0, 20, 20000)  # ms; a few 100 Hz beats
    _, env, _ = beat_waveform(t, ampA=1.0, ampB=0.6)
    assert abs(env.max() - 1.6) < 1e-3, env.max()
    assert abs(env.min() - 0.4) < 1e-3, env.min()

    # 5) Clover-leaf sanity: with the crossed layout the modulation depth at the
    #    exact centre must be LOWER than at an off-axis point on a diagonal
    #    bisector -- the whole point of IFC steering.
    g = field_grid(n=61)
    im = interference_maps(g)
    ci = len(g['x']) // 2  # centre index
    centre_depth = im['depth'][ci, ci]
    # a point off-centre along the +x axis (a modulation lobe direction here)
    xi = np.argmin(np.abs(g['x'] - 25.0))
    lobe_depth = im['depth'][ci, xi]
    assert lobe_depth > centre_depth, (centre_depth, lobe_depth)

    # 6) Vertical (perpendicular) section: along the central column (s=0, i.e.
    #    the montage centre) the two channel currents stay perpendicular-and-
    #    equal at every depth, so the modulation must be ~0 all the way down --
    #    the central null penetrates. Off-axis it must exceed that.
    gv = vertical_field_grid(n_s=61, n_z=41)
    imv = interference_maps(gv)
    sc = len(gv['s']) // 2                     # central column index
    col_depth = imv['depth'][:, sc]
    assert np.nanmax(col_depth) < 0.05, np.nanmax(col_depth)
    si = np.argmin(np.abs(gv['s'] - 25.0))     # off-axis column
    assert np.nanmax(imv['depth'][:, si]) > np.nanmax(col_depth)

    print("[self-test] interferential_field checks passed:")
    print(f"  reduces to layered_field.potential: {v_here:.6f} vs {v_lf:.6f} mV")
    print(f"  single-channel antisymmetry: V={p:.4f} vs {m:.4f} mV")
    print(f"  modulation depth: parallel=100%, perpendicular=0%, antiparallel=100%")
    print(f"  beat envelope: max={env.max():.3f} (|A+B|=1.6), min={env.min():.3f} (|A-B|=0.4)")
    print(f"  clover: centre depth {centre_depth*100:.1f}% < lobe depth {lobe_depth*100:.1f}% -- PASS")
    print(f"  vertical section: central-column max modulation {np.nanmax(col_depth)*100:.1f}% "
          f"(null penetrates to depth) -- PASS")


if __name__ == "__main__":
    lf._self_test()
    print()
    _self_test()
