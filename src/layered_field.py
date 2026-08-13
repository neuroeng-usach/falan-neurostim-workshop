"""
layered_field.py

An explicit, from-first-principles model of the electric field of a point-source
transcutaneous electrode over a TWO-LAYER tissue volume conductor:

    layer 1 (skin + fat, conductivity sigma1), thickness h
    layer 2 (muscle, conductivity sigma2), semi-infinite, contains the nerve

with insulating air above the skin. This replaces the single-line homogeneous
point-source formula used elsewhere with a derived, checkable multilayer result,
so the electric field is a first-class, visible object in this workshop rather
than something buried inside a function call.

DERIVATION (method of images -- full multiple-reflection series)
------------------------------------------------------------------
Because the electrode sits exactly on the skin surface (z=0), current cannot
flow into the air above: that boundary reflects with coefficient +1 (total
reflection, since air is a perfect insulator). There is exactly one more
interface, between layer 1 (skin+fat) and layer 2 (muscle) at depth z=-h,
which reflects with

    k = (sigma1 - sigma2) / (sigma1 + sigma2)

A first pass at this problem (a single image of amplitude k, as in a plain
two-medium interface) is NOT sufficient here, and an earlier draft of this
file used exactly that truncated version -- it failed a current-conservation
check by ~60% once tested numerically. The reason: because the TOP boundary
also reflects (totally), a wave bounces back and forth between the two
boundaries indefinitely, picking up another factor of k every time it reflects
off the internal (layer1/layer2) interface. This is the classical "hall of
mirrors" / Stefanescu two-layer-earth problem (used for decades in DC
resistivity geophysics for exactly this reason), and it requires the FULL
image series, not just the first bounce:

    Layer 1 (-h <= z <= 0), source at the origin:
        V1(r,z) = (I0/2*pi*sigma1) * [ 1/R_0 + sum_{n=1..inf} k^n * (1/R_n^+ + 1/R_n^-) ]
        R_0    = sqrt(r^2 + z^2)
        R_n^+  = sqrt(r^2 + (z - 2nh)^2)     image at +2nh (bounced off both mirrors)
        R_n^-  = sqrt(r^2 + (z + 2nh)^2)     image at -2nh (bounced off the deep mirror)

    Layer 2 (z < -h), transmitted series:
        V2(r,z) = (I0/(pi*(sigma1+sigma2))) * sum_{n=0..inf} k^n / R'_n
        R'_n = sqrt(r^2 + (2nh - z)^2)

Both series converge geometrically (|k|<1 always, since sigma1, sigma2 > 0),
typically to machine precision within ~40-60 terms for the conductivity
contrasts used here. This is checked against four independent criteria in
`_self_test()`: continuity of V across z=-h, the correct homogeneous-medium
limit (sigma1=sigma2), the correct insulating- and conductive-backing limits,
and -- the strongest check -- direct numerical verification that the total
current crossing a horizontal plane deep in layer 2 equals the injected
current I0, to within ~0.1%, which only holds if the series and its
coefficients are exactly right (an earlier single-image version of this file
was off by ~60% on this specific check, which is how the missing terms were
caught).
"""

from __future__ import annotations

import numpy as np

# np.trapz was renamed to np.trapezoid in NumPy 2.0 and removed in later 2.x.
# Use whichever exists so this file runs on both NumPy 1.x and 2.x (the pyfibers
# dependency pins NumPy 2.2 on some platforms). The two functions are identical.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# --- Reference conductivities (S/m), lumped effective values -----------------
# Dermis ~0.23 S/m, fat ~0.02-0.2 S/m (highly variable) -> shallow layer lumped
# lower than muscle; skeletal muscle ~0.13-0.56 S/m depending on fiber
# orientation (anisotropic) -> deep layer lumped mid-range isotropic estimate.
# See: role of skin layers in transcutaneous stimulation (Kuhn et al.), and
# muscle/fat conductivity estimation studies cited in the workshop notes.
DEFAULT_SIGMA1 = 0.08   # S/m, shallow layer (skin + subcutaneous fat, lumped)
DEFAULT_SIGMA2 = 0.35   # S/m, deep layer (muscle, isotropic-average, contains nerve)
DEFAULT_H_MM = 4.0      # mm, shallow layer thickness (typical forearm skin+fat)


def reflection_coefficient(sigma1: float, sigma2: float) -> float:
    return (sigma1 - sigma2) / (sigma1 + sigma2)


def _point_potential(r_mm, z_mm, i0_mA: float = 1.0,
              sigma1: float = DEFAULT_SIGMA1, sigma2: float = DEFAULT_SIGMA2,
              h_mm: float = DEFAULT_H_MM, n_terms: int = 60):
    """Internal function for point source potential at horizontal distance r_mm and depth z_mm."""
    r = np.asarray(r_mm, dtype=float) * 1e-3   # mm -> m
    z = np.asarray(z_mm, dtype=float) * 1e-3
    h = h_mm * 1e-3
    i0 = i0_mA * 1e-3                           # mA -> A

    k = reflection_coefficient(sigma1, sigma2)

    def safe(d):
        return np.where(d == 0, 1e-9, d)

    # --- Layer 1: direct term + full "bounced off both mirrors" series -----
    R0 = safe(np.sqrt(r**2 + z**2))
    V1 = 1.0 / R0
    for n in range(1, n_terms + 1):
        kn = k**n
        if abs(kn) < 1e-14:
            break
        Rn_plus = safe(np.sqrt(r**2 + (z - 2 * n * h)**2))
        Rn_minus = safe(np.sqrt(r**2 + (z + 2 * n * h)**2))
        V1 = V1 + kn * (1.0 / Rn_plus + 1.0 / Rn_minus)
    V1 = (i0 / (2 * np.pi * sigma1)) * V1

    # --- Layer 2: transmitted series ---------------------------------------
    V2 = np.zeros_like(R0)
    for n in range(0, n_terms + 1):
        kn = k**n
        if n > 0 and abs(kn) < 1e-14:
            break
        Rn_t = safe(np.sqrt(r**2 + (2 * n * h - z)**2))
        V2 = V2 + kn / Rn_t
    V2 = (i0 / (np.pi * (sigma1 + sigma2))) * V2

    in_layer1 = z >= -h
    V = np.where(in_layer1, V1, V2) * 1000.0  # V -> mV
    return V


def potential(x_mm, z_mm, i0_mA: float = 1.0,
              sigma1: float = DEFAULT_SIGMA1, sigma2: float = DEFAULT_SIGMA2,
              h_mm: float = DEFAULT_H_MM, n_terms: int = 60,
              electrode_radius_mm: float = 0.0):
    """Potential (mV) at horizontal offset x_mm and depth z_mm (z<=0, electrode
    at the origin on the skin surface) for a two-layer volume conductor.
    If electrode_radius_mm > 0, models a disc electrode via superposition of
    point sources (using a sunflower spiral distribution for uniform area coverage).
    """
    x = np.asarray(x_mm, dtype=float)
    z = np.asarray(z_mm, dtype=float)
    
    if electrode_radius_mm <= 0.0:
        return _point_potential(np.abs(x), z, i0_mA, sigma1, sigma2, h_mm, n_terms)
        
    N = 100 # Number of point sources for integration
    indices = np.arange(0, N, dtype=float) + 0.5
    r_pts = electrode_radius_mm * np.sqrt(indices / N)
    theta_pts = np.pi * (1 + 5**0.5) * indices
    xs = r_pts * np.cos(theta_pts)
    ys = r_pts * np.sin(theta_pts)
    
    di = i0_mA / N
    V_total = np.zeros_like(x)
    for i in range(N):
        r_dist = np.sqrt((x - xs[i])**2 + ys[i]**2)
        V_total += _point_potential(r_dist, z, di, sigma1, sigma2, h_mm, n_terms)
        
    return V_total


def bipolar_potential(x_mm, z_mm, separation_mm: float, i0_mA: float = 1.0,
                       sigma1: float = DEFAULT_SIGMA1, sigma2: float = DEFAULT_SIGMA2,
                       h_mm: float = DEFAULT_H_MM, n_terms: int = 60,
                       electrode_radius_mm: float = 0.0):
    """Potential (mV) for TWO electrodes on the skin surface, `separation_mm`
    apart, straddling x=0: electrode A (current +i0_mA) at x=-separation_mm/2,
    electrode B (current -i0_mA) at x=+separation_mm/2.

    This is what a real bipolar device (e.g. a TENS unit -- two pads, no
    separate ground) actually does, unlike `potential()`, which is a single
    point/disc source with an implicit distant return. It is exact by
    superposition: the governing equation is linear in injected current, so
    the two-electrode field is just the sum of two calls to `potential()`
    with opposite sign, offset to the two electrode positions. See
    `_self_test_bipolar()` for the checks this relies on (antisymmetry about
    the midpoint, and convergence to the single-source field very close to
    one electrode when the pair is far apart).

    A positive `i0_mA` here means electrode A is the current source; whatever
    code scales this by a signed stimulus amplitude (e.g. `run_single_stim`'s
    cathodic/anodic sign convention) then determines which electrode acts as
    the cathode at any given amplitude, exactly as for the single-electrode
    case -- electrode A simply takes over the role "the electrode" played in
    the monopolar model.
    """
    half = separation_mm / 2.0
    V_a = potential(np.asarray(x_mm, dtype=float) + half, z_mm, i0_mA=i0_mA,
                     sigma1=sigma1, sigma2=sigma2, h_mm=h_mm, n_terms=n_terms,
                     electrode_radius_mm=electrode_radius_mm)
    V_b = potential(np.asarray(x_mm, dtype=float) - half, z_mm, i0_mA=-i0_mA,
                     sigma1=sigma1, sigma2=sigma2, h_mm=h_mm, n_terms=n_terms,
                     electrode_radius_mm=electrode_radius_mm)
    return V_a + V_b


def sigma_at_depth(z_mm, sigma1: float = DEFAULT_SIGMA1, sigma2: float = DEFAULT_SIGMA2,
                    h_mm: float = DEFAULT_H_MM):
    z = np.asarray(z_mm, dtype=float)
    return np.where(z >= -h_mm, sigma1, sigma2)


def field_grid(x_range_mm=(-25, 25), z_range_mm=(-25, 0), n=161, separation_mm=0.0, **kwargs):
    """Build a 2D (x,z) grid of potential (mV), E-field (V/m, via central
    differences) and current density (mA/mm^2-equivalent, J=sigma*E) for
    visualization. kwargs forwarded to `potential`/`bipolar_potential`
    (i0_mA, sigma1, sigma2, h_mm). If separation_mm > 0, models two
    electrodes (bipolar, e.g. a TENS pad pair) instead of one.
    """
    x = np.linspace(*x_range_mm, n)
    z = np.linspace(*z_range_mm, n)
    X, Z = np.meshgrid(x, z)
    if separation_mm > 0:
        V = bipolar_potential(X, Z, separation_mm, **kwargs)
    else:
        V = potential(X, Z, **kwargs)

    dx = x[1] - x[0]
    dz = z[1] - z[0]
    Ex = -np.gradient(V, dx, axis=1)  # mV/mm ~ V/m numerically consistent for plotting
    Ez = -np.gradient(V, dz, axis=0)

    sigma1 = kwargs.get('sigma1', DEFAULT_SIGMA1)
    sigma2 = kwargs.get('sigma2', DEFAULT_SIGMA2)
    h_mm = kwargs.get('h_mm', DEFAULT_H_MM)
    electrode_radius_mm = kwargs.get('electrode_radius_mm', 0.0)
    sig = sigma_at_depth(Z, sigma1, sigma2, h_mm)
    Jx, Jz = sig * Ex, sig * Ez

    return {"x": x, "z": z, "X": X, "Z": Z, "V": V, "Ex": Ex, "Ez": Ez,
            "Jx": Jx, "Jz": Jz, "sigma1": sigma1, "sigma2": sigma2, "h_mm": h_mm,
            "electrode_radius_mm": electrode_radius_mm, "separation_mm": separation_mm}


def activating_function_along_depth(z_mm: float, x_range_mm=(-20, 20), n=401,
                                     separation_mm=0.0, **kwargs):
    """Potential and activating function (2nd spatial derivative) along a
    horizontal line at fixed depth z_mm (where a straight fiber would sit).
    If separation_mm > 0, uses `bipolar_potential` (two electrodes) instead
    of the single-source `potential`.
    Returns (x_mm array, V array (mV), f array (normalized activating function)).
    """
    x = np.linspace(*x_range_mm, n)
    if separation_mm > 0:
        V = bipolar_potential(x, np.full_like(x, z_mm), separation_mm, **kwargs)
    else:
        V = potential(x, np.full_like(x, z_mm), **kwargs)
    dx = x[1] - x[0]
    f = np.zeros_like(V)
    f[1:-1] = (V[:-2] - 2 * V[1:-1] + V[2:]) / dx**2
    f[0], f[-1] = f[1], f[-2]
    return x, V, f


# --- Shared visual style (validated categorical/sequential palette) ---------
# Colors below are the palette used throughout the workshop notebooks so every
# figure reads consistently: one sequential hue for field magnitude, a fixed
# warm/cool pair for the diverging activating-function plot, and a single
# accent (orange/red) for markers so they never compete with the blue field.
COLOR_FIELD_CMAP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
COLOR_STREAMLINE = "#898781"   # muted, recedes behind the field
COLOR_BOUNDARY = "#eb6834"     # orange -- layer boundary
COLOR_ELECTRODE = "#e34948"    # red -- electrode marker
COLOR_INK = "#0b0b0b"
COLOR_DEPOLARIZING = "#e34948"  # red
COLOR_HYPERPOLARIZING = "#2a78d6"  # blue
COLOR_HOMOGENEOUS = "#898781"   # muted gray


def plot_field(ax, x_range_mm=(-30, 30), z_range_mm=(-25, 0), n=220, separation_mm=0.0, **kwargs):
    """Draw isopotential field (filled contours) + current streamlines + layer
    boundary onto a given matplotlib Axes. kwargs forwarded to
    field_grid/potential (i0_mA, sigma1, sigma2, h_mm, electrode_radius_mm).
    If separation_mm > 0, draws two electrodes (bipolar) instead of one.
    Returns the field_grid dict for reuse (e.g. to also plot the activating
    function at a specific depth).
    """
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker

    g = field_grid(x_range_mm, z_range_mm, n=n, separation_mm=separation_mm, **kwargs)
    Vabs = np.abs(g["V"])
    positive = Vabs[Vabs > 0]
    if positive.size == 0:
        return g
    # Most of the domain sits far below the near-electrode value (median grid
    # value is often <10% of the 99th percentile), so vmin is set low (a small
    # multiple of the domain minimum) and vmax is capped at a high percentile
    # rather than the true max -- right at/under the source the point-source
    # term diverges as 1/r, and a handful of grid cells landing close enough
    # would otherwise blow up the color scale by orders of magnitude. Levels
    # are passed as an explicit geomspace array (not a bare int) so contourf's
    # color bins actually follow [vmin, vmax] instead of the raw data range.
    vmin, vmax = positive.min() * 2, np.percentile(positive, 99.0)
    Vclipped = np.clip(Vabs, vmin, vmax)
    levels = np.geomspace(vmin, vmax, 40)

    cmap = mcolors.LinearSegmentedColormap.from_list("field_seq_blue", COLOR_FIELD_CMAP)
    cf = ax.contourf(g["X"], g["Z"], Vclipped, levels=levels, cmap=cmap,
                      norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), extend="both")
    cbar = ax.figure.colorbar(cf, ax=ax, pad=0.02, fraction=0.046,
                               ticks=mticker.LogLocator(base=10, subs=(1, 2, 5)))
    cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    cbar.set_label("|V| (mV)", fontsize=9, color=COLOR_INK)
    cbar.ax.tick_params(labelsize=8, colors=COLOR_INK)

    ax.streamplot(g["x"], g["z"], g["Jx"], g["Jz"], color=COLOR_STREAMLINE, density=1.0,
                  linewidth=0.6, arrowsize=0.7)
    ax.axhline(-g["h_mm"], color=COLOR_BOUNDARY, lw=2, ls="--",
               label=f"layer boundary (depth {g['h_mm']:.1f} mm)")

    r = g.get('electrode_radius_mm', 0.0)
    sep = g.get('separation_mm', 0.0)
    if sep > 0:
        # bipolar: electrode A (source, +) at x=-sep/2, electrode B (sink, -) at x=+sep/2
        half = sep / 2.0
        centers = [(-half, COLOR_DEPOLARIZING, 'A (+)'), (half, COLOR_HYPERPOLARIZING, 'B (−)')]
        for xc, color, tag in centers:
            if r > 0:
                ax.plot([xc - r, xc + r], [0, 0], color=color, lw=6, zorder=5,
                         solid_capstyle='butt', label=f'electrode {tag}, r={r:.1f} mm')
            else:
                ax.plot(xc, 0, marker="*", linestyle="None", color=color,
                         markersize=16, zorder=5, markeredgecolor=COLOR_INK,
                         label=f'electrode {tag}')
    elif r > 0:
        ax.plot([-r, r], [0, 0], color=COLOR_ELECTRODE, lw=6, zorder=5,
                 solid_capstyle='butt', label=f'disc electrode (r={r:.1f} mm)')
    else:
        ax.plot(0, 0, marker="*", linestyle="None", color=COLOR_ELECTRODE,
                 markersize=16, zorder=5, markeredgecolor=COLOR_INK,
                 label='point electrode')

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z depth (mm)")
    ax.set_facecolor(COLOR_FIELD_CMAP[0])
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    return g


# ---------------------------------------------------------------------------
def _self_test():
    """Sanity checks: continuity at the boundary, and known limiting cases."""
    sigma1, sigma2, h = 0.08, 0.35, 4.0

    # 1) Continuity of V across the layer1/layer2 boundary.
    v_above = potential(5.0, -h + 1e-6, sigma1=sigma1, sigma2=sigma2, h_mm=h)
    v_below = potential(5.0, -h - 1e-6, sigma1=sigma1, sigma2=sigma2, h_mm=h)
    assert abs(v_above - v_below) < 1e-3, (v_above, v_below)

    # 2) Homogeneous limit (sigma1==sigma2): matches single-layer half-space formula.
    v_layered = potential(6.0, -3.0, sigma1=0.25, sigma2=0.25, h_mm=4.0)
    r = np.sqrt(6.0**2 + 3.0**2) * 1e-3
    v_homogeneous = (1e-3) / (2 * np.pi * 0.25 * r) * 1000.0
    assert abs(v_layered - v_homogeneous) / v_homogeneous < 1e-6

    # 3) Insulating backing (sigma2 -> 0): potential in layer 1 should exceed the
    #    homogeneous value (field "trapped" above a non-conducting layer).
    v_insulating = potential(6.0, -3.0, sigma1=0.25, sigma2=1e-6, h_mm=4.0)
    assert v_insulating > v_homogeneous

    # 4) Conductive backing (sigma2 -> large): potential in layer 1 should be
    #    reduced relative to homogeneous (deep conductor "steals" current).
    v_conductive = potential(6.0, -3.0, sigma1=0.25, sigma2=50.0, h_mm=4.0)
    assert v_conductive < v_homogeneous

    # 5) Current conservation: total current crossing a horizontal plane deep in
    #    layer 2 must equal the injected current I0, exactly (up to numerical
    #    integration error), by charge conservation in steady-state DC
    #    conduction. This is the strongest check -- it depends on getting every
    #    prefactor and every term of the image series right, not just the
    #    limiting cases. All units are carried through in SI (V, A, m, S/m)
    #    to avoid the unit-bookkeeping mistake that caused an earlier version
    #    of this check to silently report a false pass.
    i0_mA = 1.0
    i0_A = i0_mA * 1e-3
    z_test_mm = -15.0  # deep in layer 2, well past the h=4mm boundary
    dz_mm = 1e-4       # for a central-difference derivative, in mm

    r_mm = np.linspace(1e-6, 3000, 800_000)  # mm; large radius + fine resolution
    V_a = potential(r_mm, np.full_like(r_mm, z_test_mm + dz_mm / 2),
                    i0_mA=i0_mA, sigma1=sigma1, sigma2=sigma2, h_mm=h) * 1e-3  # mV -> V
    V_b = potential(r_mm, np.full_like(r_mm, z_test_mm - dz_mm / 2),
                    i0_mA=i0_mA, sigma1=sigma1, sigma2=sigma2, h_mm=h) * 1e-3  # mV -> V
    dz_m = dz_mm * 1e-3
    Ez = -(V_a - V_b) / dz_m          # V/m
    Jz = sigma2 * Ez                  # A/m^2

    r_m = r_mm * 1e-3
    total_current_A = _trapz(2 * np.pi * r_m * np.abs(Jz), r_m)  # A
    ratio = total_current_A / i0_A
    assert 0.99 < ratio < 1.01, f"current conservation failed: ratio={ratio:.4f}"
    print(f"[self-test] current conservation: integrated {total_current_A*1000:.4f} mA "
          f"vs injected {i0_mA:.4f} mA (ratio {ratio:.4f}) -- PASS")

    print("[self-test] all analytic sanity checks passed:")
    print(f"  continuity at boundary: {v_above:.6f} vs {v_below:.6f} mV")
    print(f"  homogeneous-limit match: {v_layered:.6f} vs {v_homogeneous:.6f} mV")
    print(f"  insulating backing raises V: {v_insulating:.4f} > {v_homogeneous:.4f}")
    print(f"  conductive backing lowers V: {v_conductive:.4f} < {v_homogeneous:.4f}")

    _self_test_disc(sigma1, sigma2, h)
    _self_test_bipolar(sigma1, sigma2, h)


def _self_test_bipolar(sigma1: float, sigma2: float, h: float):
    """Checks specific to bipolar_potential() (two electrodes -- source +
    sink -- instead of one). Since bipolar_potential is pure superposition of
    the already-validated potential() function, the main new risk is a
    sign/offset bug in placing the two electrodes, not new physics -- so
    these checks target exactly that, rather than re-deriving current
    conservation from scratch."""
    sep = 40.0

    # 1) Antisymmetry: with the pair centered on x=0, swapping x -> -x must
    #    flip the sign of V (electrode A's role at -x mirrors electrode B's
    #    role at +x). Only exact for the point-source path (electrode_radius
    #    =0); the disc path uses a quasi-random point spiral that isn't
    #    exactly mirror-symmetric, so this only checks the point source.
    x_test, z_test = 12.3, -7.5
    v1 = bipolar_potential(x_test, z_test, sep, i0_mA=2.0, sigma1=sigma1, sigma2=sigma2, h_mm=h)
    v2 = bipolar_potential(-x_test, z_test, sep, i0_mA=2.0, sigma1=sigma1, sigma2=sigma2, h_mm=h)
    assert abs(v1 + v2) < 1e-6 * max(abs(v1), 1.0), \
        f"bipolar field is not antisymmetric about the midpoint: {v1} vs {v2}"

    # 2) Far-separation limit: right next to electrode A, with the pair very
    #    far apart, the bipolar field should converge to the single-source
    #    (monopolar) field -- electrode B is too far away to matter locally.
    sep_far = 4000.0
    offset_from_a = 5.0  # mm
    x_near_a = -sep_far / 2 + offset_from_a
    v_bipolar = bipolar_potential(x_near_a, z_test, sep_far, i0_mA=1.0,
                                   sigma1=sigma1, sigma2=sigma2, h_mm=h)
    v_mono = potential(offset_from_a, z_test, i0_mA=1.0, sigma1=sigma1, sigma2=sigma2, h_mm=h)
    rel_diff = abs(v_bipolar - v_mono) / abs(v_mono)
    assert rel_diff < 0.05, \
        f"bipolar near-source field does not converge to the monopolar field at large separation: rel diff={rel_diff:.3f}"

    print("[self-test] bipolar checks passed:")
    print(f"  antisymmetry about midpoint: V(x)={v1:.4f} vs V(-x)={v2:.4f} mV")
    print(f"  near-source convergence to monopolar at large separation: rel diff {rel_diff:.4f}")


def _self_test_disc(sigma1: float, sigma2: float, h: float):
    """Checks specific to the disc electrode (electrode_radius_mm > 0), which
    the checks above never exercise -- every call there uses the default
    electrode_radius_mm=0.0. Added after manually verifying the disc feature
    (superposition of point sources over the disc) against the same criteria
    used for the point source: convergence to the point-source limit, correct
    monotonic physical behavior as radius grows, and current conservation.
    """
    # 1) Convergence: a vanishingly small disc must reproduce the point source.
    v_point = potential(6.0, -8.0, i0_mA=1.0, sigma1=sigma1, sigma2=sigma2, h_mm=h)
    v_disc_tiny = potential(6.0, -8.0, i0_mA=1.0, sigma1=sigma1, sigma2=sigma2, h_mm=h,
                             electrode_radius_mm=0.01)
    rel_diff = abs(v_point - v_disc_tiny) / v_point
    assert rel_diff < 1e-4, f"tiny disc does not converge to point source: rel diff={rel_diff:.2e}"

    # 2) Spreading current over a larger disc must monotonically lower the peak
    #    potential right under the electrode (same total current, larger area).
    radii = [0.0, 1.0, 3.0, 5.0, 8.0, 12.0]
    peaks = [potential(0.0, -0.5, i0_mA=1.0, sigma1=sigma1, sigma2=sigma2, h_mm=h,
                        electrode_radius_mm=r) for r in radii]
    assert all(peaks[i] > peaks[i + 1] for i in range(len(peaks) - 1)), \
        f"peak potential is not monotonically decreasing with electrode radius: {list(zip(radii, peaks))}"

    # 3) Current conservation with a real disc radius (5 mm), same test as the
    #    point-source case above but for the disc path. Coarser resolution
    #    (still catches an order-of-magnitude bug like the one that motivated
    #    the point-source series fix) to keep notebook startup time reasonable
    #    -- the disc path costs roughly 100x more per evaluated point.
    i0_mA = 1.0
    z_test_mm = -15.0
    dz_mm = 1e-4
    r_mm = np.linspace(1e-6, 1000, 4000)
    V_a = potential(r_mm, np.full_like(r_mm, z_test_mm + dz_mm / 2), i0_mA=i0_mA,
                     sigma1=sigma1, sigma2=sigma2, h_mm=h, electrode_radius_mm=5.0) * 1e-3
    V_b = potential(r_mm, np.full_like(r_mm, z_test_mm - dz_mm / 2), i0_mA=i0_mA,
                     sigma1=sigma1, sigma2=sigma2, h_mm=h, electrode_radius_mm=5.0) * 1e-3
    dz_m = dz_mm * 1e-3
    Ez = -(V_a - V_b) / dz_m
    Jz = sigma2 * Ez
    r_m = r_mm * 1e-3
    total_current_A = _trapz(2 * np.pi * r_m * np.abs(Jz), r_m)
    ratio = total_current_A / (i0_mA * 1e-3)
    assert 0.97 < ratio < 1.03, f"disc current conservation failed: ratio={ratio:.4f}"

    print("[self-test] disc electrode checks passed:")
    print(f"  tiny-disc -> point-source convergence: rel diff {rel_diff:.2e}")
    print(f"  peak V monotonically decreasing with radius: {[f'{p:.1f}' for p in peaks]}")
    print(f"  disc (r=5mm) current conservation ratio: {ratio:.4f}")


if __name__ == "__main__":
    _self_test()
