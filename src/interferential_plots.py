"""
interferential_plots.py

Every figure of the interferential-current notebook, in one place (previously a
171-line plotting cell plus the NEURON figure defined further down the
notebook).

The physics is unchanged and still entirely borrowed: each channel's potential
is a sum of `layered_field._point_potential` calls made through
`interferential_field`, so the two-layer volume conductor students explored in
the transcutaneous notebook is literally the same object here, just driven by
four pads instead of two.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import layered_field as lf
import interferential_field as ifc
import workshop_setup as ws


def _nm():
    """Return `neurostim_model` (NEURON/PyFibers), with a clear error if absent.

    Called through a function rather than imported at module top so that the
    field-only figures in this workshop keep working on a machine where NEURON
    was never installed -- only the cells that simulate an actual axon fail, and
    they fail with an actionable message instead of an ImportError in the very
    first cell of the notebook.
    """
    m = ws.have_neuron()
    if m is None:
        raise RuntimeError(
            "This figure simulates a real axon and needs NEURON/PyFibers:\n"
            "    pip install neuron pyfibers && pyfibers_compile\n"
            "(On Colab the bootstrap cell at the top of this notebook does it for you.)"
        )
    return m


MOD_CMAP = mcolors.LinearSegmentedColormap.from_list("mod_warm", ifc.COLOR_MOD_CMAP)
FIELD_CMAP = mcolors.LinearSegmentedColormap.from_list("field_seq_blue", lf.COLOR_FIELD_CMAP)


def _mark_electrodes(ax, els, which=('A', 'B')):
    """Draw the electrode pads, coloured by channel, labelled with +/- polarity."""
    colors = {'A': ifc.COLOR_CH_A, 'B': ifc.COLOR_CH_B}
    for ch in which:
        for (xe, ye, sgn) in els[ch]:
            ax.plot(xe, ye, marker='o', ms=13, color=colors[ch],
                    markeredgecolor=ifc.COLOR_INK, zorder=6)
            lbl = ch + ('+' if sgn > 0 else '\u2212')
            ax.annotate(lbl, (xe, ye), color='white', ha='center', va='center',
                        fontsize=8, fontweight='bold', zorder=7)


def _channel_panel(ax, g, ch):
    """One channel's current-density magnitude (filled contours) + streamlines."""
    gr = g['grids'][ch]
    mag = np.hypot(gr['Jx'], gr['Jy'])
    vmax = np.percentile(mag, 98)
    cf = ax.contourf(g['X'], g['Y'], np.clip(mag, 0, vmax),
                     levels=np.linspace(0, vmax, 30), cmap=FIELD_CMAP, extend='max')
    ax.streamplot(g['x'], g['y'], gr['Jx'], gr['Jy'], color=lf.COLOR_STREAMLINE,
                  density=1.1, linewidth=0.6, arrowsize=0.7)
    _mark_electrodes(ax, g['electrodes'], which=(ch,))
    ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    cb = ax.figure.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('|J| (a.u.)', fontsize=8)
    ax.set_title(f'Channel {ch} current density')


def draw_ifc_field(z_depth_mm=15.0, square_mm=80.0, sigma1=lf.DEFAULT_SIGMA1,
                   sigma2=lf.DEFAULT_SIGMA2, h_mm=lf.DEFAULT_H_MM,
                   electrode_radius_mm=0.0, i0_mA=10.0, span_mm=65.0, n=101):
    """Part 3: the two channels' fields + the interferential modulation maps on a
    horizontal plane at depth z_depth_mm below the skin."""
    g = ifc.field_grid(x_range_mm=(-span_mm, span_mm), y_range_mm=(-span_mm, span_mm),
                       n=n, z_depth_mm=z_depth_mm, square_mm=square_mm, i0_mA=i0_mA,
                       sigma1=sigma1, sigma2=sigma2, h_mm=h_mm,
                       electrode_radius_mm=electrode_radius_mm)
    im = ifc.interference_maps(g)

    fig, axs = plt.subplots(2, 2, figsize=(13, 12))
    _channel_panel(axs[0, 0], g, 'A')
    _channel_panel(axs[0, 1], g, 'B')

    # beat modulation depth -- the clover leaf
    ax = axs[1, 0]
    depth_pct = im['depth'] * 100
    cf = ax.contourf(g['X'], g['Y'], depth_pct, levels=np.linspace(0, 100, 26),
                     cmap=MOD_CMAP)
    cs = ax.contour(g['X'], g['Y'], depth_pct, levels=[50, 90], colors=ifc.COLOR_INK,
                    linewidths=0.8, linestyles=['--', '-'])
    ax.clabel(cs, fmt='%d%%', fontsize=7)
    _mark_electrodes(ax, g['electrodes'])
    ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('modulation depth (%)', fontsize=8)
    ax.set_title('Beat modulation depth ("clover leaf")')

    # low-frequency beat current amplitude
    ax = axs[1, 1]
    amp = im['amp']
    vmax = np.percentile(amp, 98)
    cf = ax.contourf(g['X'], g['Y'], np.clip(amp, 0, vmax),
                     levels=np.linspace(0, vmax, 30), cmap=MOD_CMAP, extend='max')
    _mark_electrodes(ax, g['electrodes'])
    ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('beat current amplitude (a.u.)', fontsize=8)
    ax.set_title('Low-frequency beat current amplitude')

    fig.suptitle(f'Top-down (horizontal) view @ depth {abs(z_depth_mm):.0f} mm, '
                 f'pads on a {square_mm:.0f} mm square',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    # ---- perpendicular (depth) section: how far the field/beat penetrate ----
    max_depth = max(40.0, abs(z_depth_mm) + 12.0)
    gv = ifc.vertical_field_grid(s_range_mm=(-span_mm, span_mm), depth_range_mm=(0, max_depth),
                                 n_s=161, n_z=100, square_mm=square_mm,
                                 sigma1=sigma1, sigma2=sigma2, h_mm=h_mm)
    imv = ifc.interference_maps(gv)
    carrier = 0.5 * (imv['env_max'] + imv['env_min'])   # mean current magnitude (field strength)
    Us = gv['grids']['A']['Jalong'] + gv['grids']['B']['Jalong']
    Uz = gv['grids']['A']['Jdepth'] + gv['grids']['B']['Jdepth']
    half = square_mm / 2.0

    figv, axv = plt.subplots(1, 2, figsize=(13, 5.2))
    vmax = np.percentile(carrier, 99)
    cf = axv[0].contourf(gv['S'], gv['Z'], np.clip(carrier, 0, vmax),
                         levels=np.linspace(0, vmax, 30), cmap=FIELD_CMAP, extend='max')
    axv[0].streamplot(gv['s'], gv['z'], Us, Uz, color=lf.COLOR_STREAMLINE, density=1.1,
                      linewidth=0.6, arrowsize=0.7)
    axv[0].axhline(-h_mm, color=lf.COLOR_BOUNDARY, lw=2, ls='--', label=f'layer boundary ({h_mm:.0f} mm)')
    axv[0].axhline(-abs(z_depth_mm), color='#4a3aa7', lw=1.6, ls=':',
                   label=f'horizontal-view depth ({abs(z_depth_mm):.0f} mm)')
    for sx in (-half, half):
        axv[0].plot(sx, 0, marker='v', ms=12, color=ifc.COLOR_INK, zorder=6)
    axv[0].set_xlabel('horizontal distance along cut (mm)'); axv[0].set_ylabel('depth z (mm)')
    figv.colorbar(cf, ax=axv[0], fraction=0.046, pad=0.02).set_label('current magnitude (a.u.)', fontsize=8)
    axv[0].legend(fontsize=7, loc='lower right')
    axv[0].set_title('Field penetration (current magnitude + streamlines)')

    dpct = imv['depth'] * 100
    cf = axv[1].contourf(gv['S'], gv['Z'], dpct, levels=np.linspace(0, 100, 26), cmap=MOD_CMAP)
    cs = axv[1].contour(gv['S'], gv['Z'], dpct, levels=[50, 90], colors=ifc.COLOR_INK,
                        linewidths=0.8, linestyles=['--', '-'])
    axv[1].clabel(cs, fmt='%d%%', fontsize=7)
    axv[1].axhline(-h_mm, color=lf.COLOR_BOUNDARY, lw=2, ls='--')
    axv[1].axhline(-abs(z_depth_mm), color='#4a3aa7', lw=1.6, ls=':',
                   label=f'horizontal-view depth ({abs(z_depth_mm):.0f} mm)')
    for sx in (-half, half):
        axv[1].plot(sx, 0, marker='v', ms=12, color=ifc.COLOR_INK, zorder=6)
    axv[1].set_xlabel('horizontal distance along cut (mm)'); axv[1].set_ylabel('depth z (mm)')
    figv.colorbar(cf, ax=axv[1], fraction=0.046, pad=0.02).set_label('modulation depth (%)', fontsize=8)
    axv[1].legend(fontsize=7, loc='lower right')
    axv[1].set_title('Beat modulation vs depth (central null penetrates)')

    figv.suptitle('Perpendicular (depth) section through the montage centre  '
                  f'(pads ▼ at ±{half:.0f} mm, each offset ±{half:.0f} mm out of plane)',
                  fontsize=12, fontweight='bold')
    figv.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # quick numeric read-out
    ci = n // 2
    print(f"Top-down @ {abs(z_depth_mm):.0f} mm: centre modulation depth = {depth_pct[ci, ci]:.1f}% "
          f"(currents ~perpendicular -> weak beat); peak on the plane = {depth_pct.max():.1f}% "
          f"(off-axis clover lobes).")
    mask = dpct >= 50
    if mask.any():
        print(f"Depth section: the >=50% beat-modulation region reaches ~{abs(gv['Z'][mask].min()):.0f} mm "
              f"deep, and the central null runs straight down the middle.")


def draw_beat(f1_hz=4000.0, beat_hz=100.0, ampA=1.0, ampB=1.0):
    """Part 2: the interferential beat at a single point, plus a zoom showing the
    two carriers drifting in and out of phase."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    f2 = f1_hz + beat_hz
    beat_ms = 1000.0 / beat_hz
    t = np.linspace(0, 2 * beat_ms, 8000)
    s, env, _ = ifc.beat_waveform(t, f1_hz=f1_hz, f2_hz=f2, ampA=ampA, ampB=ampB)
    axs[0].plot(t, s, color=lf.COLOR_STREAMLINE, lw=0.5, label='A + B (interferential current)')
    axs[0].plot(t, env, color=ifc.COLOR_CH_A, lw=2, label='beat envelope')
    axs[0].plot(t, -env, color=ifc.COLOR_CH_A, lw=2)
    axs[0].set_xlabel('time (ms)'); axs[0].set_ylabel('current (a.u.)')
    axs[0].legend(fontsize=8, loc='upper right')
    axs[0].set_title(f'{f1_hz:.0f} + {f2:.0f} Hz  ->  {beat_hz:.0f} Hz beat')

    m = (abs(ampA + ampB) - abs(ampA - ampB)) / (abs(ampA + ampB) + abs(ampA - ampB) + 1e-12)
    tz = np.linspace(0, 3, 3000)
    s1 = ampA * np.cos(2 * np.pi * f1_hz * tz * 1e-3)
    s2 = ampB * np.cos(2 * np.pi * f2 * tz * 1e-3)
    axs[1].plot(tz, s1, color=ifc.COLOR_CH_A, lw=1.0, alpha=0.8, label=f'ch A {f1_hz:.0f} Hz')
    axs[1].plot(tz, s2, color=ifc.COLOR_CH_B, lw=1.0, alpha=0.8, label=f'ch B {f2:.0f} Hz')
    axs[1].plot(tz, s1 + s2, color=ifc.COLOR_INK, lw=1.4, label='sum')
    axs[1].set_xlabel('time (ms)'); axs[1].set_ylabel('current (a.u.)')
    axs[1].legend(fontsize=8, loc='upper right')
    axs[1].set_title('Zoom: the two carriers drift in and out of phase')
    fig.tight_layout()
    plt.show()
    print(f"beat frequency = {beat_hz:.0f} Hz   |   modulation depth m = {m*100:.0f}%  "
          f"(100% only when the two channel amplitudes are equal).")



def draw_ifc_ap(diameter=14.0, amp_mA=65.0, f1_hz=4000.0, beat_hz=100.0,
                fiber_depth_mm=8.0, square_mm=50.0, tstop_ms=40.0, n_nodes=91):
    """Build an MRG axon in the interferential montage, drive it with the two
    carriers, and show the beat-locked firing."""
    f2 = f1_hz + beat_hz
    fiber = _nm().build_axon(diameter=diameter, n_nodes=n_nodes)
    Va, Vb = _nm().set_electrode_interferential(fiber, fiber_depth_mm=fiber_depth_mm,
                                             square_mm=square_mm)
    t, vm, n_ap = _nm().run_interferential(fiber, amp_mA, f1_hz, f2, tstop_ms=tstop_ms)

    n = vm.shape[0]
    node_z = np.linspace(0, fiber.length / 1000.0, n) - fiber.length / 2000.0

    # Local excitatory drive = activating function (2nd spatial derivative of Ve)
    # of each channel, each carrying its own frequency. The initiation site is
    # the compartment where that combined drive is strongest; its beat envelope
    # -- not the montage-centre |a+b| -- is what the AP bursts follow.
    zf = fiber.coordinates[:, 2] / 1000.0
    actA = np.gradient(np.gradient(Va, zf), zf)
    actB = np.gradient(np.gradient(Vb, zf), zf)
    j = int(np.argmax(np.abs(actA) + np.abs(actB)))
    node = int(round(j / (len(zf) - 1) * (n - 1)))
    aA, aB = amp_mA * actA[j], amp_mA * actB[j]

    def ap_times(vtrace, thr=-20.0):
        up = (vtrace[:-1] < thr) & (vtrace[1:] >= thr)
        return t[1:][up]

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), gridspec_kw={'height_ratios': [1, 1, 1.4]})
    tt = np.linspace(0, tstop_ms, 6000)
    drive = aA * np.cos(2 * np.pi * f1_hz * tt * 1e-3) + aB * np.cos(2 * np.pi * f2 * tt * 1e-3)
    env = np.sqrt(aA ** 2 + aB ** 2 + 2 * aA * aB * np.cos(2 * np.pi * (f2 - f1_hz) * tt * 1e-3))
    axs[0].plot(tt, drive, color=lf.COLOR_STREAMLINE, lw=0.4)
    axs[0].plot(tt, env, color=ifc.COLOR_CH_A, lw=2, label='beat envelope of the local drive')
    axs[0].plot(tt, -env, color=ifc.COLOR_CH_A, lw=2)
    axs[0].set_ylabel('activating fn (a.u.)'); axs[0].legend(fontsize=8, loc='upper right')
    axs[0].set_title(f'Local excitatory drive at the initiation site: '
                     f'{f1_hz:.0f}+{f2:.0f} Hz carriers beating at {beat_hz:.0f} Hz')

    aps = ap_times(vm[node])
    axs[1].plot(t, vm[node], color='#2a78d6', lw=0.8)
    for at in aps:
        axs[1].axvline(at, color=ifc.COLOR_CH_A, lw=0.8, alpha=0.6)
    axs[1].set_ylabel('Vm (mV)')
    axs[1].set_title(f'Vm at the initiation node -- {"FIRES in bursts" if len(aps) else "sub-threshold (no firing)"} '
                     f'at the beat rate')

    im = axs[2].imshow(vm, aspect='auto', origin='lower',
                       extent=[t[0], t[-1], node_z[0], node_z[-1]],
                       cmap='inferno', vmin=-90, vmax=40)
    axs[2].set_xlabel('time (ms)'); axs[2].set_ylabel('position along fibre (mm)')
    axs[2].set_title('Space-time heatmap of Vm (propagating APs)')
    fig.colorbar(im, ax=axs[2], label='Vm (mV)')
    fig.tight_layout()
    plt.show()

    n_beats = tstop_ms * beat_hz / 1000.0
    print(f"total spikes at initiation node = {len(aps)}  over ~{n_beats:.0f} beat cycles "
          f"({len(aps)/max(n_beats,1):.1f} spikes/beat).")
    if len(aps) == 0:
        print("Sub-threshold: raise the amplitude to recruit the fibre.")
