"""
transcutaneous_plots.py

Every figure of the transcutaneous-stimulation notebook, in one place.

These functions were originally a single 190-line cell inside the notebook.
They live here instead so that the notebook itself reads as a teaching
document -- physics, question, figure -- rather than as a wall of matplotlib.
Nothing was changed physically in the move: `draw_module0` still calls
`layered_field.plot_field()` / `activating_function_along_depth()`, and
`draw_module1` / `compute_module2` still drive the same MRG axon through
`neurostim_model`.

Colour conventions come from `layered_field.COLOR_*`, so "depolarizing" is the
same red and "hyperpolarizing" the same blue in every figure of BOTH notebooks.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import layered_field as lf
import weiss
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


def draw_module0(sigma1, sigma2, h_mm, i0_mA, fiber_depth_mm, electrode_radius_mm=0.0,
                  separation_mm=60.0):
    '''Module 0: field + activating function at a given fiber depth. Bipolar
    by default -- two electrodes separation_mm apart -- matching how a real
    TENS pad pair actually works (see layered_field.bipolar_potential()).'''
    xr = max(30.0, separation_mm / 2 + 25.0)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    lf.plot_field(axs[0], x_range_mm=(-xr, xr), i0_mA=i0_mA, sigma1=sigma1, sigma2=sigma2, h_mm=h_mm,
                  electrode_radius_mm=electrode_radius_mm, separation_mm=separation_mm)
    axs[0].axhline(-fiber_depth_mm, color='#4a3aa7', lw=1.5, ls=':', label='fiber depth')
    axs[0].legend(loc='lower right', fontsize=7)
    axs[0].set_title('Potential field + current streamlines')

    x, V_layered, act_layered = lf.activating_function_along_depth(
        -fiber_depth_mm, x_range_mm=(-xr, xr), i0_mA=i0_mA, sigma1=sigma1, sigma2=sigma2, h_mm=h_mm,
        electrode_radius_mm=electrode_radius_mm, separation_mm=separation_mm)

    _, _, act_homo = lf.activating_function_along_depth(
        -fiber_depth_mm, x_range_mm=(-xr, xr), i0_mA=i0_mA, sigma1=sigma2, sigma2=sigma2, h_mm=h_mm,
        electrode_radius_mm=electrode_radius_mm, separation_mm=separation_mm)

    axs[1].axhline(0, color='#c3c2b7', lw=0.8)
    axs[1].grid(alpha=0.4, lw=0.5)
    axs[1].set_axisbelow(True)

    axs[1].plot(x, act_homo, color=lf.COLOR_HOMOGENEOUS, ls='--', lw=2, label='Homogeneous (muscle only)')

    axs[1].fill_between(x, act_layered, 0, where=(act_layered >= 0), color=lf.COLOR_DEPOLARIZING,
                         alpha=0.55, label='Layered (depolarizing)')
    axs[1].fill_between(x, act_layered, 0, where=(act_layered < 0), color=lf.COLOR_HYPERPOLARIZING,
                         alpha=0.55, label='Layered (hyperpolarizing)')

    axs[1].set_xlabel('position along fiber (mm)')
    axs[1].set_ylabel('activating function (a.u.)')
    axs[1].set_title(f'Activating function at fiber depth = {fiber_depth_mm} mm')
    axs[1].legend(fontsize=8)

    k = lf.reflection_coefficient(sigma1, sigma2)
    note = ('muscle more conductive -> current pulled toward it' if k < 0
            else 'shallow layer more conductive -> current stays shallow' if k > 0
            else 'no contrast')
    print(f"reflection coefficient k = {k:.3f}  ({note})")
    plt.tight_layout()
    plt.show()


def draw_module1(diameter, amplitude_mA, pulse_width_ms, polarity,
                  fiber_depth_mm, sigma1, sigma2, h_mm, electrode_radius_mm=0.0,
                  electrode_mode='monopolar', separation_mm=60.0):
    '''Module 1: does the fiber fire where the linear activating-function
    prediction (Module 0) says it should?'''
    if electrode_mode == 'bipolar':
        fiber = _nm().build_axon(diameter=diameter, n_nodes=_nm().BIPOLAR_DEFAULT_NODES)
        pot = _nm().set_electrode_bipolar_layered(fiber, fiber_depth_mm=fiber_depth_mm,
                                                separation_mm=separation_mm,
                                                sigma1=sigma1, sigma2=sigma2, h_mm=h_mm,
                                                electrode_radius_mm=electrode_radius_mm)
    else:
        fiber = _nm().build_axon(diameter=diameter)
        pot = _nm().set_electrode_layered(fiber, fiber_depth_mm=fiber_depth_mm,
                                        sigma1=sigma1, sigma2=sigma2, h_mm=h_mm,
                                        electrode_radius_mm=electrode_radius_mm)
    sign = -1.0 if polarity == 'cathodic' else 1.0
    t, vm, fired = _nm().run_single_stim(fiber, sign * amplitude_mA, pulse_width_ms)

    # `pot`/fiber.coordinates cover EVERY compartment (nodes + internodal MYSA/
    # FLUT/STIN sections -- 551 of them for the default fiber), but `vm` (from
    # fiber.record_vm()) only has one row per NODE OF RANVIER (51) -- the MRG
    # double-cable model repeats a fixed 11-compartment unit (node + 10
    # internodal compartments) per node, so node i's position in the full
    # coordinate array is always at index i * (n_compartments-1)/(n_nodes-1).
    # Mixing these up (indexing the 551-length array with a 0-50 index) silently
    # samples the wrong end of the fiber -- caught by comparing predicted vs.
    # actual sites and finding them implausibly far apart even near threshold.
    z_full = fiber.coordinates[:, 2] / 1000.0
    node_idx_in_full = np.linspace(0, len(z_full) - 1, vm.shape[0]).round().astype(int)
    z_nodes = z_full[node_idx_in_full]

    # Signed activating function -- same sign convention as the amplitude
    # passed to run_single_stim above, so f(x) > 0 means "this run's
    # stimulus depolarizes here." Compartments are NOT evenly spaced (the
    # MRG double-cable model alternates short MYSA/FLUT sections with long
    # STIN sections between nodes), so np.gradient is called with the true
    # z_full coordinate array rather than a single scalar spacing -- using
    # a constant spacing here silently distorts the curve and shifts the
    # predicted peak by several nodes.
    act = sign * np.gradient(np.gradient(pot, z_full), z_full)
    predicted_site = z_full[np.argmax(act)]

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [1, 1, 1.5]})

    axs[0].axhline(0, color='#c3c2b7', lw=0.8)
    axs[0].fill_between(z_full, act, 0, where=(act >= 0), color=lf.COLOR_DEPOLARIZING,
                         alpha=0.55, label='depolarizing')
    axs[0].fill_between(z_full, act, 0, where=(act < 0), color=lf.COLOR_HYPERPOLARIZING,
                         alpha=0.55, label='hyperpolarizing')
    axs[0].axvline(predicted_site, color='#4a3aa7', lw=1.5, ls=':',
                    label=f'predicted site ({predicted_site:.1f} mm)')

    if fired:
        # earliest NODE (not compartment) to cross a spike-like threshold ->
        # approx. initiation site, using the correct node-only z positions.
        crossed = vm > -20.0
        any_cross = crossed.any(axis=1)
        first_cross_t = np.where(any_cross, np.argmax(crossed, axis=1), vm.shape[1])
        actual_idx = np.argmin(first_cross_t)
        if first_cross_t[actual_idx] < vm.shape[1]:
            actual_site = z_nodes[actual_idx]
            axs[0].axvline(actual_site, color='#eb6834', lw=1.5, ls='--',
                            label=f'actual initiation ({actual_site:.1f} mm)')

    axs[0].grid(alpha=0.4, lw=0.5)
    axs[0].set_axisbelow(True)
    axs[0].set_xlabel('position along fiber (mm)')
    axs[0].set_ylabel('activating function (a.u.)')
    axs[0].set_title(f'Predicted vs. actual depolarization site ({electrode_mode})')
    axs[0].legend(fontsize=7, loc='upper right')

    n_nodes = vm.shape[0]
    node_z = z_nodes
    pick = np.linspace(0, n_nodes - 1, 5).astype(int)
    node_colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
    for i, c in zip(pick, node_colors):
        axs[1].plot(t, vm[i], color=c, label=f'node {i}')
    axs[1].grid(alpha=0.4, lw=0.5)
    axs[1].set_axisbelow(True)
    axs[1].set_xlabel('time (ms)')
    axs[1].set_ylabel('Vm (mV)')
    axs[1].legend(fontsize=8, loc='upper right')
    axs[1].set_title(f"{'FIRED' if fired else 'did not fire'} — {polarity}, "
                      f"{amplitude_mA} mA, {pulse_width_ms} ms ({electrode_mode})")

    im = axs[2].imshow(vm, aspect='auto', origin='lower',
                       extent=[t[0], t[-1], node_z[0], node_z[-1]],
                       cmap='inferno', vmin=-80, vmax=40)
    axs[2].set_xlabel('time (ms)')
    axs[2].set_ylabel('position along fiber (mm)')
    axs[2].set_title('Space-Time Heatmap of Vm')
    fig.colorbar(im, ax=axs[2], label='Vm (mV)')

    plt.tight_layout()
    plt.show()
    return fired


def compute_module2(diameter, fiber_depth_mm, sigma1, sigma2, h_mm, polarity,
                     pulse_widths_ms=(0.05, 0.1, 0.2, 0.5, 1.0), electrode_radius_mm=0.0,
                     electrode_mode='monopolar', separation_mm=60.0):
    '''Module 2: run the strength-duration sweep. Returns (points, fit).'''
    if electrode_mode == 'bipolar':
        pts = _nm().strength_duration_sweep_bipolar_layered(
            diameter=diameter, fiber_depth_mm=fiber_depth_mm, separation_mm=separation_mm,
            sigma1=sigma1, sigma2=sigma2, h_mm=h_mm,
            pulse_widths_ms=list(pulse_widths_ms), polarity=polarity,
            electrode_radius_mm=electrode_radius_mm, n_nodes=_nm().BIPOLAR_DEFAULT_NODES)
    else:
        pts = _nm().strength_duration_sweep_layered(
            diameter=diameter, fiber_depth_mm=fiber_depth_mm,
            sigma1=sigma1, sigma2=sigma2, h_mm=h_mm,
            pulse_widths_ms=list(pulse_widths_ms), polarity=polarity,
            electrode_radius_mm=electrode_radius_mm)
    fit = weiss.fit_weiss(pts)
    return pts, fit


def plot_module2(pts, fit, my_pts=None, my_fit=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid(alpha=0.4, lw=0.5)
    ax.set_axisbelow(True)
    t_plot = np.logspace(np.log10(0.02), np.log10(3), 200)
    ts, ys = zip(*pts)
    ax.scatter(ts, ys, color='#2a78d6', label='simulated threshold', zorder=3)
    if fit:
        ax.plot(t_plot, fit['rheobase_mA'] * (1 + fit['chronaxie_ms'] / t_plot), color='#2a78d6', ls='--',
                label=f"sim fit: Irh={fit['rheobase_mA']:.2f}mA, tau={fit['chronaxie_ms']*1000:.0f}us")
    if my_pts:
        mts, mys = zip(*my_pts)
        ax.scatter(mts, mys, color='#eb6834', label='your forearm data', zorder=3)
    if my_fit:
        ax.plot(t_plot, my_fit['rheobase_mA'] * (1 + my_fit['chronaxie_ms'] / t_plot), color='#eb6834', ls='--',
                label=f"your fit: Irh={my_fit['rheobase_mA']:.2f}mA, tau={my_fit['chronaxie_ms']*1000:.0f}us")
    ax.axhline(weiss.LITERATURE_ULNAR_RHEOBASE_MA, color='#898781', ls=':',
               label=f"literature ulnar rheobase ({weiss.LITERATURE_ULNAR_RHEOBASE_MA} mA, near-nerve)")
    ax.set_xscale('log')
    ax.set_xlabel('pulse width (ms)')
    ax.set_ylabel('threshold (mA)')
    ax.legend(fontsize=8)
    ax.set_title('Strength-duration curve: model vs. your measurement')
    plt.tight_layout()
    plt.show()
    print(f"Literature (Tsui et al. 2014, near-nerve ulnar): "
          f"rheobase {weiss.LITERATURE_ULNAR_RHEOBASE_MA} mA, chronaxie {weiss.LITERATURE_ULNAR_CHRONAXIE_MS} ms")
