#!/usr/bin/env python3
"""
selftest.py — verify the whole workshop before you stand in front of a room.

Runs three layers of checks:

  1. The physics self-tests inside each module (current conservation, analytic
     limits, disc and bipolar checks, the interferential beat arithmetic).
  2. Every figure of both notebooks, rendered headlessly to PNG. This is what
     catches a broken plotting call that the physics tests would never see.
  3. Quantitative agreement with the literature: the cathodic/anodic threshold
     asymmetry and the fitted chronaxie must land in the published range.

Usage
-----
    python scripts/selftest.py              # full check (~3-4 min with NEURON)
    python scripts/selftest.py --fast       # skip the strength-duration sweep
    python scripts/selftest.py --no-neuron  # field-only, ~15 s, no NEURON needed

Exits non-zero on the first failure, so it works as a CI step.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FAILURES: list[str] = []


def step(name):
    """Context-manager-free helper: returns a closure that times and reports."""
    def run(fn):
        t0 = time.time()
        sys.stdout.write(f"  {name:.<58s}")
        sys.stdout.flush()
        try:
            fn()
        except Exception as exc:
            print(f"FAIL  ({time.time() - t0:.1f}s)")
            FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
        else:
            print(f"ok    ({time.time() - t0:.1f}s)")
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="two pulse widths instead of five in the sweep")
    ap.add_argument("--no-neuron", action="store_true",
                    help="skip everything that simulates an axon")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    import layered_field as lf
    import interferential_field as ifc
    import weiss
    import workshop_setup as ws
    import transcutaneous_plots as tp
    import interferential_plots as ip

    ws.apply_style()
    outdir = pathlib.Path(tempfile.mkdtemp(prefix="neurostim-selftest-"))

    def save(tag):
        """Persist whatever the figure-drawing function just produced, so a
        human can eyeball the output if a check ever looks suspicious."""
        for i, num in enumerate(plt.get_fignums()):
            plt.figure(num).savefig(outdir / f"{tag}_{i}.png", dpi=60)
        plt.close("all")

    print("\n1. Physics self-tests")
    step("layered_field (images, disc, bipolar, conservation)")(lambda: lf._self_test())
    step("interferential_field (placement, beat, modulation)")(lambda: ifc._self_test())

    print("\n2. Weiss fit (pure arithmetic, no NEURON)")

    def check_weiss():
        fit = weiss.fit_weiss([(0.05, 6.415), (0.1, 3.681), (0.2, 2.197),
                               (0.5, 1.26), (1.0, 0.967)])
        assert fit is not None, "fit returned None"
        assert 0.60 < fit["rheobase_mA"] < 0.85, fit
        assert 0.30 < fit["chronaxie_ms"] < 0.50, fit
        assert weiss.fit_weiss([(0.1, 5.0)]) is None, "single point should not fit"
        assert ws.parse_pairs("0.05, 12\n\nnonsense\n0.2 4") == [(0.05, 12.0), (0.2, 4.0)]
    step("Weiss fit + tolerant data parsing")(check_weiss)

    print("\n3. Notebook 01 figures — field")

    def m0():
        tp.draw_module0(sigma1=lf.DEFAULT_SIGMA1, sigma2=lf.DEFAULT_SIGMA2,
                        h_mm=lf.DEFAULT_H_MM, i0_mA=-2.0, fiber_depth_mm=8.0,
                        electrode_radius_mm=4.0, separation_mm=60.0)
        save("module0")
    step("Module 0 field + activating function (disc, bipolar)")(m0)

    print("\n4. Notebook 02 figures — beat and interferential field")

    def beat():
        ip.draw_beat(f1_hz=4000.0, beat_hz=100.0, ampA=1.0, ampB=1.0)
        ip.draw_beat(f1_hz=4000.0, beat_hz=2.0, ampA=1.0, ampB=0.5)  # the lab setting
        save("beat")
    step("Part 2 beat (100 Hz clinical and 2 Hz lab settings)")(beat)

    def field():
        ip.draw_ifc_field(z_depth_mm=15.0, square_mm=80.0, n=41)
        save("ifc_field")
    step("Part 3 quadripolar field maps (coarse grid)")(field)

    def clover():
        g = ifc.field_grid(n=61)
        m = ifc.interference_maps(g)["depth"]
        c = m[m.shape[0] // 2, m.shape[1] // 2]
        assert c < 0.10, f"central modulation should be a null, got {c:.3f}"
        assert m.max() > 0.20, f"off-centre lobes should be deep, got {m.max():.3f}"
    step("central null shallower than off-centre lobes")(clover)

    if args.no_neuron:
        print("\n5. Axon simulations .................................. SKIPPED (--no-neuron)")
    else:
        nm = ws.have_neuron()
        if nm is None:
            FAILURES.append("NEURON/PyFibers not importable; rerun with --no-neuron "
                            "to check the field-only half")
        else:
            print("\n5. Notebook 01 figures — axon (NEURON)")

            def m1():
                fired = tp.draw_module1(diameter=10.0, amplitude_mA=3.0, pulse_width_ms=0.3,
                                        polarity="cathodic", fiber_depth_mm=8.0,
                                        sigma1=lf.DEFAULT_SIGMA1, sigma2=lf.DEFAULT_SIGMA2,
                                        h_mm=lf.DEFAULT_H_MM, electrode_radius_mm=4.0,
                                        electrode_mode="monopolar", separation_mm=60.0)
                save("module1")
                assert fired, "10 um fibre at 8 mm, 3 mA / 0.3 ms cathodic should fire"
            step("Module 1 single stimulus (fires as expected)")(m1)

            def asymmetry():
                ths = {}
                for pol in ("cathodic", "anodic"):
                    f = nm.build_axon(diameter=10.0)
                    nm.set_electrode_layered(f, fiber_depth_mm=8.0)
                    th, _ = nm.find_threshold_current(f, 0.3, pol)
                    ths[pol] = abs(th)
                ratio = ths["anodic"] / ths["cathodic"]
                print(f"\n      cathodic {ths['cathodic']:.3f} mA, anodic {ths['anodic']:.3f} mA, "
                      f"ratio {ratio:.2f}x", end="")
                assert 2.5 < ratio < 5.5, f"anodic/cathodic ratio out of range: {ratio:.2f}"
                sys.stdout.write("\n  " + " " * 58)
            step("cathodic/anodic asymmetry ~3-4x (monopolar)")(asymmetry)

            def sweep():
                pws = (0.05, 0.5) if args.fast else (0.05, 0.1, 0.2, 0.5, 1.0)
                pts, fit = tp.compute_module2(diameter=10.0, fiber_depth_mm=8.0,
                                              sigma1=lf.DEFAULT_SIGMA1,
                                              sigma2=lf.DEFAULT_SIGMA2,
                                              h_mm=lf.DEFAULT_H_MM, polarity="cathodic",
                                              pulse_widths_ms=pws,
                                              electrode_radius_mm=0.0,
                                              electrode_mode="monopolar")
                tp.plot_module2(pts, fit, [(0.05, 12.0), (0.2, 6.0), (0.5, 4.0)],
                                weiss.fit_weiss([(0.05, 12.0), (0.2, 6.0), (0.5, 4.0)]))
                save("module2")
                assert fit and 0.20 < fit["chronaxie_ms"] < 0.60, (
                    f"chronaxie {fit and fit['chronaxie_ms']} outside the published "
                    "range for a large myelinated fibre")
            step("Module 2 sweep + Weiss fit + student-data overlay")(sweep)

            print("\n6. Notebook 02 figure — axon in the interferential montage")

            def ap():
                ip.draw_ifc_ap(diameter=14.0, amp_mA=65.0, f1_hz=4000.0, beat_hz=100.0,
                               fiber_depth_mm=8.0, square_mm=50.0, tstop_ms=15.0)
                save("ifc_ap")
            step("Part 4 beat-locked firing")(ap)

    print(f"\nFigures written to {outdir}")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("\nAll checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
