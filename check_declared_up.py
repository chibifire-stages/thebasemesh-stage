# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Gate: the declared upAxis against the geometry of objects whose up is not in doubt.

A bottle, a column and a lamp are taller than they are wide in life, so the axis their height
lands on measures the convention rather than arguing about it.

IN WORLD SPACE. GetPointsAttr is local, and Blender's USD export puts the up-axis conversion
on an ancestor xform. Measured locally, every upright object here is tallest along Z and this
gate reported the whole library as mis-declared. Composed to world they are tallest along Y,
which is what the stage says. The library was right and the first version of this file was
wrong in the same way usda_to_obj.py was.

    python check_declared_up.py [models_dir] [--self-test]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np

# The HEAD noun, which in an English compound is the last word: a ColumnLegBench is a bench
# and a BottleOpener is an opener. An earlier version matched substrings and caught `Clamp`
# on "Lamp"; splitting into words fixed that and still called a bench a column.
UPRIGHT = ("bottle", "vase", "tree", "chair", "lamp", "candle", "column", "tower",
           "wineglass", "hydrant", "cactus", "statue", "obelisk", "bookcase", "stool",
           "barrel", "ladder", "pillar", "jar", "pitcher")
DOMINANCE = 1.3
AXES = "XYZ"


def words(stem):
    return [w.lower() for w in re.split(r"[^A-Za-z]+|(?<=[a-z])(?=[A-Z])", stem) if w]


def is_upright(stem):
    parts = words(stem)
    return bool(parts) and parts[-1] in UPRIGHT


def survey(models, world=True):
    from pxr import Usd, UsdGeom
    declared, rows, ties = set(), [], 0
    for path in sorted(pathlib.Path(models).glob("S_*.usda")):
        stage = Usd.Stage.Open(str(path))
        declared.add((str(UsdGeom.GetStageUpAxis(stage)),
                      float(UsdGeom.GetStageMetersPerUnit(stage))))
        if not is_upright(path.stem[2:]):
            continue
        prim = next((p for p in stage.TraverseAll() if p.IsA(UsdGeom.Mesh)), None)
        if prim is None:
            continue
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        if world:
            matrix = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())).T
            points = points @ matrix[:3, :3].T + matrix[:3, 3]
        extent = np.ptp(points, axis=0)
        order = np.sort(extent)[::-1]
        if order[0] < DOMINANCE * order[1]:
            ties += 1
            continue
        rows.append((path.stem[2:], AXES[int(np.argmax(extent))], extent))
    return declared, rows, ties


def check(models, verbose=True):
    declared, rows, ties = survey(models)
    problems = []
    if len(declared) != 1:
        problems.append("the library declares %d different conventions: %s"
                        % (len(declared), sorted(declared)))
    if not rows:
        problems.append("no upright object had a dominant axis, so nothing was measured")
        return problems
    counts = {a: sum(1 for _, axis, _ in rows if axis == a) for a in AXES}
    winner = max(counts, key=counts.get)
    share = counts[winner] / len(rows)
    up = sorted(declared)[0][0] if declared else "?"
    if verbose:
        print("  declared        %s" % sorted(declared))
        print("  upright objects %d measured, %d dropped as near-ties" % (len(rows), ties))
        for a in AXES:
            print("    height along %s  %3d  (%.1f%%)" % (a, counts[a],
                                                          100 * counts[a] / len(rows)))
        for name, axis, extent in rows:
            if axis != winner:
                print("    against the grain: %-24s %s  %.3f x %.3f x %.3f"
                      % (name[:24], axis, *extent))
    if share < 0.9:
        problems.append("no axis carries height for 90%% of upright objects; the best is "
                        "%s at %.1f%%, so this test cannot settle the convention"
                        % (winner, 100 * share))
    elif winner != up:
        problems.append("the stage declares %s-up and %.1f%% of upright objects are tallest "
                        "along %s. Every consumer that trusts the declaration lays these "
                        "models on their side." % (up, 100 * share, winner))
    return problems


def self_test():
    """Twelve controls. Five must reject a survey that cannot see the convention."""
    r = []
    r.append(("whole words only, so Clamp is not a Lamp",
              is_upright("Lamp_01") and not is_upright("Clamp")))
    r.append(("and ColumnLegBench is not a Column",
              is_upright("Column_01") and not is_upright("ColumnLegBench")))
    r.append(("camel case is split", words("BottleOpener") == ["bottle", "opener"]))
    r.append(("a bottle opener is an opener, not a bottle", not is_upright("BottleOpener")))
    r.append(("an adhesive bottle is still a bottle", is_upright("AdhesiveBottle")))
    r.append(("an object with no upright word is skipped", not is_upright("Abacus")))

    here = pathlib.Path(__file__).resolve().parent / "models"
    if not here.is_dir():
        r.append(("the model directory is present", False))
    else:
        declared, rows, ties = survey(here)
        r.append(("the library declares exactly one convention", len(declared) == 1))
        r.append(("enough upright objects to measure", len(rows) >= 30))
        counts = {a: sum(1 for _, axis, _ in rows if axis == a) for a in AXES}
        r.append(("the near-ties were dropped rather than counted", ties > 0))
        r.append(("one axis dominates, so the answer is not noise",
                  max(counts.values()) / len(rows) > 0.9))
        r.append(("world space agrees with the declaration",
                  max(counts, key=counts.get) == sorted(declared)[0][0]))
        # THE CONTROL THAT WAS MISSING. Local space must give a DIFFERENT answer, or this
        # gate cannot tell a composed transform from an absent one, which is the mistake it
        # was written after.
        _, local_rows, _ = survey(here, world=False)
        local = {a: sum(1 for _, axis, _ in local_rows if axis == a) for a in AXES}
        r.append(("local space disagrees, so the ancestor transform is real",
                  max(local, key=local.get) != max(counts, key=counts.get)))

    bad = sum(1 for _, ok in r if not ok)
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="?",
                    default=str(pathlib.Path(__file__).resolve().parent / "models"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    problems = check(args.models)
    print()
    for p in problems:
        print("FAIL  %s" % p)
    if problems:
        return 1
    print("the declared up axis matches the geometry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
