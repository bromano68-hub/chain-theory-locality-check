"""
chain_sim_v3_large.py - the large run of experiment E4 from chain_sim_v3.py.

Question: when Alice (electron read qubit a3) keeps her free slot open or
fills it, does anything change in what Bob (hydrogen qubit p.s1, in a separate
web with no path to Alice) can see at his own site?

Same rules (the author's 38 acts), same seed (the author's 19 particle webs),
same uniform scheduler as chain_sim_v3.py. Two upgrades:

  * Many more paired trials, split across processor cores.
  * Lower-noise measurement. At every step we add the exact expected change
    in Bob's link count over all admitted moves (and the exact chance that the
    move is a direct Alice-Bob join), instead of only looking at what the one
    random move did. The average is the same; the noise is much smaller.
    If Bob's site is removed (it can be: a leaf can be cut off, or merged in a
    hub-form), his link count is counted as 0 from then on.

Coupling: each possible move gets a fixed pseudo-random key per step, and the
admitted move with the smallest key happens. That is still a uniform choice
among admitted moves, but the two copies (slot open / slot filled) now make
the same choice except where Alice's difference actually matters.

Usage: python chain_sim_v3_large.py [paired_trials] [moves] [workers]
       defaults: 4000 60 2   (needs chain_sim_v3.py, particles.json)
"""

import os
import random
import sys
import time
from multiprocessing import Pool
from statistics import mean, pstdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_sim_v3 as c  # noqa: E402


def bob_expected_change(w, inst, ib):
    """Exact expected change in Bob's link count from one uniformly chosen move."""
    if ib is None:
        return 0.0
    v0 = w.valence(ib)
    tb = w.tags[ib]
    total = 0.0
    for (a, x, y) in inst:
        touches = (x == ib) or (y == ib and not (a.kind == "attach" and a.dir == "fwd")) \
                  or (a.kind == "hubForm" and a.dir == "inv" and x == ib)
        if not touches:
            continue
        nw = c.rewrite(a, w, x, y)
        j = nw.index_of(tb)
        total += (nw.valence(j) if j is not None else 0) - v0
    return total / len(inst)


ACT_INDEX = {id(a): k for k, a in enumerate(c.ACTS)}
MASK = (1 << 61) - 1


def key(seed, t, a, w, x, y):
    """A fixed pseudo-random number for one possible move at step t, the same
    in both copies of the web. Sites are named by their permanent tags."""
    k = ACT_INDEX[id(a)]
    tx = w.tags[x]
    if a.kind == "attach" and a.dir == "fwd":
        ty = ("dim", y)
    elif a.kind == "hubForm" and a.dir == "inv":
        ty = ("hub",)
    else:
        ty = w.tags[y]
    return hash((seed, t, k, tx, ty)) & MASK


def coupled_step(w, inst, seed, t):
    """Pick the admitted move with the smallest key: a uniform random choice
    among admitted moves, coupled across copies. New sites get tags that are
    the same in both copies."""
    a, x, y = min(inst, key=lambda m: key(seed, t, m[0], w, m[1], m[2]))
    parent = w.tags[x]
    nw = c.rewrite(a, w, x, y)
    if nw.sites > w.sites:
        nw.tags[-1] = ("new", t, parent)
    return nw


def run_pair(seed, steps):
    w0, owner, ids = c.load_particles()
    alice = c.find_site(owner, ids, "electron", "a3")
    bob = c.find_site(owner, ids, "hydrogen", "p.s1")
    out = {}
    for mode in ("open", "closed"):
        w = w0.copy()
        if mode == "closed":
            w = c.rewrite(c.ACTS[1], w, alice, 3)        # A02 at Alice's own site
        ta, tb = w.tags[alice], w.tags[bob]
        r = random.Random(seed)
        exp_link = 0.0
        exp_bob = 0.0
        for t_step in range(steps):
            inst = c.instances(w)
            if not inst:
                break
            ia, ib = w.index_of(ta), w.index_of(tb)
            if ia is not None and ib is not None:
                q, p = min(ia, ib), max(ia, ib)
                hits = sum(1 for (a, x, y) in inst if a.dir == "fwd"
                           and a.kind in ("joinAcross", "joinWithin") and (x, y) == (q, p))
                exp_link += hits / len(inst)
            exp_bob += bob_expected_change(w, inst, ib)
            w = coupled_step(w, inst, seed, t_step)
        ib = w.index_of(tb)
        out[mode] = (exp_link, exp_bob, w.valence(ib) if ib is not None else 0)
    return out


def worker(args):
    wid, n, steps, base_seed = args
    rng = random.Random(base_seed + wid)
    res = []
    t0 = time.time()
    for k in range(n):
        res.append(run_pair(rng.randrange(10**12), steps))
        if (k + 1) % 100 == 0:
            print(f"  worker {wid}: {k + 1}/{n} pairs, {time.time() - t0:.0f}s", flush=True)
    return res


def paired_stats(xs, ys):
    d = [x - y for x, y in zip(xs, ys)]
    m = mean(d)
    se = pstdev(d) / len(d) ** 0.5
    return m, se, (m / se if se else 0.0)


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    per = [trials // workers + (1 if i < trials % workers else 0) for i in range(workers)]
    print(f"E4 large run: {trials} paired trials x {steps} moves, {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        parts = pool.map(worker, [(i, per[i], steps, 777) for i in range(workers)])
    res = [r for p in parts for r in p]
    lo = [r["open"][0] for r in res]
    lc = [r["closed"][0] for r in res]
    bo = [r["open"][1] for r in res]
    bc = [r["closed"][1] for r in res]
    fo = [r["open"][2] for r in res]
    fc = [r["closed"][2] for r in res]
    print(f"\nDone in {time.time() - t0:.0f}s\n")
    print("Alice = electron read qubit a3; Bob = hydrogen qubit p.s1 (separate webs, no path)")
    print("Each number: (Alice's slot OPEN) minus (Alice's slot FILLED), paired by seed\n")
    m, se, z = paired_stats(lo, lc)
    print(f"  Expected direct Alice-Bob joins:         open {mean(lo):.5f}  closed {mean(lc):.5f}"
          f"  difference {m:+.5f} +/- {se:.5f}  ({z:+.1f} sigma)")
    m, se, z = paired_stats(bo, bc)
    print(f"  Expected change in Bob's link count:     open {mean(bo):+.5f}  closed {mean(bc):+.5f}"
          f"  difference {m:+.5f} +/- {se:.5f}  ({z:+.1f} sigma)   [low-noise estimate]")
    m, se, z = paired_stats(fo, fc)
    print(f"  Bob's final link count (one sample):     open {mean(fo):.4f}   closed {mean(fc):.4f}"
          f"   difference {m:+.4f} +/- {se:.4f}  ({z:+.1f} sigma)   [plain count]")


if __name__ == "__main__":
    main()
