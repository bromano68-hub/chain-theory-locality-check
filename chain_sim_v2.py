"""
chain_sim_v2.py - a closer approximation of the Chain Theory draft's network rules.

What is taken from the draft (Chain Theory Compendium, working draft 23 Oct 2026):
  * Sites are qudits of dimension 2, 3 or 4. A site's valence is its dimension:
    it can hold at most that many link ends ("the valence of a site is its
    dimension read as a degree"; caps: "no site beyond dimension four").
  * A link joins two sites and has a genre (2,2) (2,3) (2,4) (3,3) (3,4) (4,4).
  * The seed "universe web": 444 sites (dims 2:242, 3:127, 4:75) and 551 links
    (genres 22:85, 23:34, 24:229, 33:134, 34:67, 44:2). Instrument I.2.
    (The real link list is in the author's repository, not the PDF, so this
    script builds a random web with exactly those counts.)
  * Move kinds from the move table (Table I.2.2), simplified:
      a1/a2/a3  attach a new site at an open end, genres (2,3) (2,4) (3,4)
      hf        hub-forming pairing at the carrier genre (2,2), adds 2 sites
                (half of the draft's "depth rule", the spacetime-growth rule)
      j2/j3/j4  join two existing sites of equal dimension, genres (2,2) (3,3)
                (4,4), keeps the site count (the other half of the depth rule)
      cut       remove a link (the reverse of the pairing)
    Hub steps (hsd, hsp, hsr, hsii) are left out.
  * Admission: a move is allowed only where the sites have spare valence
    ("the spare valence the whole of what the attach reads", Open result I.3.1).
    The draft's joins are not limited by distance (I.3.1: 521 of 554 admitted
    joins at distances 3 to 9).
  * Space is read from (2,2) links (chapter I.22: "the depth metric as the
    (2,2) count"), so spatial distance here = shortest path using (2,2) links only.

What is NOT in the draft and had to be invented:
  * A scheduler. The draft lists every admitted move (a census of possibilities)
    and has no rule for which move actually happens. Here: pick a move kind by
    weight, then one admitted instance uniformly at random.
  * Probabilities, amplitudes and the quantum state. This is a classical graph model.

Experiments:
  E1  Calibration: share of admitted (same-web) joins at distance >= 3, vs the
      draft's 521/554 = 94%.
  E2  Scaling (test T1): do long joins fade as the web grows?
  E3  Light cone (Open result I.4.1): do steps add links outside a cone of slope 2?
  E4  Control (the Open step): can Alice, by a purely local act (using up or
      freeing her own spare valence), change the odds that distant Bob gets a
      direct (2,2) link to her?

Usage:  python chain_sim_v2.py          (about 1-3 minutes)
        python chain_sim_v2.py --quick  (smaller runs)
"""

import random
import sys
from collections import deque
from statistics import mean, pstdev

DIMS = (2, 3, 4)
SEED_DIMS = {2: 242, 3: 127, 4: 75}
SEED_GENRES = {(2, 4): 229, (3, 4): 67, (4, 4): 2, (3, 3): 134, (2, 2): 85, (2, 3): 34}
CONE_SLOPE = 2  # from the signature Gram diag(4,-1,-1,-1): slope 2 (I.4.1)


class Web:
    def __init__(self):
        self.dim = []          # site -> dimension
        self.adj = []          # site -> {neighbor: genre}
        self.alive = []        # site -> bool

    # ---- basic structure -------------------------------------------------
    def add_site(self, d):
        self.dim.append(d)
        self.adj.append({})
        self.alive.append(True)
        return len(self.dim) - 1

    def free(self, s):
        return self.dim[s] - len(self.adj[s])

    def add_link(self, u, v):
        g = tuple(sorted((self.dim[u], self.dim[v])))
        self.adj[u][v] = g
        self.adj[v][u] = g

    def remove_link(self, u, v):
        del self.adj[u][v]
        del self.adj[v][u]

    def sites(self):
        return [s for s in range(len(self.dim)) if self.alive[s]]

    def links(self):
        out = []
        for u in self.sites():
            for v in self.adj[u]:
                if u < v:
                    out.append((u, v))
        return out

    def copy(self):
        w = Web()
        w.dim = list(self.dim)
        w.adj = [dict(a) for a in self.adj]
        w.alive = list(self.alive)
        return w

    # ---- distances -------------------------------------------------------
    def bfs(self, src, only22=False, limit=None):
        dist = {src: 0}
        q = deque([src])
        while q:
            x = q.popleft()
            if limit is not None and dist[x] >= limit:
                continue
            for y, g in self.adj[x].items():
                if only22 and g != (2, 2):
                    continue
                if y not in dist:
                    dist[y] = dist[x] + 1
                    q.append(y)
        return dist

    def components(self):
        seen, comps = set(), 0
        for s in self.sites():
            if s not in seen:
                comps += 1
                seen.update(self.bfs(s))
        return comps


# ---- seed web with the draft's census ----------------------------------------
def make_seed(rng, tries=200):
    """Random web with exactly the draft's site and genre counts.
    Ququart slots are exactly filled (300 of 300), so place (4,x) links first."""
    for _ in range(tries):
        w = Web()
        by_dim = {d: [w.add_site(d) for _ in range(n)] for d, n in SEED_DIMS.items()}
        ok = True
        for (a, b), n in SEED_GENRES.items():
            for _ in range(n):
                placed = False
                for _ in range(400):
                    # prefer sites with the most spare valence, so tight dims fill
                    ca = [s for s in rng.sample(by_dim[a], min(25, len(by_dim[a]))) if w.free(s) > 0]
                    cb = [s for s in rng.sample(by_dim[b], min(25, len(by_dim[b]))) if w.free(s) > 0]
                    if not ca or not cb:
                        continue
                    u = max(ca, key=w.free)
                    v = max(cb, key=w.free)
                    if u != v and v not in w.adj[u]:
                        w.add_link(u, v)
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return w
    raise RuntimeError("could not build a seed web with the draft's census")


# ---- moves ---------------------------------------------------------------
ATTACH_GENRES = {"a1": (2, 3), "a2": (2, 4), "a3": (3, 4)}
JOIN_DIMS = {"j2": 2, "j3": 3, "j4": 4}


def admitted_attach(w, genre):
    a, b = genre
    return [(s, b if w.dim[s] == a else a) for s in w.sites()
            if w.free(s) > 0 and w.dim[s] in genre]


def admitted_joins(w, d, locality=None, around=None):
    """All pairs of distinct, unlinked sites of dimension d with spare valence.
    locality=None reproduces the draft (no distance condition);
    locality=R keeps only pairs within graph distance R."""
    cands = [s for s in w.sites() if w.dim[s] == d and w.free(s) > 0]
    pairs = []
    for i, u in enumerate(cands):
        near = w.bfs(u, limit=locality) if locality is not None else None
        for v in cands[i + 1:]:
            if v in w.adj[u]:
                continue
            if near is not None and v not in near:
                continue
            pairs.append((u, v))
    return pairs


def step(w, rng, weights, locality=None):
    """One move: pick a kind by weight, then one admitted instance uniformly.
    Returns (kind, new_links) or (kind, None) if nothing was admitted."""
    kinds = list(weights)
    kind = rng.choices(kinds, [weights[k] for k in kinds])[0]
    if kind in ATTACH_GENRES:
        opts = admitted_attach(w, ATTACH_GENRES[kind])
        if not opts:
            return kind, None
        s, newdim = rng.choice(opts)
        n = w.add_site(newdim)
        w.add_link(s, n)
        return kind, [(s, n)]
    if kind == "hf":
        opts = [s for s in w.sites() if w.dim[s] == 2 and w.free(s) > 0]
        if not opts:
            return kind, None
        s = rng.choice(opts)
        q1 = w.add_site(2)
        q2 = w.add_site(2)
        w.add_link(s, q1)
        w.add_link(q1, q2)
        return kind, [(s, q1), (q1, q2)]
    if kind in JOIN_DIMS:
        pairs = admitted_joins(w, JOIN_DIMS[kind], locality)
        if not pairs:
            return kind, None
        u, v = rng.choice(pairs)
        w.add_link(u, v)
        return kind, [(u, v)]
    if kind == "cut":
        ls = w.links()
        if not ls:
            return kind, None
        u, v = rng.choice(ls)
        w.remove_link(u, v)
        return kind, []
    raise ValueError(kind)


GROWTH = {"a1": 2, "a2": 2, "a3": 2, "hf": 3, "j2": 1, "j3": 1, "j4": 1, "cut": 1}


# ---- E1 + E2: long joins at the seed and as the web grows ------------------------
def join_distance_profile(w, rng, sample=300):
    """Distances (in the parent web) of admitted joins, sampled."""
    pairs = []
    for d in DIMS:
        pairs += admitted_joins(w, d)
    if not pairs:
        return None
    pick = rng.sample(pairs, min(sample, len(pairs)))
    same, dists = 0, []
    for u, v in pick:
        dist = w.bfs(u).get(v)
        if dist is not None:
            same += 1
            dists.append(dist)
    return len(pairs), same / len(pick), dists


def e1_e2(rng, sizes):
    print("E1/E2  Admitted joins: how far apart are the two sites?")
    print(f"{'sites':>6} | {'admitted joins':>14} | {'same web':>8} | "
          f"{'dist>=3 (same web)':>18} | {'mean dist':>9} | {'max dist':>8}")
    print("-" * 80)
    w = make_seed(rng)
    rows = []
    for target in sizes:
        while len(w.sites()) < target:
            step(w, rng, GROWTH)
        prof = join_distance_profile(w, rng)
        if prof is None:
            print(f"{len(w.sites()):6d} | no admitted joins")
            continue
        n, same, dists = prof
        long_share = sum(1 for d in dists if d >= 3) / max(len(dists), 1)
        rows.append((len(w.sites()), long_share))
        print(f"{len(w.sites()):6d} | {n:14d} | {same:8.0%} | {long_share:18.0%} | "
              f"{(mean(dists) if dists else 0):9.1f} | {(max(dists) if dists else 0):8d}")
    print("Draft's own count at its closures: 521 of 554 = 94% at distance 3-9.\n")
    return rows


# ---- E3: the cone of influence ----------------------------------------------------
def join_step_sitekeyed(w, rng, locality, n_orig):
    """One join move with a site-keyed scheduler, so two copies of the web make
    the same random draws and differ only where a local difference matters.
    Draws: dimension, a starting site u, a partner index - always 3 numbers."""
    d = rng.choice(DIMS)
    fu = rng.random()
    fv = rng.random()
    cands = [s for s in range(n_orig) if w.alive[s] and w.dim[s] == d]
    u = cands[int(fu * len(cands))]
    if w.free(u) <= 0:
        return
    near = w.bfs(u, limit=locality) if locality is not None else None
    partners = [v for v in cands if v != u and w.free(v) > 0 and v not in w.adj[u]
                and (near is None or v in near)]
    if not partners:
        return
    v = partners[int(fv * len(partners))]
    w.add_link(u, v)


def e3_influence(trials, steps, locality, seed):
    """Copy A: the web as is. Copy B: one link at the entry site is cut first,
    freeing a slot (a local act). Run both with the same draws and record how
    far from the entry (distance in the starting web) the first difference shows."""
    rng = random.Random(seed)
    beyond, first_far, first_steps = 0, [], []
    for _ in range(trials):
        w0 = make_seed(rng)
        n0 = len(w0.dim)
        qubits = [s for s in w0.sites() if w0.dim[s] == 2 and w0.free(s) == 0]
        entry = rng.choice(qubits)
        dist0 = w0.bfs(entry)
        a, b = w0.copy(), w0.copy()
        nb = next(iter(b.adj[entry]))
        a.remove_link(entry, nb)      # same cut in both, so the only difference
        b.remove_link(entry, nb)      # is what follows:
        a.add_link(entry, nb)         # copy A puts it back (slot stays full)
        s = rng.randrange(10**12)
        ra, rb = random.Random(s), random.Random(s)
        hit = None
        for t in range(1, steps + 1):
            join_step_sitekeyed(a, ra, locality, n0)
            join_step_sitekeyed(b, rb, locality, n0)
            diff = [x for x in range(n0) if set(a.adj[x]) ^ set(b.adj[x])
                    and x != entry and x != nb]
            if diff:
                far = max(dist0.get(x, 10**6) for x in diff)
                hit = (t, far)
                break
        if hit:
            t, far = hit
            first_steps.append(t)
            first_far.append(far)
            if far > CONE_SLOPE * t:
                beyond += 1
    return beyond, len(first_far), first_far, first_steps


# ---- E4: can Alice control a shortcut to Bob? -------------------------------------
JOIN_ONLY = {"j2": 1.0}  # the spacetime join alone, isolating the Open step


def pick_far_qubits(w, rng):
    q = [s for s in w.sites() if w.dim[s] == 2 and w.free(s) > 0]
    for _ in range(500):
        a, b = rng.sample(q, 2)
        d = w.bfs(a, only22=False).get(b)
        if d is not None and d >= 6:
            return a, b, d
    a, b = rng.sample(q, 2)
    return a, b, w.bfs(a).get(b)


def e4_control(rng, trials, steps):
    """Alice's local act: 'closed' = she uses up her spare valence by
    attaching a leaf at her own site (a purely local move); 'open' = she doesn't.
    Each trial runs both versions from the same random seed."""
    got_link_to_alice = {"open": 0, "closed": 0}
    bob_new_links = {"open": [], "closed": []}
    drop22 = {"open": [], "closed": []}
    seed_rng = random.Random(rng.randrange(10**9))
    base = make_seed(seed_rng)
    for _ in range(trials):
        w0 = base.copy()
        alice, bob, _ = pick_far_qubits(w0, rng)
        s = rng.randrange(10**12)
        for mode in ("open", "closed"):
            w = w0.copy()
            if mode == "closed":
                while w.free(alice) > 0:
                    leaf = w.add_site(3)
                    w.add_link(alice, leaf)
            before22 = w.bfs(alice, only22=True).get(bob)
            bob_deg0 = len(w.adj[bob])
            r = random.Random(s)
            for _ in range(steps):
                step(w, r, JOIN_ONLY)
            if alice in w.adj[bob]:
                got_link_to_alice[mode] += 1
            bob_new_links[mode].append(len(w.adj[bob]) - bob_deg0)
            after22 = w.bfs(alice, only22=True).get(bob)
            inf = 10**6
            drop22[mode].append((before22 if before22 is not None else inf) >
                                (after22 if after22 is not None else inf))
    return got_link_to_alice, bob_new_links, drop22


def ztest_prop(k1, k2, n):
    p1, p2 = k1 / n, k2 / n
    p = (k1 + k2) / (2 * n)
    se = (2 * p * (1 - p) / n) ** 0.5
    return p1, p2, ((p1 - p2) / se if se > 0 else 0.0)


def main():
    quick = "--quick" in sys.argv
    rng = random.Random(20261101)

    w = make_seed(rng)
    census = {}
    for u, v in w.links():
        g = tuple(sorted((w.dim[u], w.dim[v])))
        census[g] = census.get(g, 0) + 1
    print(f"Seed web: {len(w.sites())} sites, {len(w.links())} links, "
          f"{w.components()} components, spare valence {sum(w.free(s) for s in w.sites())}")
    print("  genre census:", dict(sorted(census.items())),
          " (draft: 22:85 23:34 24:229 33:134 34:67 44:2; spare valence 63)\n")

    e1_e2(rng, [444, 1000, 2000] if quick else [444, 1000, 2000, 4000])

    trials3, steps3 = (40, 60) if quick else (150, 80)
    print("E3  Cone of influence: free one slot at an entry site, run two copies with")
    print("    the same random draws, and see where the first difference appears")
    for loc, label in ((None, "draft joins (no distance limit)"), (2, "local joins (radius 2)")):
        beyond, n, far, ts = e3_influence(trials3, steps3, loc, seed=7)
        if n:
            print(f"  {label:32}: effect appeared in {n}/{trials3} trials, each created by a single move;")
            print(f"  {'':32}  distance from cause: median {sorted(far)[len(far)//2]}, max {max(far)} links; "
                  f"{sum(1 for x in far if x > 2)}/{n} beyond 2 links")
        else:
            print(f"  {label:32}: no difference appeared")
    print("  Draft's own result: the rule adds a link outside its cone at the first step (I.4.1).\n")

    n4, st4 = (400, 30) if quick else (2000, 30)
    got, bobnew, drop = e4_control(random.Random(99), n4, st4)
    p_open, p_closed, z = ztest_prop(got["open"], got["closed"], n4)
    d_open, d_closed, zd = ztest_prop(sum(drop["open"]), sum(drop["closed"]), n4)
    print(f"E4  Control: {n4} paired trials, {st4} spacetime joins (j2) each, "
          "Alice and Bob qubits >= 6 links apart")
    print(f"  Bob ends up directly linked to Alice:   open slot {p_open:.1%}   "
          f"closed slot {p_closed:.1%}   ({z:+.1f} sigma)")
    print(f"  Alice-Bob (2,2) distance shrinks:        open slot {d_open:.1%}   "
          f"closed slot {d_closed:.1%}   ({zd:+.1f} sigma)")
    diffs = [o - c for o, c in zip(bobnew["open"], bobnew["closed"])]
    md = mean(diffs)
    se = pstdev(diffs) / len(diffs) ** 0.5
    print(f"  Bob's new links (his local view):        open slot {mean(bobnew['open']):.3f}   "
          f"closed slot {mean(bobnew['closed']):.3f}   (paired {md / se if se else 0:+.1f} sigma)")


if __name__ == "__main__":
    main()
