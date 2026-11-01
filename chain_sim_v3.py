"""
chain_sim_v3.py - Chain Theory network moves, ported line by line from the
author's own Lean code, run on the author's own published particle webs.

Sources (github.com/alexwg/chain-theory, release 0.1, code Apache-2.0, text CC BY 4.0):
  compendium/lean/ActClosure/ActClosure/Web.lean       move definitions and admission
  compendium/lean/ActClosure/ActClosure/Data/Acts.lean the 38 acts of record
  compendium/lean/ActClosure/ActClosure/Data/Sinks.lean sink webs (used to check the port)
  compendium/figures/particles.json                    19 particle webs (the seed here)

The 38 acts of record (Acts.lean, actsOfRecord):
  A01-A06   attach a fresh site at a free slot, one per genre
  J01-J06   join two sites of the SAME web: free slot at both, not adjacent.
            No distance condition.
  X01-X06   join two sites of DIFFERENT webs: free slot at both.
  H01       hub-form: two adjacent shared qubits become one ququart
  ...I      the inverse of each (remove a leaf; cut a non-bridge or a bridge
            pair; split a hub)

What is still NOT the author's:
  * The scheduler. The Lean code lists admitted instances (Act.instances) but
    no rule says which one happens. Here: one admitted instance, uniformly at
    random, over all 38 acts. A site-keyed variant is used for the cause-and-
    effect test.
  * The seed. The 444-site "universe web" is not published, so the seed is the
    disjoint union of the 19 published particle webs (163 sites, 174 links).
  * Quantum amplitudes (see quantum_layer.py for that side).

Usage: python chain_sim_v3.py [--quick]
       (needs particles.json and, for the check, Sinks.lean in the same folder)
"""

import json
import os
import random
import re
import sys
from collections import deque
from statistics import mean, pstdev

HERE = os.path.dirname(os.path.abspath(__file__))
GENRES = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]


def genre_of(a, b):
    return (a, b) if a <= b else (b, a)


# ---- the Web, as in Web.lean ------------------------------------------------------
class Web:
    """A web as in Web.lean: site dimensions and a list of links (pairs).
    Derived data (valence, neighbours, components, bridges) is cached; a Web
    is never changed in place, every move returns a new Web."""
    __slots__ = ("dims", "links", "tags", "_val", "_adj", "_lab", "_br")

    def __init__(self, dims, links, tags=None):
        self.dims = list(dims)
        self.links = list(links)
        # permanent site labels (not in the Lean; used only to follow sites
        # through the renumbering that removeSite does)
        self.tags = list(tags) if tags is not None else list(range(len(self.dims)))
        self._val = self._adj = self._lab = self._br = None

    def copy(self):
        return Web(self.dims, self.links, self.tags)

    def index_of(self, tag):
        try:
            return self.tags.index(tag)
        except ValueError:
            return None

    @property
    def sites(self):
        return len(self.dims)

    def _build(self):
        n = self.sites
        self._val = [0] * n
        self._adj = [set() for _ in range(n)]
        for (a, b) in self.links:
            self._val[a] += 1
            self._val[b] += 1
            self._adj[a].add(b)
            self._adj[b].add(a)

    def valence(self, i):
        if self._val is None:
            self._build()
        return self._val[i]

    def has_free(self, i):
        return self.valence(i) < self.dims[i]

    def adjacent(self, i, j):
        if self._adj is None:
            self._build()
        return j in self._adj[i]

    def neighbours(self, i):
        if self._adj is None:
            self._build()
        return list(self._adj[i])

    def adj(self):
        if self._adj is None:
            self._build()
        return self._adj

    def comp(self):
        """Component label of every site (Lean: sameWeb)."""
        if self._lab is None:
            adj = self.adj()
            lab = [-1] * self.sites
            c = 0
            for s in range(self.sites):
                if lab[s] < 0:
                    lab[s] = c
                    q = deque([s])
                    while q:
                        x = q.popleft()
                        for y in adj[x]:
                            if lab[y] < 0:
                                lab[y] = c
                                q.append(y)
                    c += 1
            self._lab = lab
        return self._lab

    def bridges(self):
        """Set of links whose removal disconnects their ends (Lean: isBridge),
        by Tarjan's low-link method."""
        if self._br is None:
            adj = self.adj()
            n = self.sites
            disc, low = [-1] * n, [0] * n
            br, t = set(), [0]
            for root in range(n):
                if disc[root] >= 0:
                    continue
                stack = [(root, -1, iter(adj[root]))]
                disc[root] = low[root] = t[0]
                t[0] += 1
                while stack:
                    v, parent, it = stack[-1]
                    advanced = False
                    for u in it:
                        if u == parent:
                            continue
                        if disc[u] < 0:
                            disc[u] = low[u] = t[0]
                            t[0] += 1
                            stack.append((u, v, iter(adj[u])))
                            advanced = True
                            break
                        low[v] = min(low[v], disc[u])
                    if not advanced:
                        stack.pop()
                        if stack:
                            p = stack[-1][0]
                            low[p] = min(low[p], low[v])
                            if low[v] > disc[p]:
                                br.add((min(p, v), max(p, v)))
            self._br = br
        return self._br

    def is_bridge(self, i, j):
        return self.adjacent(i, j) and (min(i, j), max(i, j)) in self.bridges()

    def erase_link(self, i, j):
        return Web(self.dims, [l for l in self.links
                               if not ((l[0] == i and l[1] == j) or (l[0] == j and l[1] == i))], self.tags)

    def is_leaf(self, i):
        return self.valence(i) == 1

    def remove_site(self, k):
        sh = lambda i: i - 1 if k < i else i
        return Web(self.dims[:k] + self.dims[k + 1:],
                   [(sh(a), sh(b)) for (a, b) in self.links if a != k and b != k],
                   self.tags[:k] + self.tags[k + 1:])

    def census(self):
        c = {g: 0 for g in GENRES}
        for (a, b) in self.links:
            c[genre_of(self.dims[a], self.dims[b])] += 1
        return [c[g] for g in GENRES]

    def canonical(self):
        lp = []
        for (a, b) in self.links:
            x, y = (a, b) if self.dims[a] <= self.dims[b] else (b, a)
            lp.append((self.dims[x], self.valence(x), self.dims[y], self.valence(y)))
        return (sorted(self.dims),
                sorted(genre_of(self.dims[a], self.dims[b]) for a, b in self.links),
                sorted(lp))


# ---- the acts of record, as in Acts.lean ---------------------------------------------
class Act:
    def __init__(self, label, kind, direction, genre):
        self.label, self.kind, self.dir, self.genre = label, kind, direction, genre

    def __repr__(self):
        return self.label


def acts_of_record():
    acts = []
    for k, g in enumerate(GENRES, 1):
        acts.append(Act(f"A0{k}", "attach", "fwd", g))
    for k, g in enumerate(GENRES, 1):
        acts.append(Act(f"J0{k}", "joinWithin", "fwd", g))
    for k, g in enumerate(GENRES, 1):
        acts.append(Act(f"X0{k}", "joinAcross", "fwd", g))
    acts.append(Act("H01", "hubForm", "fwd", (2, 2)))
    for k, g in enumerate(GENRES, 1):
        acts.append(Act(f"A0{k}I", "attach", "inv", g))
    for k, g in enumerate(GENRES, 1):
        acts.append(Act(f"J0{k}I", "joinWithin", "inv", g))
    for k, g in enumerate(GENRES, 1):
        acts.append(Act(f"X0{k}I", "joinAcross", "inv", g))
    acts.append(Act("H01I", "hubForm", "inv", (2, 2)))
    return acts


ACTS = acts_of_record()
assert len(ACTS) == 38


def admitted(a, w, x, y, lab=None):
    """Act.admitted, clause for clause from Web.lean."""
    n = w.sites
    if a.kind == "attach" and a.dir == "fwd":
        return x < n and y in (2, 3, 4) and genre_of(w.dims[x], y) == a.genre and w.has_free(x)
    if a.kind == "attach" and a.dir == "inv":
        return (x < n and y < n and x != y and genre_of(w.dims[x], w.dims[y]) == a.genre
                and w.is_leaf(y) and w.adjacent(x, y) and 2 <= w.valence(x))
    if a.kind in ("joinWithin", "joinAcross") and a.dir == "fwd":
        if not (x < n and y < n and x != y and genre_of(w.dims[x], w.dims[y]) == a.genre
                and w.has_free(x) and w.has_free(y)):
            return False
        lab = lab or w.comp()
        same = lab[x] == lab[y]
        if a.kind == "joinWithin":
            return (not w.adjacent(x, y)) and same
        return not same
    if a.kind in ("joinWithin", "joinAcross") and a.dir == "inv":
        if not (x < n and y < n and x != y and genre_of(w.dims[x], w.dims[y]) == a.genre
                and w.adjacent(x, y) and 2 <= w.valence(x) and 2 <= w.valence(y)):
            return False
        br = w.is_bridge(x, y)
        return (not br) if a.kind == "joinWithin" else br
    if a.kind == "hubForm" and a.dir == "fwd":
        if not (x < n and y < n and x != y and w.dims[x] == 2 and w.dims[y] == 2
                and w.adjacent(x, y) and 2 <= w.valence(x) and 2 <= w.valence(y)
                and a.genre == (2, 2)):
            return False
        nb = set(w.neighbours(y))
        if any(z != y and z in nb for z in w.neighbours(x)):
            return False
        return w.valence(x) + w.valence(y) <= 6
    if a.kind == "hubForm" and a.dir == "inv":
        return x < n and w.dims[x] == 4 and w.valence(x) <= 2 and a.genre == (2, 2)
    raise ValueError(a)


_TAG = [10**6]


def new_tag(w):
    _TAG[0] += 1
    return _TAG[0]


def rewrite(a, w, x, y):
    """Act.rewrite, from Web.lean."""
    if a.kind == "attach" and a.dir == "fwd":
        return Web(w.dims + [y], w.links + [(x, w.sites)], w.tags + [new_tag(w)])
    if a.kind == "attach" and a.dir == "inv":
        return w.erase_link(x, y).remove_site(y)
    if a.kind in ("joinWithin", "joinAcross") and a.dir == "fwd":
        return Web(w.dims, w.links + [(x, y)], w.tags)
    if a.kind in ("joinWithin", "joinAcross") and a.dir == "inv":
        return w.erase_link(x, y)
    if a.kind == "hubForm" and a.dir == "fwd":
        w1 = w.erase_link(x, y)
        dims = list(w1.dims)
        dims[x] = 4
        rp = lambda l: (x if l[0] == y else l[0], x if l[1] == y else l[1])
        return Web(dims, [rp(l) for l in w1.links], w1.tags).remove_site(y)
    if a.kind == "hubForm" and a.dir == "inv":
        h = x
        ls = [l for l in w.links if l[0] == h or l[1] == h]
        others = [l for l in w.links if not (l[0] == h or l[1] == h)]
        b = w.sites
        rp = lambda l: (b if l[0] == h else l[0], b if l[1] == h else l[1])
        dims = list(w.dims)
        dims[h] = 2
        return Web(dims + [2], others + ls[:1] + [rp(l) for l in ls[1:]] + [(h, b)], w.tags + [new_tag(w)])
    raise ValueError(a)


def args(a, w):
    """Act.args, from Web.lean."""
    n = w.sites
    if a.kind == "attach" and a.dir == "fwd":
        return [(q, d) for q in range(n) for d in (2, 3, 4)]
    if a.kind == "attach" and a.dir == "inv":
        return [(q, r) for q in range(n) for r in range(n)]
    if a.kind == "hubForm" and a.dir == "inv":
        return [(h, 0) for h in range(n)]
    return [(q, r) for q in range(n) for r in range(q + 1, n)]


def instances(w, acts=ACTS):
    """All admitted (act, x, y), computed efficiently but with the same result
    as filtering Act.args by Act.admitted."""
    lab = w.comp()
    val = [0] * w.sites
    for (u, v) in w.links:
        val[u] += 1
        val[v] += 1
    free = [i for i in range(w.sites) if val[i] < w.dims[i]]
    out = []
    for a in acts:
        if a.kind in ("joinWithin", "joinAcross") and a.dir == "fwd":
            for i, q in enumerate(free):
                for r in free[i + 1:]:
                    if admitted(a, w, q, r, lab):
                        out.append((a, q, r))
        elif a.kind == "attach" and a.dir == "fwd":
            for q in free:
                for d in (2, 3, 4):
                    if genre_of(w.dims[q], d) == a.genre:
                        out.append((a, q, d))
        elif a.kind == "attach" and a.dir == "inv":
            for (u, v) in w.links:
                for (q, r) in ((u, v), (v, u)):
                    if admitted(a, w, q, r):
                        out.append((a, q, r))
        elif a.kind in ("joinWithin", "joinAcross", "hubForm") and a.dir in ("inv", "fwd") \
                and not (a.kind == "hubForm" and a.dir == "inv"):
            for (u, v) in w.links:
                q, r = min(u, v), max(u, v)
                if admitted(a, w, q, r):
                    out.append((a, q, r))
        elif a.kind == "hubForm" and a.dir == "inv":
            for h in range(w.sites):
                if admitted(a, w, h, 0):
                    out.append((a, h, 0))
    return out


# ---- the seed: the author's 19 particle webs ------------------------------------------
def load_particles(path=os.path.join(HERE, "particles.json")):
    data = json.load(open(path))
    dims, links, owner, ids = [], [], [], []
    for p in data["particles"]:
        base = len(dims)
        idx = {}
        for s in p["sites"]:
            idx[s["id"]] = len(dims)
            dims.append(s["d"])
            owner.append(f"{p['name']} ({p['web']})")
            ids.append(s["id"])
        for l in p["links"]:
            links.append((idx[l["a"]], idx[l["b"]]))
    return Web(dims, links), owner, ids


def find_site(owner, ids, name, sid):
    for i, (o, s) in enumerate(zip(owner, ids)):
        if o.startswith(name) and s == sid:
            return i
    raise KeyError((name, sid))


# ---- check the port against the Lean data -----------------------------------------------
def check_port(rng):
    lines = []
    path = os.path.join(HERE, "Sinks.lean")
    ok = tot = 0
    if os.path.exists(path):
        txt = open(path, encoding="utf-8").read()
        for m in re.finditer(r"⟨(\d+), \"[0-9a-f]+\", \[([\d, ]+)\], ⟨\[([\d, ]*)\], \[([^\]]*)\]⟩⟩", txt):
            mem = [int(x) for x in m.group(2).split(",")]
            dims = [int(x) for x in m.group(3).split(",")] if m.group(3).strip() else []
            links = [tuple(int(v) for v in t) for t in re.findall(r"\((\d+), (\d+)\)", m.group(4))]
            tot += 1
            ok += Web(dims, links).census() == mem
        lines.append(f"  Sink webs from Sinks.lean: genre census matches the recorded membership at {ok} of {tot}")
    # inverse round trips on the seed and a grown web
    w, _, _ = load_particles()
    rt_ok = rt_tot = 0
    for _ in range(120):
        inst = [t for t in instances(w) if t[0].dir == "fwd"]
        a, x, y = rng.choice(inst)
        v = rewrite(a, w, x, y)
        inv = [t for t in instances(v) if t[0].dir == "inv" and t[0].kind == a.kind]
        back = any(rewrite(b, v, p, q).canonical() == w.canonical() for (b, p, q) in inv)
        rt_tot += 1
        rt_ok += back
        w = v
    lines.append(f"  Forward move then some inverse of the same kind restores the web: {rt_ok} of {rt_tot}")
    return lines


# ---- schedulers -----------------------------------------------------------------------
def step_uniform(w, rng, acts=ACTS):
    inst = instances(w, acts)
    if not inst:
        return w, None
    a, x, y = rng.choice(inst)
    return rewrite(a, w, x, y), (a, x, y)


def step_sitekeyed(w, rng, acts, n_fixed=None):
    """Same 3 random draws every step (act, first site, partner), so two copies
    of a web stay in step and differ only where a local difference matters."""
    a = acts[int(rng.random() * len(acts))]
    fx = rng.random()
    fy = rng.random()
    n = w.sites
    x = int(fx * (n_fixed or n))   # first site drawn from a fixed range, so the
    lab = w.comp()                 # two copies pick the same site
    if a.kind in ("joinWithin", "joinAcross") and a.dir == "fwd":
        partners = [r for r in range(n) if r != x and admitted(a, w, min(x, r), max(x, r), lab)]
        if not partners:
            return w, None
        r = partners[int(fy * len(partners))]
        return rewrite(a, w, min(x, r), max(x, r)), (a, x, r)
    if a.kind == "attach" and a.dir == "fwd":
        opts = [d for d in (2, 3, 4) if admitted(a, w, x, d)]
        if not opts:
            return w, None
        d = opts[int(fy * len(opts))]
        return rewrite(a, w, x, d), (a, x, d)
    return w, None


# ---- experiments -----------------------------------------------------------------------
def dist_from(w, s):
    adj = w.adj()
    d = {s: 0}
    q = deque([s])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y not in d:
                d[y] = d[x] + 1
                q.append(y)
    return d


def join_profile(w):
    inst = instances(w)
    within = [(x, y) for (a, x, y) in inst if a.kind == "joinWithin" and a.dir == "fwd"]
    across = [(x, y) for (a, x, y) in inst if a.kind == "joinAcross" and a.dir == "fwd"]
    ds = []
    cache = {}
    for (x, y) in within:
        if x not in cache:
            cache[x] = dist_from(w, x)
        ds.append(cache[x][y])
    return len(inst), within, across, ds


def e1_e2(rng, steps_list):
    print("E1/E2  Joins the author's rules admit, at the seed and as the web evolves")
    print(f"{'step':>5} | {'sites':>5} | {'webs':>4} | {'all moves':>9} | {'joins within':>12} | "
          f"{'dist>=3':>7} | {'mean dist':>9} | {'joins across':>12}")
    print("-" * 86)
    w, _, _ = load_particles()
    done = 0
    for target in steps_list:
        while done < target:
            w, _ = step_uniform(w, rng)
            done += 1
        n, within, across, ds = join_profile(w)
        webs = len(set(w.comp()))
        share = (sum(1 for d in ds if d >= 3) / len(ds)) if ds else 0
        print(f"{done:5d} | {w.sites:5d} | {webs:4d} | {n:9d} | {len(within):12d} | {share:7.0%} | "
              f"{(mean(ds) if ds else 0):9.1f} | {len(across):12d}")
    print("  'joins across' link two webs with no path between them at all.\n")


def e3_influence(rng, trials, steps, acts, label):
    """Copy A: as is. Copy B: Alice's site has one free slot filled first by an
    attach (a purely local move). Same draws after that. Where is the first
    difference among the original sites, measured in the seed?"""
    w0, owner, ids = load_particles()
    free_q = [i for i in range(w0.sites) if w0.dims[i] == 2 and w0.has_free(i)]
    hits = []
    for _ in range(trials):
        alice = rng.choice(free_q)
        a, b = w0.copy(), w0.copy()
        b = rewrite(ACTS[1], b, alice, 3) if admitted(ACTS[1], b, alice, 3) else b  # A02: attach (2,3)
        d0 = dist_from(w0, alice)
        s = rng.randrange(10**12)
        ra, rb = random.Random(s), random.Random(s)
        n0 = w0.sites
        for t in range(1, steps + 1):
            a, _ = step_sitekeyed(a, ra, acts, n0)
            b, _ = step_sitekeyed(b, rb, acts, n0)
            na = [set(nb for nb in a.neighbours(x) if nb < n0) for x in range(n0)]
            nbb = [set(nb for nb in b.neighbours(x) if nb < n0) for x in range(n0)]
            diff = [x for x in range(n0) if x != alice and na[x] != nbb[x]]
            if diff:
                far = [d0.get(x) for x in diff]
                hits.append((t, far, [owner[x] != owner[alice] for x in diff]))
                break
    other_web = sum(1 for h in hits if any(h[2]))
    same_dists = [d for h in hits for d, ow in zip(h[1], h[2]) if not ow and d is not None]
    print(f"  {label:38}: effect appeared in {len(hits)}/{trials} trials; "
          f"in a DIFFERENT web in {other_web}/{len(hits) if hits else 0}; "
          f"same-web distances {sorted(same_dists)[:12]}{'...' if len(same_dists) > 12 else ''}")


def e4_control(rng, trials, steps):
    """Alice = a free read qubit of the electron web; Bob = a free qubit of the
    hydrogen web (a different web: no path between them). Alice's local act:
    keep her slot open, or fill it with an attach at her own site (A02).
    For each step we add the exact chance that the uniformly chosen move is the
    direct Alice-Bob join, which gives the same expected value as counting the
    rare events, with far less noise."""
    w0, owner, ids = load_particles()
    alice = find_site(owner, ids, "electron", "a3")
    bob = find_site(owner, ids, "hydrogen", "p.s1")
    exp_link = {"open": [], "closed": []}
    bob_val = {"open": [], "closed": []}
    for _ in range(trials):
        s = rng.randrange(10**12)
        for mode in ("open", "closed"):
            w = w0.copy()
            if mode == "closed":
                w = rewrite(ACTS[1], w, alice, 3)       # A02 at Alice's own site
            ta, tb = w.tags[alice], w.tags[bob]
            r = random.Random(s)
            acc = 0.0
            for _ in range(steps):
                inst = instances(w)
                ia, ib = w.index_of(ta), w.index_of(tb)
                if ia is not None and ib is not None and inst:
                    q, p = min(ia, ib), max(ia, ib)
                    hits = sum(1 for (a, x, y) in inst if a.dir == "fwd"
                               and a.kind in ("joinAcross", "joinWithin") and (x, y) == (q, p))
                    acc += hits / len(inst)
                if not inst:
                    break
                a, x, y = r.choice(inst)
                w = rewrite(a, w, x, y)
            exp_link[mode].append(acc)
            ib = w.index_of(tb)
            bob_val[mode].append(w.valence(ib) if ib is not None else 0)

    def paired(xs, ys):
        d = [x - y for x, y in zip(xs, ys)]
        se = pstdev(d) / len(d) ** 0.5
        return mean(d), (mean(d) / se if se else 0.0)

    m, z = paired(exp_link["open"], exp_link["closed"])
    mb, zb = paired(bob_val["open"], bob_val["closed"])
    print("E4  Control: Alice = electron read qubit a3, Bob = hydrogen qubit p.s1 (separate webs)")
    print(f"    {trials} paired trials x {steps} moves, uniform over all admitted moves of the 38 acts")
    print(f"  Expected number of direct Alice-Bob joins: open slot {mean(exp_link['open']):.4f}   "
          f"closed slot {mean(exp_link['closed']):.4f}   (paired {z:+.1f} sigma)")
    print(f"  Bob's link count at the end (his local view): open {mean(bob_val['open']):.3f}   "
          f"closed {mean(bob_val['closed']):.3f}   (paired {zb:+.1f} sigma)")


def main():
    quick = "--quick" in sys.argv
    rng = random.Random(20261101)
    w, owner, _ = load_particles()
    print(f"Seed: the author's 19 particle webs, {w.sites} sites, {len(w.links)} links, "
          f"{len(set(w.comp()))} webs, census {w.census()} (genres {GENRES})")
    print("Port check:")
    for line in check_port(random.Random(3)):
        print(line)
    print()
    e1_e2(rng, [0, 50, 100, 200] if quick else [0, 50, 100, 200, 400])
    print("E3  Cause and effect: fill one free slot at Alice's site, run two copies with the")
    print("    same draws, and see where the first difference appears")
    joins = [a for a in ACTS if a.kind in ("joinWithin", "joinAcross") and a.dir == "fwd"]
    within = [a for a in ACTS if a.kind == "joinWithin" and a.dir == "fwd"]
    t3, s3 = (40, 60) if quick else (120, 100)
    e3_influence(random.Random(7), t3, s3, joins, "within + across joins (author's rules)")
    e3_influence(random.Random(7), t3, s3, within, "within-web joins only")
    print()
    t4, s4 = (150, 40) if quick else (600, 40)
    e4_control(random.Random(99), t4, s4)


if __name__ == "__main__":
    main()
