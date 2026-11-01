"""
Toy Chain-Theory-inspired network simulator: can a local action at A
signal a distant observer B through long (2,2) "shortcut" joins?

This is NOT the compendium's 28-move automaton. It is a deliberately simple
model that keeps the three ingredients that matter for Open-Go-Close:

  * Space   = a ring of (2,2) links. Distance = shortest path in links.
  * Open    = a "long join": a new (2,2) link between two sites that are
              far apart. Each step, a random far pair is proposed.
  * Close   = each long link is cut with some probability per step.

Alice sits at site A and chooses a bit: her local state s_A = 0 or 1.
Bob sits at site B, half a ring away, and only looks at things at his own
site: how many long links touch B, and his distance to A.

We compare join rules (how likely a proposed join is to form):

  blind   - the join chance ignores every site's state.
  gated   - the join chance depends on the states of the two endpoints
            (each factor is local to one end), the kind of rule the draft
            leaves open.
  dark    - same as gated, but the new link is a hidden (3,3) "corridor":
            it exists, but Bob's spatial reading (2,2 only) can't see it,
            like the draft's Theorem 113.2.

If Bob's statistics differ between Alice's two choices, Alice has signaled.

Usage:  python toy_chain_sim.py
"""

import random
from collections import deque
from statistics import mean, pstdev

N_SITES = 60          # ring size
A, B = 0, 30          # Alice and Bob, half a ring apart (distance 30)
STEPS = 40            # updates per trial
P_JOIN = 0.02         # base chance a proposed far pair joins
GATE = 4.0            # how strongly a site's state boosts its joins
P_CUT = 0.10          # chance per step a long link is cut (Close)
MIN_FAR = 5           # a join only counts as "long" at distance >= 5


def ring_neighbors(i):
    return ((i - 1) % N_SITES, (i + 1) % N_SITES)


def distance(src, dst, long_links):
    """Shortest path counting ring links and visible long links."""
    adj = {i: list(ring_neighbors(i)) for i in range(N_SITES)}
    for (u, v) in long_links:
        adj[u].append(v)
        adj[v].append(u)
    seen = {src: 0}
    q = deque([src])
    while q:
        x = q.popleft()
        if x == dst:
            return seen[x]
        for y in adj[x]:
            if y not in seen:
                seen[y] = seen[x] + 1
                q.append(y)
    return None


def ring_dist(i, j):
    d = abs(i - j) % N_SITES
    return min(d, N_SITES - d)


def join_chance(rule, i, j, state):
    if rule == "blind":
        return P_JOIN
    # gated and dark: each endpoint's state boosts the chance, a local factor
    boost = (1 + GATE * state[i]) * (1 + GATE * state[j])
    return min(1.0, P_JOIN * boost / (1 + GATE))  # normalized so s=0 rarer


def run_trial(rule, alice_bit, rng):
    # Every other site gets a random hidden state; only Alice picks hers.
    state = [rng.randint(0, 1) for _ in range(N_SITES)]
    state[A] = alice_bit
    long_links = set()
    for _ in range(STEPS):
        # Close: cut existing long links
        long_links = {e for e in long_links if rng.random() > P_CUT}
        # Open: propose a random far pair
        i, j = rng.sample(range(N_SITES), 2)
        if ring_dist(i, j) >= MIN_FAR and rng.random() < join_chance(rule, i, j, state):
            long_links.add((min(i, j), max(i, j)))
    visible = set() if rule == "dark" else long_links
    links_at_B = sum(1 for e in visible if B in e)
    links_at_A = sum(1 for e in visible if A in e)
    dist_AB = distance(A, B, visible)
    return links_at_B, dist_AB, links_at_A


def paired(rule, trials, seed):
    """Run each trial twice from the same random seed, once per Alice bit,
    so the only difference between the two runs is Alice's choice."""
    master = random.Random(seed)
    d_links_B, d_dist, d_links_A = [], [], []
    raw = {0: [], 1: []}
    for _ in range(trials):
        s = master.randrange(10**12)
        out = {}
        for bit in (0, 1):
            out[bit] = run_trial(rule, bit, random.Random(s))
            raw[bit].append(out[bit])
        d_links_B.append(out[1][0] - out[0][0])
        d_dist.append(out[0][1] - out[1][1])
        d_links_A.append(out[1][2] - out[0][2])
    return raw, d_links_B, d_dist, d_links_A


def z(xs):
    m = mean(xs)
    se = pstdev(xs) / len(xs) ** 0.5
    return m, (m / se if se > 0 else 0.0)


def main():
    trials = 20000
    print(f"Ring of {N_SITES} sites; Alice at {A}, Bob at {B}; "
          f"{trials} paired trials x {STEPS} steps per rule\n")
    print(f"{'rule':6} | {'shortcuts at Alice':>22} | {'shortcuts at Bob':>20} | "
          f"{'A-B distance drop':>20} | verdict")
    print("-" * 100)
    for rule in ("blind", "gated", "dark"):
        raw, dB, dD, dA = paired(rule, trials, seed=2026)
        (mA, zA), (mB, zB), (mD, zD) = z(dA), z(dB), z(dD)
        bob_signal = abs(zB) > 4
        travel = abs(zD) > 4
        verdict = []
        verdict.append("Bob's local view: " + ("CHANGES" if bob_signal else "unchanged"))
        verdict.append("shortcut control: " + ("YES" if travel else "no"))
        print(f"{rule:6} | {mA:+.4f} ({zA:+6.1f} sig) | {mB:+.4f} ({zB:+5.1f} sig) | "
              f"{mD:+.3f} ({zD:+6.1f} sig) | " + "; ".join(verdict))
    print("\nColumns show (Alice bit 1) minus (Alice bit 0), paired by seed.")
    print("'sig' = how many standard errors the difference is from zero.")


if __name__ == "__main__":
    main()
