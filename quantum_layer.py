"""
quantum_layer.py - exact quantum test of a Chain Theory "long join".

The classical simulators found that the draft's long joins let a local act by
Alice change what distant Bob sees. This script asks whether that survives
when the join is a real quantum operation.

Setup (exact, no approximation): four qudits of dimension d, in this order
    A'  A   |   B   B'
    Alice's site   |   Bob's site
A' and A form one maximally entangled pair (a link at Alice's site), and B and
B' another (a link at Bob's site), as the axiom requires: every qudit is half
of a maximally entangled pair. A and B are far apart.

Step 1, Alice's local choice (acts only on A' and A):
    nothing         - do nothing
    measure Z       - measure her partner A' in the standard basis
    measure X       - measure her partner A' in the Fourier basis
    reset           - throw A away and put in a fresh |0> (cut the link,
                      "free the slot")
    rotate          - apply a fixed unitary to A

Step 2, a long join between A and B (a two-end act). Four versions:
    Bell, all outcomes      - project A,B onto the d^2 maximally entangled pairs
                              (the draft's act read as a channel, summed over
                              outcomes; "branch probability 1/d^2")
    Bell, outcome at Bob    - same, but the outcome is recorded at Bob's end
                              (the draft's "pairing record ... at the two ends")
    pair or not, flag at Bob- the rank-one projection onto ONE pair vs. the
                              rest, and Bob sees whether a link formed
    swap                    - a joint unitary that exchanges A and B
                              (reference case: known to signal)

Bob's view = his two qudits B, B' plus any record left at his end.
Signal = trace distance between Bob's view after Alice's "nothing" and after
her other choice. 0 means Bob learns nothing; 1 means he can tell perfectly.

Usage: python quantum_layer.py
"""

import itertools
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def ket(d, i):
    v = np.zeros(d, dtype=complex)
    v[i] = 1
    return v


def phi(d):
    """Maximally entangled pair (1/sqrt d) sum_i |i>|i>."""
    return sum(np.kron(ket(d, i), ket(d, i)) for i in range(d)) / np.sqrt(d)


def X(d):
    return np.roll(np.eye(d), 1, axis=0).astype(complex)


def Z(d):
    w = np.exp(2j * np.pi / d)
    return np.diag([w ** k for k in range(d)])


def F(d):
    w = np.exp(2j * np.pi / d)
    return np.array([[w ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d)


def bell_basis(d):
    """The d^2 generalized Bell states (X^j Z^k ⊗ 1)|phi>."""
    I = np.eye(d)
    out = []
    for j, k in itertools.product(range(d), repeat=2):
        U = np.linalg.matrix_power(X(d), j) @ np.linalg.matrix_power(Z(d), k)
        out.append(np.kron(U, I) @ phi(d))
    return out


def op_on(d, n, ops):
    """Tensor product placing ops (dict position->matrix) on n qudits."""
    m = np.array([[1]], dtype=complex)
    for i in range(n):
        m = np.kron(m, ops.get(i, np.eye(d)))
    return m


def two_site_op(d, M, i, j, n=4):
    """Embed a d^2 x d^2 operator M acting on qudits i<j (adjacent here: 1,2)."""
    assert (i, j) == (1, 2)
    return np.kron(np.kron(np.eye(d), M), np.eye(d))


def ptrace(rho, d, keep, n=4):
    """Partial trace keeping the listed qudits."""
    rho = rho.reshape([d] * (2 * n))
    tr = [i for i in range(n) if i not in keep]
    # trace out from the highest index down
    for q in sorted(tr, reverse=True):
        nn = rho.ndim // 2
        rho = np.trace(rho, axis1=q, axis2=q + nn)
    k = len(keep)
    return rho.reshape(d ** k, d ** k)


def trace_distance(r, s):
    ev = np.linalg.eigvalsh((r - s + (r - s).conj().T) / 2)
    return 0.5 * np.sum(np.abs(ev))


# ---- Alice's local choices ----------------------------------------------------
def alice(rho, d, choice):
    if choice == "nothing":
        return rho
    if choice in ("measure Z", "measure X"):
        B = np.eye(d) if choice == "measure Z" else F(d)
        out = np.zeros_like(rho)
        for i in range(d):
            v = B[:, i]
            P = op_on(d, 4, {0: np.outer(v, v.conj())})
            out += P @ rho @ P
        return out
    if choice == "reset":
        # trace out A (qudit 1) and replace with |0><0|
        r = rho.reshape([d] * 8)
        r = np.trace(r, axis1=1, axis2=5)          # now A' B B' ; A' B B'
        r = r.reshape(d ** 3, d ** 3)
        zero = np.outer(ket(d, 0), ket(d, 0))
        # reinsert at position 1: build A' ⊗ A ⊗ (B B') ordering
        r = r.reshape(d, d ** 2, d, d ** 2)
        out = np.einsum("aibj,xy->axibyj", r, zero).reshape(d ** 4, d ** 4)
        return out
    if choice == "rotate":
        rng = np.random.default_rng(1)
        M = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        U, _ = np.linalg.qr(M)
        W = op_on(d, 4, {1: U})
        return W @ rho @ W.conj().T
    raise ValueError(choice)


# ---- the long join and Bob's view -----------------------------------------------
def bob_view(rho, d, join):
    """Return Bob's view as a block-diagonal matrix: one block per record value
    Bob can see at his end, each block the (unnormalized) state of B, B'."""
    keep = [2, 3]
    if join == "none":
        return [ptrace(rho, d, keep)]
    if join == "Bell, all outcomes":
        total = np.zeros((d * d, d * d), dtype=complex)
        for b in bell_basis(d):
            P = two_site_op(d, np.outer(b, b.conj()), 1, 2)
            total += ptrace(P @ rho @ P, d, keep)
        return [total]
    if join == "Bell, outcome at Bob":
        blocks = []
        for b in bell_basis(d):
            P = two_site_op(d, np.outer(b, b.conj()), 1, 2)
            blocks.append(ptrace(P @ rho @ P, d, keep))
        return blocks
    if join == "pair or not, flag at Bob":
        b = phi(d)
        P = two_site_op(d, np.outer(b, b.conj()), 1, 2)
        Q = np.eye(d ** 4) - P
        return [ptrace(P @ rho @ P, d, keep), ptrace(Q @ rho @ Q, d, keep)]
    if join == "swap":
        S = np.zeros((d * d, d * d), dtype=complex)
        for i, j in itertools.product(range(d), repeat=2):
            S[j * d + i, i * d + j] = 1
        W = two_site_op(d, S, 1, 2)
        return [ptrace(W @ rho @ W.conj().T, d, keep)]
    raise ValueError(join)


def view_distance(v1, v2):
    """Trace distance between two classical-quantum views (block lists)."""
    return sum(trace_distance(a, b) for a, b in zip(v1, v2))


def initial(d):
    p = phi(d)
    psi = np.kron(p, p)                    # (A' A) ⊗ (B B')
    return np.outer(psi, psi.conj())


CHOICES = ["measure Z", "measure X", "reset", "rotate"]
JOINS = ["none", "Bell, all outcomes", "Bell, outcome at Bob",
         "pair or not, flag at Bob", "swap"]


def flag_probability(d, choice):
    rho = alice(initial(d), d, choice)
    b = phi(d)
    P = two_site_op(d, np.outer(b, b.conj()), 1, 2)
    return np.real(np.trace(P @ rho))


def main():
    for d in (2, 3, 4):
        print(f"=== qudit dimension d = {d}  (join genre ({d},{d})) ===")
        print(f"{'long join':28} | " + " | ".join(f"{c:>9}" for c in CHOICES) + " | verdict")
        print("-" * 95)
        rho0 = initial(d)
        for join in JOINS:
            base = bob_view(alice(rho0, d, "nothing"), d, join)
            row = []
            for c in CHOICES:
                v = bob_view(alice(rho0, d, c), d, join)
                row.append(view_distance(base, v))
            verdict = "SIGNAL" if max(row) > 1e-9 else "no signal"
            print(f"{join:28} | " + " | ".join(f"{x:9.4f}" for x in row) + f" | {verdict}")
        print(f"Chance a link forms (pair-or-not): "
              + ", ".join(f"{c}: {flag_probability(d, c):.4f}" for c in ["nothing"] + CHOICES)
              + f"   (1/d^2 = {1 / d ** 2:.4f})\n")


# ---- general check: what decides whether Alice can signal? ----------------------
def random_unitary(n, rng):
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(M)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def general_check(d=2, samples=300, seed=5):
    """Alice applies random operations at her own site:
      kind 1: separate unitaries on A' and on A (keeps the A'-A pair maximally
              entangled, as the axiom requires)
      kind 2: one joint unitary on A' and A together (can weaken or break the
              pair, which the axiom forbids)
    For each, measure how far A's own state is from maximally mixed, and the
    largest signal Bob can get over three two-end joins (Bell outcome at Bob,
    pair-or-not with flag, swap). Also random two-end unitaries on A,B."""
    rng = np.random.default_rng(seed)
    rho0 = initial(d)
    I_d = np.eye(d) / d
    joins = ["Bell, outcome at Bob", "pair or not, flag at Bob", "swap"]
    base = {j: bob_view(rho0, d, j) for j in joins}
    results = {"separate (axiom kept)": [], "joint (axiom broken)": []}
    for kind in results:
        for _ in range(samples):
            if kind.startswith("separate"):
                W = op_on(d, 4, {0: random_unitary(d, rng), 1: random_unitary(d, rng)})
            else:
                W = np.kron(random_unitary(d * d, rng), np.eye(d * d))
            rho = W @ rho0 @ W.conj().T
            rA = ptrace(rho, d, [1])
            dev = trace_distance(rA, I_d)
            sig = max(view_distance(base[j], bob_view(rho, d, j)) for j in joins)
            results[kind].append((dev, sig))
    # random two-end unitaries on A,B as the join, Alice keeps the axiom
    rand_join = []
    for _ in range(samples):
        U = two_site_op(d, random_unitary(d * d, rng), 1, 2)
        W = op_on(d, 4, {0: random_unitary(d, rng), 1: random_unitary(d, rng)})
        r1 = U @ rho0 @ U.conj().T
        r2 = U @ (W @ rho0 @ W.conj().T) @ U.conj().T
        rand_join.append(trace_distance(ptrace(r1, d, [2, 3]), ptrace(r2, d, [2, 3])))
    return results, rand_join


def report_general():
    print("=== General check (d = 2, 300 random cases each) ===")
    results, rand_join = general_check()
    for kind, rows in results.items():
        devs = [r[0] for r in rows]
        sigs = [r[1] for r in rows]
        zero_dev = [s for dv, s in rows if dv < 1e-9]
        print(f"Alice's {kind:23}: A's state off maximally mixed by up to {max(devs):.4f}; "
              f"Bob's best signal up to {max(sigs):.4f}")
        if kind.startswith("joint"):
            corr = np.corrcoef(devs, sigs)[0, 1]
            print(f"    signal vs how far A is from maximally mixed: correlation {corr:+.3f}")
        if zero_dev:
            print(f"    cases with A exactly maximally mixed: {len(zero_dev)}, max signal {max(zero_dev):.2e}")
    print(f"Random two-end unitary joins, Alice keeps the axiom: max signal {max(rand_join):.2e}")


if __name__ == "__main__":
    main()
    report_general()
