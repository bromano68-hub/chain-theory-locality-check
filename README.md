# A locality check of Chain Theory's join rules

An independent test of the network moves in A. D. Wissner-Gross's
*Chain Theory* compendium (release 0.1, 23 October 2026), using the author's own
rules and data from [github.com/alexwg/chain-theory](https://github.com/alexwg/chain-theory).

**Main finding.** Under the 38 acts of record, the free slot at one site (a purely
local state) controls the odds of a (2,2) join to a site in a *separate* web. At the
author's 19 particle webs, 80% of admitted moves at the seed are joins across webs
(X01–X06). Nothing detectable changes in the far site's own link count, and an exact
quantum check shows no signal through any two-end act while the axiom holds.
This bears on Open results I.3.1 and I.4.1 and on the empty chapter I.8.

## Results

| Test | Result |
| --- | --- |
| Port check | All 43 sink webs in `Sinks.lean` reproduce their recorded genre census; 120/120 forward moves undone by an inverse of their kind |
| Admitted moves at the seed (163 sites, 19 webs) | 1,997, of which 1,588 (80%) are X01–X06 joins across webs; the webs fuse into 1–2 within ~100 moves |
| Within-web joins at distance ≥ 3 | 70% at the seed, then 97–100% |
| Cause and effect: fill one free slot at a qubit | First effect in a *different* particle web in 103 of 109 trials; with J01–J06 only, 0 of 23 |
| Control: electron `a3` slot open vs filled, hydrogen `p.s1` as the far site (3,000 paired trials × 60 moves) | Expected direct joins 0.0108 vs 0.0006 (+46 σ); far site's final link count +0.003 ± 0.009 (no detectable change) |
| Quantum check, four qudits A′–A, B–B′, d = 2, 3, 4 | With the axiom kept, no two-end act (Bell projection, pair-or-not, swap) changes the far side's view; only an unpaired qudit signals |

## Caveats

- **Scheduler.** The Lean code lists admitted instances but no rule for which move
  happens. These runs use a uniform choice among admitted moves, coupled across
  paired copies by fixed per-move keys. Different dynamics could change the numbers.
- **Seed.** The 444-site universe web (Instrument I.2) is not published, so the seed
  is the disjoint union of the 19 particle webs in `figures/particles.json`.
- **Classical structure.** The network simulators treat the web's shape classically;
  `quantum_layer.py` treats one two-end act exactly.
- The far-site null result (+0.003 ± 0.009) cannot yet exclude an uncompensated
  shift of about +0.011; roughly 4× more trials would.

## Files

| File | What it is |
| --- | --- |
| `chain_sim_v3.py` | The 38 acts ported from `ActClosure/Web.lean` and `Data/Acts.lean`, run on the particle webs (E1–E4). `python chain_sim_v3.py` (~12 min) or `--quick` (~3 min) |
| `chain_sim_v3_large.py` | The large coupled control run. `python chain_sim_v3_large.py 3000 60 2` (~24 min on 2 cores) |
| `quantum_layer.py` | Exact four-qudit check of two-end acts (needs numpy, ~2 s) |
| `chain_sim_v2.py`, `toy_chain_sim.py` | Earlier, simpler models (random seed with the I.2 census; a toy ring) |
| `particles.json`, `Sinks.lean` | Copied unchanged from the author's repository, needed by v3 |
| `results_*.txt` | Output of each script |

Run everything with Python 3.10+; only `quantum_layer.py` needs numpy.

## Questions for the author

1. What dynamics, or rate, is intended for choosing among admitted moves?
2. Will joins, especially X01–X06, get a locality condition in a later release?
3. Is the 444-site universe web available?

## Attribution and license

`particles.json` and `Sinks.lean` are from
[alexwg/chain-theory](https://github.com/alexwg/chain-theory) (code Apache-2.0, text
CC BY 4.0), copied unchanged. The move rules in `chain_sim_v3.py` are a Python
translation of that repository's `ActClosure` Lean package. The rest of this
repository's code is released under the Apache License 2.0.

This is an independent check, not affiliated with the author or Physical
Superintelligence PBC.
