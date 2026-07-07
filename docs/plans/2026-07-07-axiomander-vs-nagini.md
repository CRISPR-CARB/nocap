# Axiomander vs. Nagini — Balanced Audit

Based on Axiomander's README + `CONTRACT_LANGUAGE.md` + architecture, and Nagini's contract language + hands-on verification of `scc_perturb` and `cyclic_single_door`.

---

## 1. Python Language Support

| Feature | Axiomander | Nagini |
|---|---|---|
| ints / bools | ✓ | ✓ |
| floats | ✓ (fixed-point `VFloat`, `float_scale=100` → 2 dp) | ✓ (real or IEEE32 encodings, configurable) |
| strings | ✓ **strong** — equality, contains, prefix/suffix, **regex** via `str.in_re` | ✓ (str domain) — no regex theory |
| lists | ✓ (len, index, slice-length, quantifiers) | ✓ **strong** — `list_pred`, `PSeq`, take/drop/update |
| dicts / sets | ✓ (membership, len) | ✓ **strong** — `dict_pred`, `set_pred`, `PSet`, `PMultiset` |
| tuples | value comparison | ✓ typed, but no indexed `ResultT(Tuple[...])` |
| loops | ✓ invariants (`assert` in loop body) | ✓ `Invariant()` + `Decreases()` |
| function calls / frames | ✓ explicit `reads:`/`modifies:` + decomposed CCall obligations | ✓ modular pre/post; permissions encode framing |
| classes / fields | ✓ flattened (`obj.field`→`obj_field`); Pydantic/dataclass **Shape IR** | ✓ full OO, inheritance, behavioural subtyping, `@Predicate` |
| exceptions | ✓ `raises:` outcome predicates | ✓ `Exsures()` |
| heap / aliasing / mutation | ⚠️ frame-based, no true aliasing model | ✓ **separation logic** — precise heap/aliasing |
| generics / higher-order | limited | ✓ generics, first-class quantifiers `Forall`/`Exists`/`Forall2..6` |
| **dimensional analysis** | ✓ **unique** — `units:` catches unit errors | ✗ |
| concurrency / IO | ✗ | ✓ threads, locks, IO, obligations, SIF |

**Verdict:** Nagini has broader and deeper coverage of *runtime Python semantics* — especially heap mutation, aliasing, OO, collections-as-live-objects, and concurrency — because separation logic models the actual object graph. Axiomander is competitive on pure/functional-style code and **uniquely strong** in two niches Nagini lacks entirely: **dimensional analysis** and **regex-membership contracts**.

This matches our migration experience: the nocap functions mutate `nx.DiGraph`/dict/list state and return heterogeneous dicts/tuples — squarely in Nagini's wheelhouse.

---

## 2. Ergonomics

**Axiomander advantages**
- **Zero-import / non-invasive** — contracts are plain `assert`s or docstring `axiomander:` blocks. Production code stays dependency-free (nocap's house convention). This is a genuine, significant ergonomic win.
- Contracts double as executable runtime checks (the `assert` form).
- **Incremental evidence graph** — `verify-changed`/`verify-impacted`/`explain-cache` track staleness across the callee/caller dependency graph. Nagini re-verifies per file/method with a result cache but has no first-class dependency/staleness model.
- SMT/regex **counterexamples** are typed and concrete.

**Nagini advantages**
- More **expressive, uniform** spec language in one place (`Requires`/`Ensures`/`Invariant`/`Forall`/`Old`/`Result`/predicates).
- Predicates + `Fold`/`Unfold` give real abstraction over data-structure invariants.
- Behaves like a type-checker: fast, deterministic pass/fail.

**Nagini costs (we hit these)**
- **Invasive**: requires `from nagini_contracts.contracts import *`; the file won't type-check under normal Pylance/mypy without the venv. We needed a separate `*_nagini.py` companion file.
- Sharp edges: no `from __future__ import annotations`; no `Acc()` on method refs; no `list_pred(Result())` in `@Pure`; no indexed `ResultT(Tuple[...])`. External libs (networkx/y0) are opaque → must build abstract models + `@ContractOnly` stubs.

**Verdict:** Axiomander is more ergonomic for *keeping specs close to unmodified production code* and for *incremental, iterative* workflows. Nagini's language is more expressive but more invasive and has more idiosyncratic restrictions.

---

## 3. Verification Stack & Trust

| Dimension | Axiomander | Nagini |
|---|---|---|
| Pipeline | Python → IMP → Coq obligations → SMT/Hammer → **LLM oracle** | Python → Viper (Silver) → Silicon (symbolic exec) **or** Carbon (VCG/Boogie) → Z3 |
| Foundation | Coq WP calculus (custom `Imp.v`/`Wp.v`) | Separation logic (Viper), well-studied |
| Soundness story | L1 Coq-kernel-checked (high assurance); L2 SMT; **L3 LLM-repaired proofs** (assurance depends on whether the Coq kernel re-checks the LLM output) | SMT-based; sound modulo Viper/Z3 encoding; **no LLM in the trust path** — fully deterministic |
| Maturity | Young, fast-moving, research-grade; **dogfoods itself** | Mature (ETH Zürich, Eilers & Müller); large functional test suite |
| Determinism | L1/L2 deterministic; **L3 non-deterministic** (LLM) | Deterministic |
| Extra theories | String/regex (QF_SLIA), fixed-point floats, **dimensions** | Real/IEEE floats, bitvectors; SIF (security info-flow) |
| External deps | OCaml+Coq, z3/cvc5, coqpyt, LLM API key | JVM + Viper + Z3 |

**Verdict:** Nagini has the more **mature and self-contained** stack, and — importantly — its trust path is **fully deterministic** (no LLM). Axiomander's L1 obligations are Coq-kernel-checked (arguably the *highest* assurance tier of either tool), but its reliance on an LLM oracle at L3 introduces non-determinism and a trust dependency that Nagini avoids. Axiomander's incremental evidence graph is architecturally more sophisticated for large, evolving codebases.

---

## Bottom Line (for the nocap migration)

Nagini is the right tool **for these specific modules** because they are heap-mutating, OO/collection-heavy Python where separation logic pays off, and because its deterministic stack + mature toolchain lower risk. That's an honest, scoped justification — **not** a claim that Nagini dominates Axiomander universally. Axiomander remains superior for dimensional analysis, regex contracts, non-invasive/executable specs, and incremental evidence tracking; if nocap grows code needing units-checking or regex gating, Axiomander is the better fit there.

