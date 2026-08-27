# Regulatory VAE Traceability

This matrix tracks specification version 0.1. `Implemented` means production
behavior exists; `Verified` additionally requires the listed executable check
to pass. `Planned` identifies behavior that remains to be built.

| Requirement | Implementation | Executable evidence | Status |
| --- | --- | --- | --- |
| CFG-001..CFG-006 | `nocap.scanvi.config` | config unit, Hypothesis, Specsaver scenarios | Planned |
| DATA-001..DATA-009 | `nocap.scanvi.data` | data unit, Hypothesis, Specsaver scenarios | Planned |
| GRAPH-001 | `graphml_to_sparse` | `test_undirected_and_parallel_edge_graphs_are_rejected` | Verified |
| GRAPH-002..GRAPH-004 | `graphml_to_sparse` | `test_gene_order_is_authoritative_and_cycles_are_retained` | Verified |
| GRAPH-005 | graph projection and solver validation | self-loop graph and topology tests | Verified |
| GRAPH-006 | solver topology validation | duplicate topology test | Verified |
| GRAPH-007..GRAPH-008 | `graphml_to_sparse` | graph-only/isolated-gene test | Verified |
| GRAPH-009 | graph polarity and `effective_weights` | polarity tests in graph and solver suites | Verified |
| GRAPH-010 | graph projection | confidence-metadata regression | Planned |
| GRAPH-011 | `GraphAlignmentReport` | graph accounting unit and Hypothesis tests | Implemented |
| SOLVER-001..SOLVER-003 | `SparseRegulatoryLayer` validation | invalid topology/input unit tests | Implemented |
| SOLVER-004..SOLVER-006 | `effective_weights` | normalization and polarity tests | Verified |
| SOLVER-007..SOLVER-008 | `sparse_matvec` | orientation, linearity, and sparse/dense property tests | Verified |
| SOLVER-009 | solver methods | input and topology immutability property | Verified |
| SOLVER-010..SOLVER-011 | `_solve_message` | fixed recurrence and identity tests | Verified |
| SOLVER-012..SOLVER-015 | `_solve_message` | tolerance and unstable-raw tests | Verified |
| SOLVER-016 | `dense_adjacency` | dense-limit test | Verified |
| SOLVER-017 | `_solve_linalg` | message/linalg state and gradient tests | Verified |
| SOLVER-018 | `_solve_linalg` | singular-system domain-error test | Verified |
| SOLVER-019 | `EquilibriumResult` | unstable-but-solvable test | Verified |
| SOLVER-020 | `EquilibriumResult` | arbitrary leading-shape property tests | Implemented |
| SOLVER-021 | normalization and solver math | contraction property and method agreement tests | Implemented |
| SOLVER-022 | fixed message solver | Hypothesis geometric error-bound test | Verified |
| SOLVER-023 | signed-system documentation | residual non-monotonicity regression | Planned |
| MODEL-001..MODEL-009 | `nocap.scanvi.model` and training | model/guide, gradient, provenance scenarios | Planned |
| CF-001..CF-008 | `nocap.scanvi.counterfactual` | ChiRho trace and factual-world scenarios | Planned |
| PROV-001..PROV-004 | checkpoint and run metadata | checkpoint round-trip tests | Planned |
| ERR-001 | config/data errors | validation scenario outlines | Planned |
| ERR-002 | graph projection | graph rejection tests | Implemented |
| ERR-003 | solver validation | invalid solver-input tests | Implemented |
| ERR-004 | dense solver | dense-limit and singular tests | Implemented |
| ERR-005 | common result/error boundary | `specs/regulatory_solver.fizz` safety and reachability assertions | Verified |

## Planned Artifact Map

| Artifact | Requirements |
| --- | --- |
| `features/regulatory_vae/experiment_config.feature` | CFG-001..CFG-006, ERR-001 |
| `features/regulatory_vae/anndata_ingestion.feature` | DATA-001..DATA-009 |
| `features/regulatory_vae/graphml_alignment.feature` | GRAPH-001..GRAPH-011, ERR-002 |
| `features/regulatory_vae/equilibrium_solver.feature` | SOLVER-001..SOLVER-020, ERR-003..ERR-005 |
| `features/regulatory_vae/regulatory_vae.feature` | MODEL-001..MODEL-009, PROV-001..PROV-004 |
| `features/regulatory_vae/counterfactuals.feature` | CF-001..CF-008 |
| `specs/regulatory_solver.fizz` | SOLVER-010..SOLVER-020, ERR-003..ERR-005 |
| solver Hypothesis suite | SOLVER-006..SOLVER-023 |
| graph Hypothesis suite | GRAPH-002..GRAPH-011 |
