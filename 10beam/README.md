# 10-beam example

This case is a six-node, ten-member planar steel truss with four grouped
categorical variables: horizontal, vertical, rising-diagonal, and
falling-diagonal members. Nodes 0 and 3 are fixed, a 10 kN downward force acts
at the lower-right node, and self-weight is included. Each variable selects one
of 54 nominal I-sections.

## Uncertainty model

Every MC-FEA realization independently samples:

- I-section dimensions `b`, `h`, `tf`, and `tw` with 5% coefficient of
  variation, followed by recomputation of `A`, `Iy`, and `Iz`;
- Young's modulus with 5% coefficient of variation;
- external load with 1% coefficient of variation; and
- every member length with 0.1% coefficient of variation.

All distributions are normal and the four source classes are mutually
independent. The robust mass limit is 240 kg.

## Reproduce the paper protocol

From `core/`:

```bash
python main_cobalt.py \
  --seed 42 --n-init 15 --n-iter 185 --n-mc 500 \
  --n-validation 3000 --gamma 1.0 --k 5 --m 2 \
  --output ../results/cobalt_paper_10beam_seed42.json
```

This performs 200 counted catalogue-level evaluations. The entry point uses
the random-tree Additive-SAAS-GP/NUTS surrogate and graph-only trust-region
search defined in the paper.

## Core files

- `tenbar.py`: topology, four-source uncertainty sampler, and FEA oracle.
- `section_catalog.py`: 54-section nominal I-section catalogue.
- `cobalt_engine.py`: random trees, additive SAAS-GP, NUTS, LCB, and graph
  trust-region search.
- `run_cobalt.py`: robust MC-FEA, heteroscedastic variance, checkpoints, and
  independent validation.
- `main_cobalt.py`: 10-beam command-line entry point.
- `test_cobalt_engine.py`: fast algorithm-invariant tests.
