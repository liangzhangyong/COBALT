# 105-beam COBALT benchmark

This is the paper's 6-by-6-node planar truss: 30 horizontal, 25 vertical, 25
rising-diagonal, and 25 falling-diagonal members. The six left-boundary nodes
are fixed. A 5 kN downward force acts at the lower-right node and self-weight
is included. All 105 members independently select one of 54 nominal
I-sections, giving a search space of `54^105`.

## Uncertainty model

Every MC-FEA realization independently samples:

- I-section dimensions `b`, `h`, `tf`, and `tw` with 5% coefficient of
  variation, followed by recomputation of `A`, `Iy`, and `Iz`;
- Young's modulus with 5% coefficient of variation;
- the 5000 N external load with 1% coefficient of variation; and
- all 105 member lengths with 0.1% coefficient of variation.

All distributions are normal and the four source classes are mutually
independent. The robust mass limit is 5000 kg.

## Reproduce the revised protocol

From `core/`:

```bash
python cobalt_105d_main.py \
  --seed 42 --n-init 54 --n-iter 30 --n-mc 500 \
  --n-validation 3000 --gamma 1.0 --k 7 --m 2 \
  --output ../results/cobalt_paper_105beam_seed42.json
```

The 54 counted initial evaluations are the 54 homogeneous catalogue designs;
30 sequential evaluations follow, for a total budget of 84. Every sampled
tree has 104 edges and spans all 105 independent variables.

## Core files

- `beam105.py`: 36-node/105-member topology, four-source uncertainty sampler,
  and FEA oracle.
- `section_catalog.py`: 54-section nominal I-section catalogue.
- `cobalt_engine.py`: random trees, additive SAAS-GP, NUTS, LCB, and graph
  trust-region search.
- `run_cobalt.py`: robust MC-FEA, heteroscedastic variance, checkpoints, and
  independent validation.
- `cobalt_105d_main.py`: 105-beam command-line entry point.
- `test_cobalt_engine.py`: fast algorithm-invariant tests.
