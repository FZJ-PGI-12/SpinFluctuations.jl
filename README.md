# Adiabatic bottlenecks in quantum annealing and nonequilibrium dynamics of paramagnons - SpinFluctuations.jl

[Phys. Rev. A __110__, 012611](https://doi.org/10.1103/PhysRevA.110.012611)

## Abstract

The correspondence between long-range interacting quantum spin glasses and combinatorial optimization problems underpins the physical motivation for adiabatic quantum computing. On one hand, in disordered (quantum) spin systems, the focus is on exact methods such as the replica trick that allow the calculation of system quantities in the limit of infinite system and ensemble size. On the other hand, when solving a given instance of an optimization problem, disorder-averaged quantities are of no relevance, as one is solely interested in instance-specific, finite-size properties, in particular the optimal solution. Here, we apply the nonequilibrium Green's function formalism to the spin coherent-state path integral to obtain the statistical fluctuations and the collective-excitation spectrum along the annealing path. For the example of the quantum Sherrington-Kirkpatrick spin glass, by comparing to extensive numerically exact results, we show that this method provides access to the instance-specific bottlenecks of the annealing protocol.


## Code

The code for regenerating the plots in the paper from the hard-instance [data](data) can be found in our [notebooks folder](notebooks).


## Citation

If you are using code from this repository, please cite our work:
```
@article{PhysRevA.110.012611,
  title = {Adiabatic bottlenecks in quantum annealing and nonequilibrium dynamics of paramagnons},
  author = {Bode, Tim and Wilhelm, Frank K.},
  journal = {Phys. Rev. A},
  volume = {110},
  issue = {1},
  pages = {012611},
  numpages = {13},
  year = {2024},
  month = {Jul},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevA.110.012611},
  url = {https://link.aps.org/doi/10.1103/PhysRevA.110.012611}
}
```

