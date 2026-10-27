# Source and implementation references

Validated against Newton 1.5.2, Warp 1.16.0, Pyomo 6.10.1, USD Core 26.3,
and the bundled tetrahedral bear asset. These Python packages are published on
PyPI. Installed release source and metadata take precedence over changing
`main` branches.

- Newton diffsim bear example, material/solver/contact setup:
  https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_bear.py
  Original adapter reference commit: `516aa47f261e98d8fe0d7a52e609e950b287227b`.
  The 0.2.0 adapter has actually run against Newton 1.5.2.
- Bundled bear asset, copied unchanged from Newton 1.5.2:
  https://github.com/newton-physics/newton/blob/v1.5.2/newton/examples/assets/bear.usd
  Bundled SHA-256:
  `a2b1dbe891e8ed0d8bb6cd12a3907b37b6abbab84f64a5c398e8ebf0ea31658d`.
- Pyomo Python external-function interface:
  https://pyomo.readthedocs.io/en/latest/api/pyomo.core.base.external.PythonCallbackFunction.html
- Pyomo native reverse automatic differentiation:
  https://pyomo.readthedocs.io/en/stable/_modules/pyomo/core/expr/calculus/diff_with_pyomo.html
- IPOPT C callback interface and lifecycle:
  https://coin-or.github.io/Ipopt/INTERFACES.html
- IPOPT options and stopping/scaling controls:
  https://coin-or.github.io/Ipopt/OPTIONS.html
- IPOPT limited-memory second derivatives:
  https://coin-or.github.io/Ipopt/SPECIALS.html
- Warp tape API:
  https://nvidia.github.io/warp/modules/differentiability.html

The C binding was also checked against IPOPT 3.14.11 `IpStdCInterface.h` and
`IpoptConfig.h`. Its callbacks use standard double-precision numbers and
32-bit indices. Original native libraries are not redistributed in this
archive.
