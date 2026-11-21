"""Public bridge to the compiled application-simulator runtime.

The platform-specific runtime wheel contains the private response functions.
This module deliberately exposes only the factory and public problem-name data
needed by the supplied optimization scaffold.
"""

try:
    import hw5_bo_simulator_runtime as _runtime
except ImportError as error:  # pragma: no cover - installation diagnostic
    raise ImportError(
        "The compiled HW5 simulator runtime is not installed. Follow the "
        "platform-specific installation steps in README.md."
    ) from error

if _runtime.__version__ != "0.2.0":  # pragma: no cover - installation diagnostic
    raise ImportError(
        "The HW5 simulator runtime has an incompatible version: expected "
        f"0.2.0, found {_runtime.__version__}."
    )

NAMES = _runtime.NAMES
make_problem = _runtime.make_problem
pipeline_space = _runtime.pipeline_space


__all__ = ["NAMES", "make_problem", "pipeline_space"]
