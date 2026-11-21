"""Two required homework applications and one completed diagnostic."""

PROBLEM_NAMES = ("kinematics", "phase")


def make_problem(name, *, dtype="float64", device="auto", **kwargs):
    if name == "toy":
        from .toy import QuadraticDemo

        cls = QuadraticDemo
    elif name == "kinematics":
        from .kinematics import InverseKinematics

        cls = InverseKinematics
    elif name == "phase":
        from .phase import PhaseRetrieval

        cls = PhaseRetrieval
    else:
        raise ValueError(
            f"Unknown problem {name!r}; use toy or one of {PROBLEM_NAMES}"
        )
    return cls(dtype=dtype, device=device, **kwargs)
