"""Small, explicit ctypes binding to IPOPT's public C callback interface.

No CasADi expression engine, NL writer, ASL extension, or compiled Python
extension is required. Uses standard double precision / 32-bit IPOPT indices.
Supply BEAR_IPOPT_LIBRARY to select a shared library explicitly. A pre-existing
CasADi installation may supply libipopt.so, but CasADi is never imported.

Reference: https://coin-or.github.io/Ipopt/INTERFACES.html#INTERFACE_C
"""
from __future__ import annotations
import ctypes as ct
import ctypes.util
import importlib.util
import os
from pathlib import Path
import re
import sys
import numpy as np

I, D, B, V = ct.c_int, ct.c_double, ct.c_bool, ct.c_void_p
DP, IP = ct.POINTER(D), ct.POINTER(I)
F = ct.CFUNCTYPE(B, I, DP, B, DP, V)
G = ct.CFUNCTYPE(B, I, DP, B, I, DP, V)
JAC = ct.CFUNCTYPE(B, I, DP, B, I, I, IP, IP, DP, V)
HESS = ct.CFUNCTYPE(B, I, DP, B, D, I, DP, B, I, IP, IP, DP, V)
INTERMEDIATE = ct.CFUNCTYPE(B, I, I, D, D, D, D, D, D, D, D, I, V)

STATUS = {
    0: "Solve_Succeeded", 1: "Solved_To_Acceptable_Level", 2: "Infeasible_Problem_Detected",
    3: "Search_Direction_Becomes_Too_Small", 4: "Diverging_Iterates", 5: "User_Requested_Stop",
    6: "Feasible_Point_Found", -1: "Maximum_Iterations_Exceeded", -2: "Restoration_Failed",
    -3: "Error_In_Step_Computation", -4: "Maximum_CpuTime_Exceeded", -5: "Maximum_WallTime_Exceeded",
    -10: "Not_Enough_Degrees_Of_Freedom", -11: "Invalid_Problem_Definition",
    -12: "Invalid_Option", -13: "Invalid_Number_Detected", -100: "Unrecoverable_Exception",
    -101: "NonIpopt_Exception_Thrown", -102: "Insufficient_Memory", -199: "Internal_Error",
}


def find_library():
    explicit = os.environ.get("BEAR_IPOPT_LIBRARY")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"BEAR_IPOPT_LIBRARY does not exist: {path}")
        return str(path)
    prefix = Path(sys.prefix)
    for path in (
        prefix / "lib/libipopt.so",
        prefix / "lib/libipopt.dylib",
        prefix / "Library/bin/ipopt.dll",
        prefix / "Library/bin/libipopt.dll",
    ):
        if path.is_file():
            return str(path)
    # Conda packages sometimes use versioned shared-library names. Search only
    # the conventional library directories inside the active environment.
    for directory, patterns in (
        (prefix / "lib", ("libipopt.so.*", "libipopt.*.dylib")),
        (prefix / "Library/bin", ("ipopt-*.dll", "libipopt-*.dll")),
    ):
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return str(matches[0])
    found = ctypes.util.find_library("ipopt")
    if found:
        return found
    # Optional library distribution, not an optimization/modeling backend.
    spec = importlib.util.find_spec("casadi")
    if spec and spec.submodule_search_locations:
        root = Path(next(iter(spec.submodule_search_locations)))
        for name in ("libipopt.so", "libipopt.dylib", "ipopt.dll", "libipopt.dll"):
            if (root / name).is_file():
                return str(root / name)
        # CasADi's Windows wheels use a versioned DLL name such as
        # ``libipopt-3.dll``. Keep this fallback narrowly scoped to the
        # already-discovered CasADi package directory.
        versioned_windows = sorted(root.glob("libipopt-*.dll"))
        if versioned_windows:
            return str(versioned_windows[0])
    raise ImportError("IPOPT shared library missing. Install IPOPT and set BEAR_IPOPT_LIBRARY to "
                      "libipopt.so/ipopt.dll (double precision, 32-bit indices), or install cyipopt and "
                      "use engine='cyipopt'. See docs/INSTALL.md.")


def _open_library(path):
    """Load IPOPT while exposing a selected Windows DLL's dependency folder."""
    dll_directory = None
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if sys.platform == "win32" and Path(path).is_absolute() and add_dll_directory:
        dll_directory = add_dll_directory(str(Path(path).resolve().parent))
    try:
        return ct.CDLL(path)
    finally:
        if dll_directory is not None:
            dll_directory.close()


def load_library():
    path = find_library()
    # Reject a known incompatible ABI before entering native code. If headers
    # are absent the administrator must provide the documented standard ABI.
    root = Path(path).resolve().parent
    for config in (root / "include/coin-or/IpoptConfig.h", root.parent / "include/coin-or/IpoptConfig.h"):
        if config.is_file():
            text = config.read_text()
            if re.search(r"^\s*#\s*define\s+IPOPT_(?:INT64|SINGLE)\b", text, re.M):
                raise RuntimeError("The ctypes adapter requires double precision and 32-bit IPOPT indices.")
    lib = _open_library(path)
    signatures = {
        "CreateIpoptProblem": (V, [I, DP, DP, I, DP, DP, I, I, I, F, G, F, JAC, HESS]),
        "FreeIpoptProblem": (None, [V]),
        "AddIpoptStrOption": (B, [V, ct.c_char_p, ct.c_char_p]),
        "AddIpoptNumOption": (B, [V, ct.c_char_p, D]),
        "AddIpoptIntOption": (B, [V, ct.c_char_p, I]),
        "SetIntermediateCallback": (B, [V, INTERMEDIATE]),
        "IpoptSolve": (I, [V, DP, DP, DP, DP, DP, DP, V]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = restype, argtypes
    return lib, path


def _ptr(array):
    return array.ctypes.data_as(DP)


def solve(problem, initial, options):
    oracle = problem.oracle
    lib, library_path = load_library()
    n, m = problem.layout.n_controls, oracle.task.row_count - 1
    if max(n, m, n * m) > np.iinfo(np.int32).max:
        raise MemoryError("IPOPT C interface dimensions exceed its 32-bit index range.")
    x = np.ascontiguousarray(initial, dtype=np.float64).copy()
    lb, ub = problem.lower.ravel().copy(), problem.upper.ravel().copy()
    gl = np.maximum(oracle.task.lower, -1e19).copy()
    gu = np.minimum(oracle.task.upper, 1e19).copy()
    fatal, evaluation_errors, history = [], [], []
    # IPOPT 3.14+ exposes accepted iterates during the intermediate callback.
    # Read those values directly: the oracle cache may contain a rejected
    # line-search trial and would give misleading convergence plots.
    current_iterate = getattr(lib, "GetIpoptCurrentIterate", None)
    if current_iterate is not None:
        current_iterate.restype = B
        current_iterate.argtypes = [V, B, I, DP, DP, DP, I, DP, DP]
    current_x = np.zeros(n, dtype=np.float64)
    current_g = np.zeros(m, dtype=np.float64)

    def guarded(function):
        def callback(*args):
            if fatal:
                return False
            try:
                function(*args)
                return True
            except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
                # IPOPT can backtrack away from an invalid trial rollout.
                evaluation_errors.append(f"{type(exc).__name__}: {exc}")
                return False
            except BaseException as exc:
                # Never let ctypes swallow an exception, including Ctrl-C.
                fatal.append(exc)
                return False
        return callback

    def view(pointer, size):
        return np.ctypeslib.as_array(pointer, shape=(size,))

    @F
    @guarded
    def objective(n_, x_, new_x, output, user):
        output[0] = oracle.objective(view(x_, n))

    @F
    @guarded
    def gradient(n_, x_, new_x, output, user):
        view(output, n)[:] = oracle.gradient(view(x_, n))

    @G
    @guarded
    def constraints(n_, x_, new_x, m_, output, user):
        if m:
            view(output, m)[:] = oracle.constraints(view(x_, n))

    @JAC
    @guarded
    def jacobian(n_, x_, new_x, m_, nonzeros, rows, columns, output, user):
        if m == 0:
            return
        if not output:
            r, c = oracle.jacobianstructure()
            view(rows, n * m)[:] = r
            view(columns, n * m)[:] = c
        else:
            view(output, n * m)[:] = oracle.jacobian(view(x_, n))

    @HESS
    def hessian(*args):
        # Not called with hessian_approximation=limited-memory.
        return False

    @INTERMEDIATE
    def intermediate(mode, iteration, obj, inf_pr, inf_du, mu, dnorm, reg, alpha_du, alpha_pr, ls, user):
        try:
            row = {"iteration": int(iteration), "objective": float(obj),
                   "primal_infeasibility": float(inf_pr), "dual_infeasibility": float(inf_du),
                   "barrier_parameter": float(mu), "line_search_trials": int(ls),
                   "restoration": bool(mode)}
            if current_iterate is not None and current_iterate(
                handle, False, n, _ptr(current_x), None, None,
                m, _ptr(current_g) if m else None, None,
            ):
                row["controls"] = current_x.tolist()
                row["constraint_values"] = current_g.tolist()
                row["physical_constraint_values"] = (
                    current_g * oracle.task.physical_scales
                ).tolist()
                violation = np.maximum(
                    np.maximum(oracle.task.lower - current_g, current_g - oracle.task.upper),
                    0.0,
                )
                row["task_violation_inf"] = float(np.max(violation, initial=0.0))
            history.append(row)
            return not fatal
        except BaseException as exc:
            fatal.append(exc)
            return False

    handle = lib.CreateIpoptProblem(n, _ptr(lb), _ptr(ub), m, _ptr(gl), _ptr(gu), n * m,
                                    0, 0, objective, constraints, gradient, jacobian, hessian)
    if not handle:
        raise RuntimeError("CreateIpoptProblem failed; check the IPOPT ABI and NLP dimensions.")
    try:
        number_names = {"tol", "acceptable_tol", "constr_viol_tol", "acceptable_constr_viol_tol",
                        "bound_relax_factor", "dual_inf_tol", "compl_inf_tol", "max_cpu_time",
                        "max_wall_time", "acceptable_obj_change_tol", "mu_init", "bound_push",
                        "bound_frac", "nlp_lower_bound_inf", "nlp_upper_bound_inf"}
        for key, value in options.items():
            key_b = key.encode("utf-8")
            if isinstance(value, str):
                ok = lib.AddIpoptStrOption(handle, key_b, value.encode("utf-8"))
            elif isinstance(value, (int, np.integer)) and key not in number_names:
                ok = lib.AddIpoptIntOption(handle, key_b, int(value))
            elif isinstance(value, (int, float, np.number)):
                ok = lib.AddIpoptNumOption(handle, key_b, float(value))
            else:
                raise TypeError(f"IPOPT option {key!r} must be a string, integer, or float.")
            if not ok:
                raise ValueError(f"IPOPT rejected option {key}={value!r}; check its name, type, and value.")
        if not lib.SetIntermediateCallback(handle, intermediate):
            raise RuntimeError("Cannot register IPOPT iteration callback.")
        g = np.zeros(m, dtype=np.float64)
        f = D()
        code = int(lib.IpoptSolve(handle, _ptr(x), _ptr(g), ct.byref(f), None, None, None, None))
    finally:
        lib.FreeIpoptProblem(handle)
    if fatal:
        raise RuntimeError("A Python callback failed during IPOPT; see the original exception.") from fatal[0]
    return x, {"status": STATUS.get(code, f"IPOPT_status_{code}"), "solver_success": code in (0, 1),
               "iterations": history[-1]["iteration"] if history else -1,
               "history": {"iterations": history, "evaluation_errors": evaluation_errors,
                           "library": library_path, "status_code": code}}
