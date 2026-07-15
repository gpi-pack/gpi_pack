"""Small, dependency-light helpers shared by the Optuna tuners."""

from __future__ import annotations

import ast
import copy
import warnings
from collections.abc import Sequence
from numbers import Integral
from typing import Any, Callable

import numpy as np


def normalize_options(
    values: Any,
    *,
    name: str,
    cast: Callable[[Any], Any],
) -> tuple[Any, ...]:
    if isinstance(values, np.ndarray):
        raw = values.tolist()
    elif isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raw = [values]
    else:
        raw = list(values)
    if not raw:
        raise ValueError(f"{name} must contain at least one value.")
    try:
        return tuple(cast(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value in {name}: {values!r}") from exc


def normalize_architectures(values: Any, *, name: str) -> tuple[tuple[int, ...], ...]:
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, str):
        candidates = [values]
    elif isinstance(values, Sequence) and values and all(
        isinstance(value, Integral) and not isinstance(value, bool) for value in values
    ):
        candidates = [values]
    elif isinstance(values, Sequence):
        candidates = list(values)
    else:
        raise ValueError(f"{name} must be an architecture or sequence of architectures.")
    if not candidates:
        raise ValueError(f"{name} must contain at least one architecture.")

    parsed = []
    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                candidate = ast.literal_eval(candidate)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Invalid architecture in {name}: {candidate!r}") from exc
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
            raise ValueError(f"Each {name} architecture must be a sequence of integers.")
        if any(
            not isinstance(width, Integral) or isinstance(width, bool)
            for width in candidate
        ):
            raise ValueError(f"Each {name} architecture must contain only integers.")
        architecture = tuple(int(width) for width in candidate)
        if not architecture or any(width <= 0 for width in architecture):
            raise ValueError(f"Each {name} architecture must contain positive integers.")
        parsed.append(architecture)
    return tuple(parsed)


def suggest_float(
    trial: Any,
    name: str,
    values: tuple[float, ...],
    *,
    log: bool = False,
) -> float:
    if len(values) == 1:
        return float(values[0])
    low, high = min(values), max(values)
    return float(trial.suggest_float(name, low, high, log=log))


def suggest_categorical(trial: Any, name: str, values: tuple[Any, ...]) -> Any:
    if len(values) == 1:
        return copy.deepcopy(values[0])
    return copy.deepcopy(trial.suggest_categorical(name, list(values)))


def suggest_architecture(
    trial: Any,
    name: str,
    values: tuple[tuple[int, ...], ...],
) -> list[int]:
    if len(values) == 1:
        return list(values[0])
    labels = tuple(repr(list(value)) for value in values)
    selected = trial.suggest_categorical(name, list(labels))
    mapping = dict(zip(labels, values))
    return list(mapping[selected])


def record_resolved_params(trial: Any, params: dict[str, Any]) -> None:
    if hasattr(trial, "set_user_attr"):
        trial.set_user_attr("resolved_params", copy.deepcopy(params))


def report_and_prune(trial: Any, value: float, step: int) -> None:
    if hasattr(trial, "report"):
        trial.report(float(value), step=int(step))
    if hasattr(trial, "should_prune") and trial.should_prune():
        optuna = require_optuna()
        raise optuna.TrialPruned()


def validate_minimize_study(study: Any) -> None:
    """Reject studies that would select the largest validation loss."""
    directions = getattr(study, "directions", None)
    if directions is None:
        direction = getattr(study, "direction", None)
        directions = () if direction is None else (direction,)
    if len(directions) != 1:
        raise ValueError(
            "The Optuna study must be single-objective with direction='minimize'."
        )
    direction_name = str(
        getattr(directions[0], "name", directions[0])
    ).lower()
    if direction_name != "minimize":
        raise ValueError("The Optuna study direction must be 'minimize'.")


def validate_n_jobs(n_jobs: int) -> int:
    """Validate Optuna concurrency and flag thread-shared PyTorch state."""
    value = int(n_jobs)
    if value == 0:
        raise ValueError("n_jobs must be nonzero.")
    if value != 1:
        warnings.warn(
            "Parallel Optuna trials share PyTorch RNG and accelerator state; "
            "use n_jobs=1 for reproducible tuning or run process-isolated workers.",
            RuntimeWarning,
            stacklevel=2,
        )
    return value


def require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for tune(); install it with "
            "`pip install 'gpi_pack[tune]'`."
        ) from exc
    return optuna
