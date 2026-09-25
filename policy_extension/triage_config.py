"""Canonical TRIAGE environment defaults and validation (no GPU dependencies)."""

import math
import os

ENV_VALUES = {}


def _boolean(value):
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("expected a boolean flag")


def _env(name, default, convert=str):
    """Read and register a setting once for both the loss and remote actors."""
    raw = os.environ.get(name, str(default))
    try:
        value = convert(raw)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}; got {raw!r}") from exc
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    ENV_VALUES[name] = value
    return value


def _scope(value):
    return value.strip().lower()


RECENTER_ENABLE = _env("TRIAGE_RECENTER_ENABLE", 1, _boolean)
ADV_QUANTILE = _env("TRIAGE_ADV_QUANTILE", "0.70", float)
GATE_ENABLE = _env("TRIAGE_GATE_ENABLE", 0, _boolean)
REPAIR_ENABLE = _env("TRIAGE_REPAIR_ENABLE", 0, _boolean)

# Gate includes the discrete weights 1.0/0.3/0.1 and the Tail-N2 midpoint 0.55.
# Integer hundredths preserve every supported weight exactly while letting
# Megatron retain an integer token-normalizer accumulator.
TOKEN_WEIGHT_SCALE = 100
TOKEN_WEIGHT_KEY = "triage_token_weight_units"

if not 0.0 < ADV_QUANTILE <= 1.0:
    raise ValueError(f"TRIAGE_ADV_QUANTILE must be in (0, 1], got {ADV_QUANTILE}")


# Method parameters.
ADV_ALPHA = _env("TRIAGE_ADV_ALPHA", "0.25", float)
ADV_MIN = _env("TRIAGE_ADV_MIN_SCALE", "0.70", float)
ADV_MAX = _env("TRIAGE_ADV_MAX_SCALE", "1.0", float)
GATE_SIZE = _env("TRIAGE_GATE_SIZE", "64", int)
GATE_NEG_DELTA = _env("TRIAGE_GATE_NEG_DELTA", "-0.5", float)
GATE_NEG_W = _env("TRIAGE_GATE_NEG_WEIGHT", "0.3", float)
GATE_SEV_DELTA = _env("TRIAGE_GATE_SEVERE_DELTA", "-1.5", float)
GATE_SEV_W = _env("TRIAGE_GATE_SEVERE_WEIGHT", "0.1", float)
GATE_BAD_DELTA = _env("TRIAGE_GATE_BAD_DELTA", "-6.0", float)
GATE_BAD_COUNT_FULL = _env("TRIAGE_GATE_BAD_COUNT_FULL", "2", int)
GATE_NEG_ADV_MAX = _env("TRIAGE_GATE_NEG_ADV_MAX", "0.0", float)
GATE_SCOPES = frozenset({"q3", "whole"})
GATE_SCOPE = _env("TRIAGE_GATE_SCOPE", "whole", _scope)
REPAIR_SIZE = _env("TRIAGE_REPAIR_SIZE", "128", int)
REPAIR_COEF = _env("TRIAGE_REPAIR_COEF", "0.015", float)
REPAIR_START_STEP = _env("TRIAGE_REPAIR_START_STEP", "0", int)
REPAIR_HUBER_BETA = _env("TRIAGE_REPAIR_HUBER_BETA", "0.022360679775", float)
REPAIR_TARGET = _env("TRIAGE_REPAIR_TARGET", "-0.03", float)
REPAIR_SCOPES = frozenset({"whole", "adv_positive", "q2"})
REPAIR_SCOPE = _env("TRIAGE_REPAIR_SCOPE", "whole", _scope)

if ADV_ALPHA < 0:
    raise ValueError(f"TRIAGE_ADV_ALPHA must be non-negative, got {ADV_ALPHA}")
if not 0 <= ADV_MIN <= ADV_MAX:
    raise ValueError(f"require 0 <= TRIAGE_ADV_MIN_SCALE <= TRIAGE_ADV_MAX_SCALE, got {ADV_MIN}, {ADV_MAX}")
if GATE_SIZE <= 0 or REPAIR_SIZE <= 0:
    raise ValueError(f"segment sizes must be positive, got Gate={GATE_SIZE}, Repair={REPAIR_SIZE}")
if GATE_SCOPE not in GATE_SCOPES:
    raise ValueError(
        "TRIAGE_GATE_SCOPE must be one of "
        f"{sorted(GATE_SCOPES)}, got {GATE_SCOPE!r}"
    )
if GATE_BAD_COUNT_FULL <= 0:
    raise ValueError(
        "TRIAGE_GATE_BAD_COUNT_FULL must be positive, got "
        f"{GATE_BAD_COUNT_FULL}"
    )
if not 0 < GATE_SEV_W <= GATE_NEG_W <= 1:
    raise ValueError(
        "require 0 < TRIAGE_GATE_SEVERE_WEIGHT <= TRIAGE_GATE_NEG_WEIGHT <= 1, got "
        f"{GATE_SEV_W}, {GATE_NEG_W}"
    )
if REPAIR_COEF < 0:
    raise ValueError(f"TRIAGE_REPAIR_COEF must be non-negative, got {REPAIR_COEF}")
if REPAIR_START_STEP < 0:
    raise ValueError(
        f"TRIAGE_REPAIR_START_STEP must be non-negative, got {REPAIR_START_STEP}"
    )
if REPAIR_HUBER_BETA <= 0:
    raise ValueError(f"TRIAGE_REPAIR_HUBER_BETA must be positive, got {REPAIR_HUBER_BETA}")
if REPAIR_SCOPE not in REPAIR_SCOPES:
    raise ValueError(
        "TRIAGE_REPAIR_SCOPE must be one of "
        f"{sorted(REPAIR_SCOPES)}, got {REPAIR_SCOPE!r}"
    )


CALIBRATION_PATH = _env("TRIAGE_CALIBRATION_PATH", "")
CALIBRATION_EXPORT_DIR = _env("TRIAGE_CALIBRATION_EXPORT_DIR", "")

def environment():
    """Resolved values to pass to remote actors, including library defaults."""
    return {key: str(int(value)) if isinstance(value, bool) else str(value)
            for key, value in ENV_VALUES.items()}
