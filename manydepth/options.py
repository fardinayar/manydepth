# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import sys
import types
from typing import Any, Dict, List, Tuple, Union, get_origin, get_args

try:
    import yaml
except ImportError:
    yaml = None

from config import TrainConfig, load_yaml_with_extends, get_config_schema

# CLI options we consume ourselves (not TrainConfig keys)
_CLI_SPECIAL = frozenset({"config", "c", "load_weights_folder"})
_NONE_TYPE = type(None)


def _is_union_origin(origin: Any) -> bool:
    return origin is Union or origin is getattr(types, "UnionType", None)


def _strip_optional(typ: Any) -> Tuple[Any, bool]:
    origin = get_origin(typ)
    args = get_args(typ)
    if _is_union_origin(origin) and _NONE_TYPE in args:
        non_none = [arg for arg in args if arg is not _NONE_TYPE]
        if len(non_none) == 1:
            return non_none[0], True
    return typ, False


def _parse_bool(raw: str) -> bool:
    value = raw.lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError("Expected a boolean value, got {!r}".format(raw))


def _convert_scalar(raw: str, typ: Any):
    if typ is bool:
        return _parse_bool(raw)
    if typ is int:
        return int(raw)
    if typ is float:
        return float(raw)
    if typ is str:
        return raw
    return raw


def _convert_sequence(values: list, typ: Any):
    origin = get_origin(typ)
    args = get_args(typ)
    elem_type = args[0] if args else str
    converted = [_convert_scalar(value, elem_type) for value in values]
    if origin is tuple:
        return tuple(converted)
    return converted


def _collect_config_from_argv(argv: list) -> Tuple[list, str]:
    """Collect -c/--config paths and --load_weights_folder from argv. Returns (config_paths, load_weights_folder)."""
    config_paths = []
    load_weights_folder = None
    i = 0
    while i < len(argv):
        if argv[i] in ("-c", "--config") and i + 1 < len(argv):
            config_paths.append(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--load_weights_folder" and i + 1 < len(argv):
            load_weights_folder = os.path.expanduser(argv[i + 1])
            i += 2
            continue
        i += 1
    return config_paths, load_weights_folder


def _parse_cli_overrides(argv: list) -> dict:
    """
    Parse argv for --key [value] overrides. No defaults.
    Uses TrainConfig schema: bool = flag (--key means True), list = multiple values, optional = allow null.
    """
    _, type_hints = get_config_schema()
    valid_keys = set(type_hints)
    overrides = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith("--") or len(arg) == 2:
            i += 1
            continue
        key = arg[2:].replace("-", "_")
        if key in _CLI_SPECIAL or key not in valid_keys:
            i += 1
            continue
        typ, optional = _strip_optional(type_hints[key])
        origin = get_origin(typ)

        if typ is bool:
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                overrides[key] = _parse_bool(argv[i + 1])
                i += 2
            else:
                overrides[key] = True
                i += 1
            continue
        if origin in (list, tuple):
            # List[int], Tuple[int, ...], etc.: consume values until next --
            vals = []
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                vals.append(argv[i])
                i += 1
            if optional and len(vals) == 1 and vals[0].lower() in ("null", "none"):
                overrides[key] = None
            else:
                overrides[key] = _convert_sequence(vals, typ)
            continue
        # Single value
        i += 1
        if i >= len(argv):
            continue
        raw = argv[i]
        i += 1
        if optional and raw.lower() in ("null", "none"):
            overrides[key] = None
            continue
        overrides[key] = _convert_scalar(raw, typ)
    return overrides


def _build_merged_config(
    config_paths: list,
    load_weights_folder: str,
    cli_overrides: dict,
) -> dict:
    """
    Merge config from: saved config (if load_weights_folder: config.yaml) -> YAML from each -c (with extends) -> CLI.
    No defaults; user must provide a complete config via YAML (and optional extends) or saved run.
    """
    config = {}
    if load_weights_folder:
        try:
            cfg = TrainConfig.from_saved_run_dir(load_weights_folder)
            config.update(cfg.to_dict())
        except FileNotFoundError as e:
            if not config_paths:
                raise FileNotFoundError(
                    "Could not load saved config from {!r}: {}. Provide -c/--config as a fallback.".format(
                        load_weights_folder, e
                    )
                ) from e
            print(
                "Warning: could not load saved config from {!r}: {}. Using explicit config file(s).".format(
                    load_weights_folder, e
                )
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load saved config from {!r}: {}".format(load_weights_folder, e)
            ) from e
    for path in config_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Config file not found: {!r} (cwd={!r}). Use an absolute path or run from repo root.".format(
                    path, os.getcwd()
                )
            )
        if yaml is None:
            raise ImportError("PyYAML is required to load config. pip install PyYAML")
        try:
            data, _ = load_yaml_with_extends(path)
            config.update(data)
        except Exception as e:
            raise RuntimeError("Failed to load config from {!r}: {}".format(path, e)) from e
    config.update(cli_overrides)
    if load_weights_folder is not None:
        config["load_weights_folder"] = load_weights_folder
    return config


class MonodepthOptions:
    """Parse options from YAML (with optional extends) and CLI. No built-in defaults."""

    def parse(self) -> TrainConfig:
        if "--help" in sys.argv or "-h" in sys.argv:
            print("Usage: provide config via -c/--config <file.yaml> (use 'extends: base.yaml' for a full template).")
            print("Override with --key value. Example: -c configs/base.yaml --log_dir outs/run1")
            sys.exit(0)
        config_paths, load_weights_folder = _collect_config_from_argv(sys.argv)
        cli_overrides = _parse_cli_overrides(sys.argv)
        merged = _build_merged_config(config_paths, load_weights_folder, cli_overrides)
        if not merged:
            raise ValueError(
                "No config provided. Use -c/--config <file.yaml> (with optional 'extends: base.yaml' in the file) "
                "and/or --load_weights_folder <run_dir> to load a saved run's config."
            )
        # Normalize list/tuple fields for from_dict
        if merged.get("fusion_neighborhood_size") is not None:
            ne = merged["fusion_neighborhood_size"]
            if isinstance(ne, (list, tuple)):
                if len(ne) == 0:
                    merged["fusion_neighborhood_size"] = None
                elif len(ne) == 1:
                    merged["fusion_neighborhood_size"] = (int(ne[0]), int(ne[0]))
                else:
                    merged["fusion_neighborhood_size"] = tuple(int(x) for x in ne)
        return TrainConfig.from_dict(merged)
