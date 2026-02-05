# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import sys
from typing import Any, Dict, List, Tuple, Union, get_origin, get_args

try:
    import yaml
except ImportError:
    yaml = None

from config import TrainConfig, load_yaml_with_extends, get_config_schema

# CLI options we consume ourselves (not TrainConfig keys)
_CLI_SPECIAL = frozenset({"config", "c", "load_weights_folder"})


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
        typ = type_hints[key]
        origin = get_origin(typ)
        args_ = get_args(typ)

        if typ is bool:
            overrides[key] = True
            i += 1
            continue
        if origin is list:
            # List[int], List[str], etc.: consume values until next --
            vals = []
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                vals.append(argv[i])
                i += 1
            if args_:
                elem_type = args_[0]
                if elem_type is int:
                    overrides[key] = [int(x) for x in vals]
                elif elem_type is float:
                    overrides[key] = [float(x) for x in vals]
                else:
                    overrides[key] = vals
            else:
                overrides[key] = vals
            continue
        # Single value
        i += 1
        if i >= len(argv):
            continue
        raw = argv[i]
        i += 1
        if raw.lower() in ("null", "none") and (origin is type(Union) and type(None) in (args_ or ())):
            overrides[key] = None
            continue
        if typ is int:
            overrides[key] = int(raw)
        elif typ is float:
            overrides[key] = float(raw)
        elif typ is str:
            overrides[key] = raw
        elif origin is type(Union) and args_:
            non_none = [a for a in args_ if a is not type(None)]
            if non_none and non_none[0] is int:
                overrides[key] = int(raw)
            elif non_none and non_none[0] is float:
                overrides[key] = float(raw)
            elif non_none and non_none[0] is str:
                overrides[key] = raw
            else:
                overrides[key] = raw
        else:
            overrides[key] = raw
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
        except Exception:
            pass
    for path in config_paths:
        if os.path.isfile(path) and yaml is not None:
            try:
                data, _ = load_yaml_with_extends(path)
                config.update(data)
            except Exception:
                pass
    config.update(cli_overrides)
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
