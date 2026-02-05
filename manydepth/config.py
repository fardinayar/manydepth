# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

"""
Dataclass-based config for training and evaluation.
Supports load/save from YAML. Saved run config is config.yaml only.
YAML configs can inherit a base via 'extends: base.yaml' (path relative to the config file).
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

def get_config_schema() -> Tuple[tuple, Dict[str, type]]:
    """Return (field_names, name -> type) for TrainConfig. Used for CLI parsing."""
    from typing import get_type_hints
    hints = get_type_hints(TrainConfig)
    names = tuple(TrainConfig.__dataclass_fields__)
    return names, hints

file_dir = os.path.dirname(os.path.abspath(__file__))


def load_yaml_with_extends(
    path: Union[str, Path],
) -> Tuple[dict, bool]:
    """
    Load a YAML file, resolving 'extends' so the base config is inherited in the file.
    Supports:
      extends: base.yaml
      extends: [base.yaml, other.yaml]   # first is base, then overrides, then this file
    Paths in 'extends' are resolved relative to the current config file's directory.
    Returns (merged_dict, had_extends). The merged dict does not contain the 'extends' key.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML config. pip install PyYAML")
    path = os.path.abspath(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    data = dict(data or {})
    if "extends" not in data:
        return (data, False)
    extends = data.pop("extends")
    base_paths = [extends] if isinstance(extends, str) else list(extends)
    cfg_dir = os.path.dirname(path)
    merged = {}
    for bp in base_paths:
        resolved = os.path.normpath(os.path.join(cfg_dir, bp))
        base_dict, _ = load_yaml_with_extends(resolved)
        merged.update(base_dict)
    merged.update(data)
    return (merged, True)


def _ensure_tuple(v: Union[None, int, List[int], Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
    """Convert list or int to tuple for neighborhood_size; None stays None."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v)
    return (int(v), int(v))


@dataclass
class TrainConfig:
    """
    Flat config for training and evaluation. All options + ablation parameters.
    No defaults: every key must be present in the config (or in a base via extends).
    from_dict raises ValueError if any required key is missing.
    """

    # PATHS
    data_path: str
    log_dir: str

    # TRAINING
    model_name: str
    split: str
    dataset: str
    png: bool
    height: int
    width: int
    disparity_smoothness: float
    scales: List[int]
    max_depth: float
    frame_ids: List[int]

    # OPTIMIZATION
    batch_size: int
    learning_rate: float
    num_epochs: int
    pytorch_random_seed: Optional[int]
    max_grad_norm: float
    warmup_steps: int

    # ABLATION
    weights_init: str
    num_matching_frames: int
    no_temporal_fusion: bool
    no_lora: bool
    no_consistency_loss: bool
    no_loss_dynamic_weight: bool
    ignore_high_low_loss_pixels: bool

    # SYSTEM
    no_cuda: bool
    num_workers: int

    # LOADING
    load_weights_folder: Optional[str]
    mono_weights_folder: Optional[str]
    models_to_load: List[str]

    # LOGGING
    log_frequency: int
    save_frequency: int
    save_intermediate_models: bool

    # EVALUATION
    disable_median_scaling: bool
    pred_depth_scale_factor: float
    ext_disp_to_eval: Optional[str]
    eval_split: str
    save_pred_disps: bool
    no_eval: bool
    eval_eigen_to_benchmark: bool
    eval_teacher: bool

    # DEPTH_ANYTHING
    depth_anything_encoder: str
    depth_anything_checkpoint_dir: str
    encoder_lr_coef: float
    fusion_lr_coef: float
    g2s: bool
    data_percent: float
    pose_from_scratch: bool
    gradient_accumulation_steps: int

    num_passes: int
    num_register_tokens: int

    # Ablation: LoRA (encoder/decoder)
    lora_rank: int
    lora_alpha: float
    lora_dropout: float

    # Ablation: feature fusion
    fusion_neighborhood_size: Optional[Tuple[int, ...]]
    fusion_num_scales: int
    fusion_lora_rank: int
    fusion_lora_alpha: float
    fusion_dropout: float
    fusion_drop_path: float

    # Ablation: scheduler
    scheduler_gamma: float
    scheduler_step_epochs: int

    # Ablation: pose encoder
    pose_encoder_num_layers: int

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-serializable). Tuples as lists."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        """
        Build config from dict. Raises ValueError if any required key is missing.
        """
        d = dict(d)
        fields = cls.__dataclass_fields__
        missing = [name for name in fields if name not in d]
        if missing:
            raise ValueError(
                "Config is missing required keys: {}. "
                "Ensure your config file (or its base via 'extends') defines every option.".format(
                    ", ".join(sorted(missing))
                )
            )
        if d["fusion_neighborhood_size"] is not None:
            d["fusion_neighborhood_size"] = _ensure_tuple(d["fusion_neighborhood_size"])
        if isinstance(d["scales"], (list, tuple)):
            d["scales"] = list(d["scales"])
        if isinstance(d["frame_ids"], (list, tuple)):
            d["frame_ids"] = list(d["frame_ids"])
        if isinstance(d["models_to_load"], (list, tuple)):
            d["models_to_load"] = list(d["models_to_load"])
        kwargs = {name: d[name] for name in fields}
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainConfig":
        """Load config from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config. pip install PyYAML")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TrainConfig":
        """Load config from a JSON file (e.g. saved opt.json or config.json)."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_saved_run_dir(cls, folder: Union[str, Path]) -> "TrainConfig":
        """Load config from a run dir: config.yaml only (in folder or its parent)."""
        folder = Path(folder)
        yaml_path = folder / "config.yaml"
        if yaml_path.is_file():
            return cls.from_yaml(yaml_path)
        parent = folder.parent / "config.yaml"
        if parent.is_file():
            return cls.from_yaml(parent)
        raise FileNotFoundError("No config.yaml in {} or its parent".format(folder))

    def to_json(self, path: Union[str, Path]) -> None:
        """Save config to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_yaml(self, path: Union[str, Path]) -> None: 
        """Save config to YAML (used for saved run config)."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config. pip install PyYAML")
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
