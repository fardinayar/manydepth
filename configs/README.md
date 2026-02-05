# Config files and overrides

## Override order

Priority (lowest to highest):

1. **Parser defaults** (in `manydepth/options.py`)
2. **Config file(s)** from `-c` / `--config` (if you pass multiple, **later files override earlier**)
3. **Command line** arguments

So: **CLI overrides config file overrides defaults.**

## Base config inherited in-file (`extends`)

A YAML config can inherit a base **inside the file** via `extends`, so you don't need to pass the base on the CLI.

- **`extends: base.yaml`** – path is relative to the config file's directory. The base is loaded first, then the current file's keys override.
- **`extends: [first.yaml, second.yaml]`** – multiple bases; first, then second, then this file.

Example: `configs/ablation_no_lora.yaml` contains:

```yaml
extends: base.yaml
no_lora: true
model_name: "mdp_no_lora"
```

Then a single `-c configs/ablation_no_lora.yaml` gives you the full base plus these overrides (no need for `-c configs/base.yaml`).

## Ablation configs

- **`base.yaml`** – full default set; use as the common base for experiments.
- **`ablation_*.yaml`** – each extends `base.yaml` and overrides only the relevant flag(s) and `model_name`. No need to write a full config per ablation.

| Config | Override | model_name |
|--------|----------|------------|
| `ablation_no_lora.yaml` | no_lora: true | mdp_no_lora |
| `ablation_no_temporal_fusion.yaml` | no_temporal_fusion: true | mdp_no_temporal_fusion |
| `ablation_no_consistency_loss.yaml` | no_consistency_loss: true | mdp_no_consistency_loss |
| `ablation_no_loss_dynamic_weight.yaml` | no_loss_dynamic_weight: true | mdp_no_loss_dynamic_weight |
| `ablation_no_ignore_high_low_loss.yaml` | ignore_high_low_loss_pixels: false | mdp_no_ignore_high_low_loss |

### Using one config file

```bash
# Any ablation: base inherited via extends, only the flag and model_name differ
python manydepth/train.py -c configs/ablation_no_lora.yaml --log_dir outs/ablation_no_lora
python manydepth/train.py -c configs/ablation_no_temporal_fusion.yaml --log_dir outs/no_temporal_fusion
# Or base + CLI overrides
python manydepth/train.py -c configs/base.yaml --log_dir outs/exp1 --no_lora
```

### Layering configs via CLI (optional)

You can still pass **multiple** `-c` options; later files override earlier ones:

```bash
python manydepth/train.py -c configs/base.yaml -c configs/ablation_no_lora.yaml --log_dir "$LOG_BASE_DIR/baseline_no_lora"
```

## Examples matching ablation.sh

```bash
# Full model (base + g2s via CLI)
python manydepth/train.py -c configs/base.yaml --data_path "$DATA_PATH" --log_dir "$LOG_BASE_DIR/full_model" --png --g2s

# No LoRA ablation (base + ablation file + CLI)
python manydepth/train.py -c configs/base.yaml -c configs/ablation_no_lora.yaml --data_path "$DATA_PATH" --log_dir "$LOG_BASE_DIR/baseline_no_lora" --png --g2s

# No temporal fusion (ablation config)
python manydepth/train.py -c configs/ablation_no_temporal_fusion.yaml --data_path "$DATA_PATH" --log_dir "$LOG_BASE_DIR/no_temporal_fusion" --png --g2s
```
