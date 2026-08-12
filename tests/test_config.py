import pytest

from src.config import Config


CONFIG_YAML = """
seed: 7
output_dir: outputs
checkpoint_dir: checkpoints
data:
  batch_size: 128
  image_size: 32
model:
  patch_size: 4
  num_heads: 6
  embedding_dimension: 384
  num_classes: 100
training:
  epochs: 5
  learning_rate: 0.001
"""


def write_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML)
    return path


def test_from_yaml_loads_nested_sections(tmp_path):
    config = Config.from_yaml(write_config(tmp_path))
    assert config.seed == 7
    assert config.data.batch_size == 128
    assert config.training.epochs == 5
    assert config.num_patches == 64


def test_apply_overrides_routes_to_correct_section(tmp_path):
    config = Config.from_yaml(write_config(tmp_path))
    config.apply_overrides({"batch_size": 256, "epochs": 10, "seed": 99})
    assert config.data.batch_size == 256
    assert config.training.epochs == 10
    assert config.seed == 99


def test_apply_overrides_ignores_none(tmp_path):
    config = Config.from_yaml(write_config(tmp_path))
    config.apply_overrides({"batch_size": None})
    assert config.data.batch_size == 128


def test_apply_overrides_rejects_unknown_key(tmp_path):
    config = Config.from_yaml(write_config(tmp_path))
    with pytest.raises(KeyError):
        config.apply_overrides({"nonexistent": 1})


def test_validate_rejects_indivisible_patch_size(tmp_path):
    config = Config.from_yaml(write_config(tmp_path))
    config.model.patch_size = 5
    with pytest.raises(ValueError):
        config.validate()
