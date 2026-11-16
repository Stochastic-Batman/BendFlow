import pytest

from bendflow.config import CONFIG_DIR, load_math_config, load_networks_config


def test_default_files_load():
    math = load_math_config()
    assert 0 < math.active_set.delta < 1
    assert math.sampler.h_min <= math.sampler.h_max
    nets = load_networks_config()
    assert isinstance(nets.unet.channel_mults, tuple)
    assert nets.unet_train.epochs >= 1 and nets.dit_train.epochs >= 1


def edited(tmp_path, name, old, new):
    text = (CONFIG_DIR / name).read_text()
    assert old in text
    path = tmp_path / name
    path.write_text(text.replace(old, new))
    return path


def test_unknown_missing_and_invalid_values_raise(tmp_path):
    with pytest.raises(ValueError, match="unknown keys \\['tolerance'\\], missing keys \\['tol'\\]"):
        load_math_config(edited(tmp_path, "math.toml", "tol =", "tolerance ="))
    with pytest.raises(ValueError, match="delta"):
        load_math_config(edited(tmp_path, "math.toml", "delta = 0.01", "delta = 1.5"))
    with pytest.raises(ValueError, match="h_min"):
        load_math_config(edited(tmp_path, "math.toml", "h_min = 0.005", "h_min = 0.5"))
    with pytest.raises(ValueError, match="lr_schedule"):
        load_networks_config(edited(tmp_path, "networks.toml", '"cosine"', '"linear"'))


def test_max_steps_must_cover_algorithm_1(tmp_path):
    # h_min = 0.005 allows up to ceil(1 / 0.005) + 1 steps.
    with pytest.raises(ValueError, match="max_steps must be at least"):
        load_math_config(edited(tmp_path, "math.toml", "max_steps = 256", "max_steps = 150"))
    cfg = load_math_config().sampler
    assert cfg.max_steps >= 1 / cfg.h_min + 1
