from pathlib import Path

from omegaconf import OmegaConf
from train import find_last_checkpoint_path


ROOT = Path(__file__).resolve().parents[1]


def test_batch12_and_validation_schedule_are_configured():
    cfg = OmegaConf.load(ROOT / "conf" / "maplocnet.yaml")
    OmegaConf.resolve(cfg)

    assert cfg.data.train.batch_size == 12
    assert cfg.data.val.batch_size == 12
    assert cfg.training.trainer.max_steps == 200_000
    assert cfg.training.trainer.val_check_interval == 10_000


def test_baseline_preserving_hybrid_training_is_configured():
    cfg = OmegaConf.load(ROOT / "conf" / "maplocnet.yaml")
    OmegaConf.resolve(cfg)

    assert cfg.experiment.name == "maplocnet__CNN_GNN_safe_v1"
    assert cfg.data.val_citys == ["SanFrancisco"]
    assert cfg.model.image_encoder.backbone.pretrained is False
    assert cfg.model.image_encoder.backbone.freeze_batch_normalization is True
    assert cfg.model.map_encoder.freeze_batch_normalization is True
    assert cfg.training.baseline_checkpoint is None
    assert Path(cfg.training.topveloc_checkpoint).is_file()
    assert cfg.model.map_encoder.fusion.gate_init == 20.0
    assert cfg.model.safeguard.weight == 0
    assert cfg.model.safeguard.margin >= 0
    assert 0 < cfg.training.optimizer.baseline_lr_scale < 1
    assert cfg.training.optimizer.baseline_warmup_steps == 0
    assert cfg.training.lr == 1e-6


def test_trainer_allows_step_validation_across_epochs():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "check_val_every_n_epoch=None" in source


def test_checkpoint_callbacks_separate_recovery_and_best_models():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "checkpointing_recovery" in source
    assert "checkpointing_best" in source
    assert "monitor=None" in source
    assert "save_on_train_epoch_end=False" in source


def test_find_last_checkpoint_uses_recovery_checkpoint_name(tmp_path):
    recovery = tmp_path / "last-step-checkpointing.ckpt"
    recovery.touch()

    assert find_last_checkpoint_path(tmp_path) == str(recovery)
