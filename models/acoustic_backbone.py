"""Backbone loading and the trainability strategies of Experiment 1."""
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Model

PRE = Path(__file__).resolve().parent / "pretrained"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj"]
STRATEGIES = ["full", "frozen_cnn", "dense_only", "lora"]


def load_backbone(name, spec_augment=False, mms_adapter="eng", weights=True):
    """xlsr53 or mms1b, without its head. MMS keeps one pretrained adapter set, because its
    attention blocks expect adapters and random ones would corrupt every layer."""
    path = PRE / name
    cfg = Wav2Vec2Config.from_pretrained(path)
    # SpecAugment acts on encoder frames (20 ms each): T=40 at 10 ms is 20 frames here; F=30
    # masks 30 of the projected feature dimensions, since there are no Mel bins at this layer.
    cfg.apply_spec_augment = bool(spec_augment)
    cfg.mask_time_prob, cfg.mask_time_length = (0.05, 20) if spec_augment else (0.0, 10)
    cfg.mask_feature_prob, cfg.mask_feature_length = (0.05, 30) if spec_augment else (0.0, 10)
    cfg.layerdrop = 0.0
    if not weights:
        # Inference from a saved checkpoint: the architecture comes from config.json, and the caller
        # loads every weight from the checkpoint (UniversalASR raises if any are missing).
        return Wav2Vec2ForCTC(cfg).wav2vec2 if name == "mms1b" else Wav2Vec2Model(cfg)
    if name == "mms1b":
        m = Wav2Vec2ForCTC.from_pretrained(path, config=cfg, target_lang=mms_adapter,
                                           ignore_mismatched_sizes=True)
        return m.wav2vec2
    return Wav2Vec2Model.from_pretrained(path, config=cfg)


def apply_strategy(backbone, strategy, r=8, alpha=16, lora_dropout=0.05):
    """Returns the backbone (possibly wrapped) with requires_grad set for the strategy."""
    if strategy == "full":
        for p in backbone.parameters():
            p.requires_grad = True
        return backbone
    if strategy == "frozen_cnn":
        for p in backbone.parameters():
            p.requires_grad = True
        for p in backbone.feature_extractor.parameters():
            p.requires_grad = False
        return backbone
    if strategy == "dense_only":
        for p in backbone.parameters():
            p.requires_grad = False
        for p in backbone.feature_projection.parameters():
            p.requires_grad = True
        return backbone
    if strategy == "lora":
        from peft import LoraConfig, get_peft_model
        for p in backbone.parameters():
            p.requires_grad = False
        cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=LORA_TARGETS,
                         lora_dropout=lora_dropout, bias="none")
        return get_peft_model(backbone, cfg)
    raise ValueError(strategy)
