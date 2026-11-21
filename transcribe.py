#!/usr/bin/env python3
"""Audio in; transcript and language intervals out.

    PYTHONPATH=. python transcribe.py recording.wav [--run NAME] [--out result.json] [--device cpu]

Prints one line per language interval. Audio in any sample rate or channel count is accepted.
An interval that matches no trained language is tagged "unknown"; it keeps its phones and carries
a best-guess spelling in the nearest known language. Everything runs locally.
"""
import argparse
import json
import os
from pathlib import Path

os.umask(0o077)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import soundfile as sf
import torch
import yaml

PROJ = Path(__file__).resolve().parent
SYSTEM_CONFIG = PROJ / "results/system_config.json"
WINDOW_S, OVERLAP_S, SR = 30.0, 4.0, 16000


class Recogniser:
    def __init__(self, run, device=None):
        from experiments import common as C
        from models.complete_system import UniversalASR
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = yaml.safe_load(open(PROJ / "results/logs" / run / "config.yaml"))
        self.vocab = C.load_vocab()
        state = torch.load(PROJ / "results/checkpoints" / run / "best.pt", map_location="cpu", weights_only=True)
        self.model = UniversalASR(cfg["backbone"], len(self.vocab), "frozen_cnn", init_state=state).to(self.device).eval()

    @torch.no_grad()
    def _forward(self, x):
        audio = torch.from_numpy((x - x.mean()) / (x.std() + 1e-7))[None].to(self.device)
        mask = torch.ones_like(audio, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            ctc, lid = self.model(audio, mask)
        return ctc.float().log_softmax(-1)[0].cpu().numpy(), lid.float().softmax(-1)[0].cpu().numpy()

    def emissions(self, x):
        """Log phone posteriors and language posteriors per 20 ms frame. Long audio runs in
        overlapping windows, and each frame comes from the window where it is furthest from an edge."""
        win, hop = int(WINDOW_S * SR), int((WINDOW_S - OVERLAP_S) * SR)
        if len(x) <= win:
            return self._forward(x)
        lps, lids, keep_from = [], [], 0
        margin = int(OVERLAP_S / 2 / 0.02)
        start = 0
        while True:
            lp, lid = self._forward(x[start:start + win])
            last = start + win >= len(x)
            end = len(lp) if last else len(lp) - margin
            lps.append(lp[keep_from:end])
            lids.append(lid[keep_from:end])
            if last:
                break
            start += hop
            keep_from = margin
        return np.concatenate(lps), np.concatenate(lids)


def load_audio(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


class Decoders:
    """Language models and spelling tables, loaded when first needed."""

    def __init__(self, vocab):
        from experiments import evaluate_all as E
        E.init_worker(vocab, ["hi", "mr", "bn", "gu", "ta", "te", "en"])
        self.E = E

    def __call__(self, name):
        return self.E.decoder(name)

    @property
    def p2g(self):
        return self.E._p2g

    @property
    def inv(self):
        return self.E._inv


def run_file(path, rec, dec, cfg):
    from decoding.system import decode
    x = load_audio(path)
    lp, lid = rec.emissions(x)
    return dict(file=str(path), duration=round(len(x) / SR, 2), segments=decode(lp, lid, cfg, dec, dec.p2g, dec.inv, dec.E.norm_text))


def main():
    from decoding.system import DEFAULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--run", default=None, help="checkpoint name under results/checkpoints; default from results/system_config.json")
    ap.add_argument("--out", default=None, help="write JSON here")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    if SYSTEM_CONFIG.exists():
        cfg.update(json.loads(SYSTEM_CONFIG.read_text()))
    run = a.run or cfg.get("run")
    if not run:
        raise SystemExit("no model chosen: pass --run or create results/system_config.json")
    rec = Recogniser(run, a.device)
    dec = Decoders(rec.vocab)
    results = []
    for p in a.audio:
        r = run_file(p, rec, dec, cfg)
        results.append(r)
        print(f"\n{r['file']} ({r['duration']} s)")
        for s in r["segments"]:
            tag = s["language_name"]
            if s["language"] == "unknown":
                print(f"[{s['start']:7.2f} - {s['end']:7.2f}] {tag:9s} phones: {s['phones']}")
                print(f"{'':30s}best guess ({s['best_guess_language']}): {s['transcript']}")
            else:
                print(f"[{s['start']:7.2f} - {s['end']:7.2f}] {tag:9s} {s['transcript']}")
    if a.out:
        Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
