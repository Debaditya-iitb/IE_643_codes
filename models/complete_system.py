"""Joint model: L = L_ctc + lambda * L_lid, plus checkpoint helpers."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.acoustic_backbone import apply_strategy, load_backbone
from models.multi_task_heads import Heads, UNK_LANG

BLANK = 0
IGNORE = -100


class UniversalASR(nn.Module):
    def __init__(self, backbone_name, vocab_size, strategy="frozen_cnn", r=8, alpha=16,
                 spec_augment=False, init_state=None):
        super().__init__()
        bb = load_backbone(backbone_name, spec_augment=spec_augment, weights=init_state is None)
        if init_state is not None:
            # masked_spec_embed exists only when SpecAugment is on, so a checkpoint can lack or carry it.
            missing, unexpected = bb.load_state_dict(init_state["backbone"], strict=False)
            bad = [k for k in list(missing) + list(unexpected) if k != "masked_spec_embed"]
            if bad:
                raise RuntimeError(f"backbone state mismatch: {bad[:10]}")
        self.hidden = bb.config.hidden_size
        self.strategy, self.r, self.alpha = strategy, r, alpha
        self.base_config = bb.config
        self.lengths_fn = bb._get_feat_extract_output_lengths
        self.backbone = apply_strategy(bb, strategy, r=r, alpha=alpha)
        self.heads = Heads(self.hidden, vocab_size)
        if init_state is not None:
            self.heads.load_state_dict(init_state["heads"])
        self.ctc_loss = nn.CTCLoss(blank=BLANK, zero_infinity=True, reduction="mean")

    def frame_lengths(self, sample_lengths):
        return self.lengths_fn(sample_lengths).long()

    def forward(self, audio, attention_mask):
        out = self.backbone(input_values=audio, attention_mask=attention_mask)
        return self.heads(out.last_hidden_state)

    def loss(self, batch, lambda_lid):
        ctc_logits, lid_logits = self(batch["audio"], batch["attention_mask"])
        flen = self.frame_lengths(batch["sample_lengths"])
        T = ctc_logits.size(1)
        flen = flen.clamp(max=T)

        # CTC stays in float32 with cudnn off: it is unstable under fp16.
        lp = F.log_softmax(ctc_logits.float(), dim=-1)
        has = batch["target_lengths"] > 0
        if has.any():
            with torch.backends.cudnn.flags(enabled=False):
                ctc = self.ctc_loss(lp[has].transpose(0, 1), batch["targets"][has],
                                    flen[has], batch["target_lengths"][has])
        else:
            ctc = lp.sum() * 0.0

        labels = self.frame_labels(batch, lp, flen, T)
        lid = F.cross_entropy(lid_logits.float().reshape(-1, lid_logits.size(-1)),
                              labels.reshape(-1), ignore_index=IGNORE)
        infeasible = int((batch["target_lengths"][has] > flen[has]).sum()) if has.any() else 0
        return ctc + lambda_lid * lid, dict(ctc=float(ctc.detach()), lid=float(lid.detach()),
                                            infeasible=infeasible)

    @torch.no_grad()
    def frame_labels(self, batch, lp, flen, T):
        """Monolingual and outlier rows: every frame takes the utterance class. Code-switched
        rows: Viterbi-align the phones under the current model, then each frame takes the
        language of the word its phone belongs to; blanks inherit the previous phone."""
        import torchaudio.functional as TAF
        labels = torch.full((lp.size(0), T), IGNORE, dtype=torch.long, device=lp.device)
        for i in range(lp.size(0)):
            n = int(flen[i])
            cls = batch["utt_class"][i]
            if int(batch["bound"][i]) >= 0:
                # Splice with a known join: the frame labels need no alignment.
                f = min(n, int(self.frame_lengths(batch["bound"][i:i + 1])[0]))
                labels[i, :f] = batch["lang_a"][i]
                labels[i, f:n] = batch["lang_b"][i]
                continue
            if int(cls) >= 0:
                labels[i, :n] = cls
                continue
            tl = int(batch["target_lengths"][i])
            if tl == 0 or tl > n:
                continue
            tgt = batch["targets"][i, :tl][None].cpu()
            path, _ = TAF.forced_align(lp[i, :n][None].cpu(), tgt,
                                       torch.tensor([n]), torch.tensor([tl]), blank=BLANK)
            path = path[0].numpy()
            tlangs = batch["token_langs"][i]
            cur, k, prev = None, 0, BLANK
            seq = []
            for t in path:
                if t != BLANK and (t != prev or prev == BLANK):
                    cur = int(tlangs[min(k, tl - 1)])
                    k += 1
                prev = t
                seq.append(cur)
            first = next((x for x in seq if x is not None), None)
            if first is None:
                continue
            seq = [first if x is None else x for x in seq]
            labels[i, :n] = torch.tensor(seq, device=lp.device)
        return labels


def trainable_state(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.cpu() for k, v in model.state_dict().items() if k in keep}


def full_backbone_state(model):
    """Plain backbone weights with any LoRA merged, for starting the next stage."""
    bb = model.backbone
    if model.strategy == "lora":
        import copy
        bb = copy.deepcopy(bb).merge_and_unload()
    return {k: v.cpu() for k, v in bb.state_dict().items()}


__all__ = ["UniversalASR", "trainable_state", "full_backbone_state", "UNK_LANG", "BLANK"]
