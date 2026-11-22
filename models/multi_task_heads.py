"""Head A: universal phonetic CTC. Head B: frame-level open-set language identification."""
import torch.nn as nn

LANGS = ["hi", "mr", "bn", "gu", "ta", "te", "en"]
UNK_LANG = len(LANGS)
LID_CLASSES = len(LANGS) + 1
LANG2ID = {l: i for i, l in enumerate(LANGS)}


class Heads(nn.Module):
    def __init__(self, hidden, vocab_size, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.ctc = nn.Linear(hidden, vocab_size)
        self.lid = nn.Linear(hidden, LID_CLASSES)

    def forward(self, h):
        h = self.drop(h)
        return self.ctc(h), self.lid(h)
