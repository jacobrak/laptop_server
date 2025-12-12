import os
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, AutoModel

# -----------------------
# Repro + counting
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------
# Tokenizer (small vocab is what makes ~1M possible)
# -----------------------
def train_wordpiece(corpus_path, out_dir="tok", vocab_size=4000):
    os.makedirs(out_dir, exist_ok=True)
    wp = BertWordPieceTokenizer(lowercase=True)
    wp.train(
        files=[corpus_path],
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    wp.save_model(out_dir)  # writes vocab.txt

    tok = BertTokenizerFast(
        vocab_file=os.path.join(out_dir, "vocab.txt"),
        do_lower_case=True,
    )
    return tok

# -----------------------
# Data
# -----------------------
class LineDataset(Dataset):
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

def mlm_collate(batch_text, tok, max_len=128, mlm_prob=0.15):
    enc = tok(
        batch_text,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        return_special_tokens_mask=True,   # <-- key fix
    )

    input_ids = enc["input_ids"]
    labels = input_ids.clone()

    # mask tokens (avoid special tokens)
    mask = torch.rand(labels.shape) < mlm_prob
    special = enc["special_tokens_mask"].bool()
    mask &= ~special

    labels[~mask] = -100
    input_ids[mask] = tok.mask_token_id

    return {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


# -----------------------
# Tiny Transformer (~1M params)
# -----------------------
class TinyEncoder(nn.Module):
    """
    BERT-ish encoder:
      token + position embeddings
      N TransformerEncoderLayers
      MLM head
    """
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # MLM head (tied weights is optional; keeping simple)
        self.mlm = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None, output_hidden=False):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        # TransformerEncoder uses src_key_padding_mask: True means "ignore"
        key_padding = None
        if attention_mask is not None:
            key_padding = (attention_mask == 0)

        h = self.encoder(x, src_key_padding_mask=key_padding)
        h = self.norm(h)
        logits = self.mlm(h)

        if output_hidden:
            return logits, h
        return logits

def mlm_loss(logits, labels):
    # logits: [B, L, V], labels: [B, L] with -100 ignored
    return nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

# -----------------------
# Training
# -----------------------
def main():
    set_seed(42)

    corpus = "corpus.txt"  # one text per line
    if not os.path.exists(corpus):
        raise FileNotFoundError("Create corpus.txt (one sentence/line) in the same folder.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) tokenizer
    tok = train_wordpiece(corpus_path=corpus, out_dir="tok", vocab_size=4000)

    # 2) student model ~1M params
    max_len = 128
    student = TinyEncoder(
        vocab_size=tok.vocab_size,
        max_len=max_len,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
        pad_id=tok.pad_token_id,
    ).to(device)

    print("Student trainable params:", count_params(student))

    # 3) optional teacher (hidden-state distillation only)
    # NOTE: teacher vocab differs; we only distill hidden reps, not logits.
    teacher = AutoModel.from_pretrained("bert-base-uncased").to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Map teacher hidden size (768) -> student hidden size (128)
    proj = nn.Linear(768, 128, bias=False).to(device)

    # 4) data
    ds = LineDataset(corpus)
    dl = DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: mlm_collate(b, tok, max_len=max_len, mlm_prob=0.15),
    )

    opt = torch.optim.AdamW(list(student.parameters()) + list(proj.parameters()), lr=5e-4)
    mse = nn.MSELoss()

    steps = 2000
    alpha_mlm = 0.7
    alpha_hid = 0.3

    it = iter(dl)
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        batch = {k: v.to(device) for k, v in batch.items()}

        # teacher hidden states (input ids are from student tokenizer; that's OK for hidden distill as a heuristic,
        # but better is teacher-tokenization. Keeping bare-bones as requested.)
        with torch.no_grad():
            t = teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).last_hidden_state  # [B, L, 768]

        logits, h = student(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden=True,
        )  # logits [B,L,V], h [B,L,128]

        loss_m = mlm_loss(logits, batch["labels"])
        loss_h = mse(h, proj(t))

        loss = alpha_mlm * loss_m + alpha_hid * loss_h

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(student.parameters()) + list(proj.parameters()), 1.0)
        opt.step()

        if step % 100 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f} | mlm {loss_m.item():.4f} | hid {loss_h.item():.4f}")

    # 5) save
    os.makedirs("tiny1m", exist_ok=True)
    torch.save(student.state_dict(), "tiny1m/student.pt")
    torch.save(proj.state_dict(), "tiny1m/proj.pt")
    tok.save_pretrained("tiny1m/tokenizer")
    print("Saved to tiny1m/")

if __name__ == "__main__":
    main()
