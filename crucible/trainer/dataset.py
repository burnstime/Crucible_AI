import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class InteractionDataset(Dataset):
    def __init__(
        self,
        log_path: str,
        tokenizer_name: str,
        max_len: int = 256,
        sep: str = "<|sep|>",
    ):
        self.samples = []
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                inp = entry.get("input", "")
                resp = entry.get("model_response", "")
                text = f"{inp}{sep}{resp}"
                tokens = self.tokenizer.encode(
                    text, truncation=True, max_length=max_len
                )
                self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # Pad to max_len for consistent batch sizes
        if len(tokens) < self.max_len:
            tokens = tokens + [self.tokenizer.pad_token_id or 0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        return tokens
