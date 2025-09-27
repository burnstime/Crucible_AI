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
        return self.samples[idx]
