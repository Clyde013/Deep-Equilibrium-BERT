from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase, RobertaTokenizer
from torch.utils.data import DataLoader
import datasets
from pytorch_lightning import LightningDataModule


class BookCorpusDataModule(LightningDataModule):
    """
    Data Module specifically for bookcorpus dataset. Works with any tokenizer (even custom ones).
    If you get SSL certificate errors with these, go to https://mystic.the-eye.eu/ and download the certificate,
    and install it into trusted root certification authorities.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            buffer_size: int = 10000,
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.dataset = None

    def setup(self, stage: str = None):
        # stream the dataset as it is too large to download
        self.dataset = datasets.load_dataset("bookcorpusopen", streaming=True, split="train")
        self.dataset = self.dataset.shuffle(seed=69, buffer_size=self.buffer_size)

        # tokenize the dataset. scuffed af to manually remove denote the remove_columns but it works
        self.dataset = self.dataset.map(self.encode, batched=True, remove_columns=["text", "title"]).with_format("torch")

    # since the dataset is a stream and masking is dynamic we can't exactly split it. train/val/test splits are
    # basically the same.
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.train_batch_size,
                          collate_fn=DataCollatorForLanguageModeling(self.tokenizer))

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.eval_batch_size,
                          collate_fn=DataCollatorForLanguageModeling(self.tokenizer))

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.eval_batch_size,
                          collate_fn=DataCollatorForLanguageModeling(self.tokenizer))

    def encode(self, example_batch):
        # tokenize the text
        features = self.tokenizer(example_batch["text"], max_length=self.max_seq_length, padding="max_length",
                                  truncation=True, return_tensors="pt")
        return features


if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    bookcorpus_datamodule = BookCorpusDataModule(tokenizer)
    bookcorpus_datamodule.setup()
    bookcorpus_dataset = bookcorpus_datamodule.dataset

    print(next(iter(bookcorpus_dataset)))
