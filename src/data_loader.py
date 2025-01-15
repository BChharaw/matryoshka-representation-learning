import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DefaultDataCollator
from datasets import load_dataset

class Data_Loader:
    def __init__(self, model_name="bert-base-uncased", max_length=256, batch_size=32, split="split"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.split = split

    def preprocess_function(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def get_dataset(self):
        data = load_dataset("dair-ai/emotion", self.split)
        train_ds = data["train"].map(self.preprocess_function, batched=True)
        val_ds = data["validation"].map(self.preprocess_function, batched=True)
        test_ds = data["test"].map(self.preprocess_function, batched=True)

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=DefaultDataCollator())
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=DefaultDataCollator())
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=DefaultDataCollator())
        return train_dl, val_dl, test_dl

    def analyse_text_percentiles(self, dataset):
        lengths = sorted(len(ex["text"]) for ex in dataset)
        percentiles = [50, 75, 90, 95, 99]
        values = [lengths[int(len(lengths) * (p / 100))] for p in percentiles]
        for p, v in zip(percentiles, values):
            print(f"{p}th percentile: {v}")

if __name__ == "__main__":
    dataset = Data_Loader()
    dataset = load_dataset("dair-ai/emotion", dataset.split)
    train_ds, val_ds, test_ds = dataset["train"], dataset["validation"], dataset["test"]
    print("Train dataset:")
    dataset.analyse_text_percentiles(train_ds)
    print("Validation dataset:")
    dataset.analyse_text_percentiles(val_ds)
    print("Test dataset:")
    dataset.analyse_text_percentiles(test_ds)