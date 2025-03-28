from MRL.engine import MRLLoss
from dataset import get_dataset
from MRL.engine import EmotionClassifier

import torch
from tqdm import trange, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

def train(num_epochs: int = 10):
    trainloader, valloader, _ = get_dataset()

    model = EmotionClassifier(model_name="bert-base-uncased", num_classes=6, apply_mrl=True)
    model.half().to("cuda")

    criterion = MRLLoss(cm=torch.tensor([1, 1, 1, 1, 1, 1]).to("cuda"))
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(trainloader) * num_epochs)

    min_val_loss = float("inf")

    for epoch in trange(num_epochs):
        model.train()
        train_loss = 0
        for data in tqdm(trainloader, desc=f"Training Epoch {epoch}"):
            input_ids = data["input_ids"].to("cuda")
            attention_mask = data["attention_mask"].to("cuda")
            targets = data["labels"].to("cuda")

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        # Learning rate scheduler step after each epoch
        scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in tqdm(valloader, desc=f"Validation Epoch {epoch}"):
                input_ids = data["input_ids"].to("cuda")
                attention_mask = data["attention_mask"].to("cuda")
                targets = data["labels"].to("cuda")

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Save model if validation loss is the lowest so far
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), "emotion_classifier_model.pth")

        # Logging the loss for the epoch
        print(f"Epoch {epoch}, Train Loss: {train_loss / len(trainloader)}, Val Loss: {val_loss / len(valloader)}")

if __name__ == "__main__":
    train()