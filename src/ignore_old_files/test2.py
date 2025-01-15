from MRL.loss import MRLLoss
from dataset import get_dataset
from engine import EmotionClassifier

import torch
from tqdm import trange, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

def train(num_epochs: int = 10):
    """
    Train the EmotionClassifier with Matryoshka Representation Learning.
    
    Args:
        num_epochs (int): Number of epochs to train for.
    """
    # Load datasets
    trainloader, valloader, _ = get_dataset()

    # Initialize the model and move it to GPU
    model = EmotionClassifier(model_name="bert-base-uncased", num_classes=6, apply_mrl=True)
    model.half().to("cuda")

    # Define the loss function and optimizer
    criterion = MRLLoss(cm=torch.tensor([1, 1, 1, 1, 1, 1]).to("cuda"))
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=len(trainloader) * num_epochs
    )

    # Track the best validation loss
    min_val_loss = float("inf")

    print(f"{'=' * 30} Training Started {'=' * 30}")

    for epoch in trange(num_epochs, desc="Epochs"):
        # Initialize loss accumulators
        train_loss = 0
        val_loss = 0

        # Training Phase
        model.train()
        for batch_idx, data in enumerate(tqdm(trainloader, desc=f"Training Epoch {epoch + 1}")):
            input_ids = data["input_ids"].to("cuda")
            attention_mask = data["attention_mask"].to("cuda")
            targets = data["labels"].to("cuda")

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)

            train_loss += loss.item()

            # Debugging nested logits for the first batch of the first epoch
            if epoch == 0 and batch_idx == 0:
                for g, output in zip(model.m, outputs):
                    print(f"Granularity {g}D Logit Sample: {output[0].detach().cpu().numpy()}")

            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation Phase
        model.eval()
        with torch.no_grad():
            for data in tqdm(valloader, desc=f"Validation Epoch {epoch + 1}"):
                input_ids = data["input_ids"].to("cuda")
                attention_mask = data["attention_mask"].to("cuda")
                targets = data["labels"].to("cuda")

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Compute average losses
        train_loss /= len(trainloader)
        val_loss /= len(valloader)

        # Save model if validation loss improves
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), "model.pth")
            print(f"Model saved with validation loss: {val_loss:.4f}")

        # Print epoch summary
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")

    print(f"{'=' * 30} Training Completed {'=' * 30}")

if __name__ == "__main__":
    train()