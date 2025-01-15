import torch
from torch import nn, Tensor
from typing import List
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup

# ------------------------------------------------------------------
#   MRL Layer
# ------------------------------------------------------------------
class MRLLayer(nn.Module):
    def __init__(self, num_classes: int, m: List[int]):
        super(MRLLayer, self).__init__()
        self.m = m
        for doll in m:
            setattr(self, f"mrl_classifier_{doll}", nn.Linear(doll, num_classes))

    def forward(self, x: Tensor):
        # For each dimension in self.m, slice x and push it through the corresponding classifier
        return [getattr(self, f"mrl_classifier_{doll}")(x[:, :doll]) for doll in self.m]

# ------------------------------------------------------------------
#   EmotionClassifier
# ------------------------------------------------------------------
class EmotionClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, m: List[int], apply_mrl: bool = True):
        super(EmotionClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.apply_mrl = apply_mrl
        self.m = m

        if apply_mrl:
            self.classifier = MRLLayer(num_classes, m)
        else:
            self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        return self.classifier(cls_output)  # If MRL, returns list; else single Tensor

# ------------------------------------------------------------------
#   MRLLoss
# ------------------------------------------------------------------
class MRLLoss(nn.Module):
    def __init__(self, cm: torch.Tensor = None):
        super(MRLLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.cm = cm  # weighting for each scale

    def forward(self, nested_preds, ground_truth: Tensor, cm: torch.Tensor = None):
        # No MRL case
        if isinstance(nested_preds, torch.Tensor):
            return self.ce_loss(nested_preds, ground_truth)

        # MRL case
        if cm is None and self.cm is None:
            cm = torch.ones(len(nested_preds), device=ground_truth.device)
        elif cm is None:
            cm = self.cm.to(ground_truth.device)

        losses = torch.stack([self.ce_loss(pred, ground_truth) for pred in nested_preds])
        return (losses * cm).sum()

# ------------------------------------------------------------------
#   Trainer Class
# ------------------------------------------------------------------
class MRLTrainer:
    def __init__(
        self,
        model: nn.Module,
        trainloader,
        valloader,
        num_classes: int,
        m_scales: List[int],
        lr: float = 1e-5,
        cm: torch.Tensor = None,
        num_epochs: int = 6,
        device: str = "cuda"
    ):
        """
        User-friendly trainer to handle:
          - Training loop
          - Validation loop
          - Logging of scale-wise metrics
        """
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_classes = num_classes
        self.m_scales = m_scales
        self.lr = lr
        self.cm = cm
        self.num_epochs = num_epochs
        self.device = device

        # Loss, optimizer, scheduler
        self.criterion = MRLLoss(cm=self.cm).to(device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(trainloader) * self.num_epochs
        )

        # For plotting
        self.epoch_list = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.scale_loss_history = [[] for _ in self.m_scales]

        # We'll also track overall train/val accuracy
        self.train_acc_history = []
        self.val_acc_history = []
        # And scale-wise accuracy for train/val (list of lists)
        self.scale_train_acc_history = [[] for _ in self.m_scales]
        self.scale_val_acc_history = [[] for _ in self.m_scales]

        # Model saving
        self.best_val_loss = float("inf")

    def _compute_scale_wise_loss(self, outputs, targets):
        """
        Returns a list of CE losses (one per scale).
        """
        ce_loss_fn = nn.CrossEntropyLoss()
        return [ce_loss_fn(pred, targets) for pred in outputs]

    def _compute_scale_wise_accuracy(self, outputs, targets):
        """
        Returns a list of accuracies (one per scale).
        """
        # If not MRL, wrap in list
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]

        accs = []
        for pred in outputs:
            preds = pred.argmax(dim=1)
            correct = (preds == targets).float().sum().item()
            accs.append(correct / len(preds))
        return accs

    def train_one_epoch(self, epoch_idx: int):
        """
        One epoch of training; returns average loss, scale-wise avg losses, and scale-wise accuracy.
        """
        self.model.train()

        total_loss = 0.0
        scale_loss_sum = [0.0] * len(self.m_scales)
        scale_acc_sum = [0.0] * len(self.m_scales)
        total_batches = 0

        for batch in tqdm(self.trainloader, desc=f"Train Epoch {epoch_idx}"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["labels"].to(self.device)  # dair-ai/emotion => "label"

            self.optimizer.zero_grad()

            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()

            # Scale-wise losses & accuracies
            if isinstance(outputs, list):
                sw_losses = self._compute_scale_wise_loss(outputs, targets)
                for i, sw_l in enumerate(sw_losses):
                    scale_loss_sum[i] += sw_l.item()

            sw_accs = self._compute_scale_wise_accuracy(outputs, targets)
            for i, acc_val in enumerate(sw_accs):
                scale_acc_sum[i] += acc_val

            loss.backward()
            self.optimizer.step()
            total_batches += 1

        self.scheduler.step()

        # Averages
        avg_loss = total_loss / total_batches
        avg_scale_loss = [sl / total_batches for sl in scale_loss_sum]
        avg_scale_acc = [sa / total_batches for sa in scale_acc_sum]

        # For overall accuracy, let's use the largest scale's predictions
        # (the last scale in the list).
        overall_acc = avg_scale_acc[-1]

        return avg_loss, avg_scale_loss, overall_acc, avg_scale_acc

    def validate_one_epoch(self, epoch_idx: int):
        """
        One epoch of validation; returns average loss, scale-wise avg losses, and scale-wise accuracy.
        """
        self.model.eval()
        total_loss = 0.0
        scale_loss_sum = [0.0] * len(self.m_scales)
        scale_acc_sum = [0.0] * len(self.m_scales)
        total_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.valloader, desc=f"Val Epoch {epoch_idx}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Scale-wise losses & accuracies
                if isinstance(outputs, list):
                    sw_losses = self._compute_scale_wise_loss(outputs, targets)
                    for i, sw_l in enumerate(sw_losses):
                        scale_loss_sum[i] += sw_l.item()

                sw_accs = self._compute_scale_wise_accuracy(outputs, targets)
                for i, acc_val in enumerate(sw_accs):
                    scale_acc_sum[i] += acc_val

                total_batches += 1

        avg_loss = total_loss / total_batches
        avg_scale_loss = [sl / total_batches for sl in scale_loss_sum]
        avg_scale_acc = [sa / total_batches for sa in scale_acc_sum]

        # Overall val accuracy using the largest scale (last index)
        overall_acc = avg_scale_acc[-1]
        return avg_loss, avg_scale_loss, overall_acc, avg_scale_acc

    def fit(self):
        """
        Full training routine: runs for self.num_epochs, logs everything, saves best model.
        """
        for epoch in range(1, self.num_epochs + 1):
            # ---- Train ----
            train_loss, train_scale_loss, train_overall_acc, train_scale_acc = self.train_one_epoch(epoch)
            # ---- Validate ----
            val_loss, val_scale_loss, val_overall_acc, val_scale_acc = self.validate_one_epoch(epoch)

            # Logging
            self.epoch_list.append(epoch)
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            for i, sl in enumerate(train_scale_loss):
                self.scale_loss_history[i].append(sl)

            self.train_acc_history.append(train_overall_acc)
            self.val_acc_history.append(val_overall_acc)

            for i, acc_val in enumerate(train_scale_acc):
                self.scale_train_acc_history[i].append(acc_val)

            for i, acc_val in enumerate(val_scale_acc):
                self.scale_val_acc_history[i].append(acc_val)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), "emotion_classifier_model.pth")

            # Console Summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss:   {train_loss:.4f},  Val Loss:   {val_loss:.4f}")
            print(f"  Train Acc:    {train_overall_acc:.4f},  Val Acc:    {val_overall_acc:.4f}")
            print("  Scale-wise Train Losses:", [f"{x:.4f}" for x in train_scale_loss])
            print("  Scale-wise Val   Losses:", [f"{x:.4f}" for x in val_scale_loss])
            print("  Scale-wise Train Acc:  ", [f"{x:.4f}" for x in train_scale_acc])
            print("  Scale-wise Val   Acc:  ", [f"{x:.4f}" for x in val_scale_acc])
            print("")

        # After all epochs, produce final plots
        self.plot_loss()
        self.plot_accuracy()

    def plot_loss(self):
        """
        Plots overall train/val loss and scale-wise train loss.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self.epoch_list, self.train_loss_history, label='Train Loss', marker='o')
        plt.plot(self.epoch_list, self.val_loss_history, label='Val Loss', marker='o')

        for i, scale in enumerate(self.m_scales):
            plt.plot(self.epoch_list, self.scale_loss_history[i], 
                     label=f"Train Loss (Scale {scale})", linestyle='--')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss + Scale-wise Train Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig("losses_plot.png")
        plt.close()

    def plot_accuracy(self):
        """
        Plots overall train/val accuracy and scale-wise train/val accuracy.
        """
        plt.figure(figsize=(8, 6))
        # Overall lines
        plt.plot(self.epoch_list, self.train_acc_history, label='Train Acc (Largest Scale)', marker='o')
        plt.plot(self.epoch_list, self.val_acc_history, label='Val Acc (Largest Scale)', marker='o')

        # Scale-wise lines
        for i, scale in enumerate(self.m_scales):
            plt.plot(self.epoch_list, self.scale_train_acc_history[i],
                     label=f"Train Acc (Scale {scale})", linestyle='--')
            plt.plot(self.epoch_list, self.scale_val_acc_history[i],
                     label=f"Val Acc (Scale {scale})", linestyle=':')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train & Validation Accuracy (All Scales)')
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig("accuracies_plot.png")
        plt.close()

# ------------------------------------------------------------------
#   get_dataset() - from your dataset.py
# ------------------------------------------------------------------
from dataset import get_dataset

# ------------------------------------------------------------------
#   Main
# ------------------------------------------------------------------
def main(num_epochs: int = 6):
    # 1) Get Data
    trainloader, valloader, _ = get_dataset()

    # 2) Instantiate Model
    m_scales = [16, 64, 128, 256, 512, 768]
    model = EmotionClassifier(
        model_name="bert-base-uncased",
        num_classes=6,
        m=m_scales,
        apply_mrl=True
    )

    # 3) Create trainer
    trainer = MRLTrainer(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        num_classes=6,
        m_scales=m_scales,
        lr=1e-5,
        cm=torch.tensor([1,1,1,1,1,1], dtype=torch.float),
        num_epochs=num_epochs,
        device="cuda"
    )

    # 4) Fit
    trainer.fit()

if __name__ == "__main__":
    main(num_epochs=6)