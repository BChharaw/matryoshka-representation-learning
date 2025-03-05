import torch
from torch import nn, Tensor
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

class Architecture:
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        m_scales: List[int],
        apply_mrl: bool,
        trainloader,
        valloader,
        cm: torch.Tensor,
        lr: float,
        num_epochs: int,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.m_scales = sorted(m_scales)  # just making sure that scales is passed in the correct order
        self.apply_mrl = apply_mrl
        self.trainloader = trainloader
        self.valloader = valloader
        self.cm = cm
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device

        # Load BERT
        self.bert_model = AutoModel.from_pretrained(model_name).to(device)
        
        if apply_mrl:
            max_scale = max(self.m_scales)
            self.mrl_projection = nn.Linear(768, max_scale).to(device)
            self.shared_classifier = nn.Linear(max_scale, num_classes).to(device)
        else:
            self.shared_classifier = nn.Linear(768, num_classes).to(device)
        
        if apply_mrl:
            params = list(self.bert_model.parameters()) + list(self.mrl_projection.parameters()) + list(self.shared_classifier.parameters())
        else:
            params = list(self.bert_model.parameters()) + list(self.shared_classifier.parameters())
        self.optimizer = AdamW(params, lr=lr)
        steps = len(trainloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 0, steps)
        self.ce_loss = nn.CrossEntropyLoss()

        # collecting data for tracking
        self.epoch_list = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.scale_loss_history = [[] for _ in self.m_scales]
        self.train_acc_history = []
        self.val_acc_history = []
        self.scale_train_acc_history = [[] for _ in self.m_scales]
        self.scale_val_acc_history = [[] for _ in self.m_scales]
        self.best_val_loss = float("inf")

    def forward(self, input_ids: Tensor, attention_mask: Tensor):
        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = out.last_hidden_state[:, 0, :]
        
        if self.apply_mrl:
            proj_out = self.mrl_projection(cls_out)  # shape: [batch, max_scale]
            outputs = []
            # For each scale d, use the first d dimensions of both representation and classifier weight.
            for d in self.m_scales:
                nested_rep = proj_out[:, :d]
                nested_weight = self.shared_classifier.weight[:, :d]
                logits = F.linear(nested_rep, nested_weight, self.shared_classifier.bias)
                outputs.append(logits)
            return outputs
        else:
            return self.shared_classifier(cls_out)

    def compute_loss(self, preds, labels: Tensor):
        if isinstance(preds, torch.Tensor):
            return self.ce_loss(preds, labels)
        cm = self.cm.to(labels.device) if self.cm is not None else torch.ones(len(preds), device=labels.device)
        losses = [self.ce_loss(p, labels) for p in preds]
        return (torch.stack(losses) * cm).sum()

    def compute_scale_wise_loss(self, preds, labels):
        if isinstance(preds, torch.Tensor):
            preds = [preds]
        return [self.ce_loss(p, labels).item() for p in preds]

    def compute_scale_wise_accuracy(self, preds, labels):
        if isinstance(preds, torch.Tensor):
            preds = [preds]
        accs = []
        for p in preds:
            correct = (p.argmax(dim=1) == labels).sum().item()
            accs.append(correct / len(labels))
        return accs

    def train_one_epoch(self, epoch_idx):
        self.bert_model.train()
        if self.apply_mrl:
            self.mrl_projection.train()
        self.shared_classifier.train()
        total_loss, n_batches = 0.0, 0
        scale_loss_sum = [0.0] * len(self.m_scales)
        scale_acc_sum = [0.0] * len(self.m_scales)

        for batch in tqdm(self.trainloader, desc=f"Train Epoch {epoch_idx}"):
            inp = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.forward(inp, mask)
            loss = self.compute_loss(outputs, labels)
            total_loss += loss.item()

            sw_losses = self.compute_scale_wise_loss(outputs, labels)
            sw_accs = self.compute_scale_wise_accuracy(outputs, labels)
            for i, val in enumerate(sw_losses):
                scale_loss_sum[i] += val
            for i, val in enumerate(sw_accs):
                scale_acc_sum[i] += val

            loss.backward()
            self.optimizer.step()
            n_batches += 1

        self.scheduler.step()
        avg_loss = total_loss / n_batches
        avg_scale_loss = [v / n_batches for v in scale_loss_sum]
        avg_scale_acc = [v / n_batches for v in scale_acc_sum]
        overall_acc = avg_scale_acc[-1] if self.apply_mrl else avg_scale_acc[0]
        return avg_loss, avg_scale_loss, overall_acc, avg_scale_acc

    def validate_one_epoch(self, epoch_idx):
        self.bert_model.eval()
        if self.apply_mrl:
            self.mrl_projection.eval()
        self.shared_classifier.eval()
        total_loss, n_batches = 0.0, 0
        scale_loss_sum = [0.0] * len(self.m_scales)
        scale_acc_sum = [0.0] * len(self.m_scales)

        with torch.no_grad():
            for batch in tqdm(self.valloader, desc=f"Val Epoch {epoch_idx}"):
                inp = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.forward(inp, mask)
                loss = self.compute_loss(outputs, labels)
                total_loss += loss.item()

                sw_losses = self.compute_scale_wise_loss(outputs, labels)
                sw_accs = self.compute_scale_wise_accuracy(outputs, labels)
                for i, val in enumerate(sw_losses):
                    scale_loss_sum[i] += val
                for i, val in enumerate(sw_accs):
                    scale_acc_sum[i] += val
                n_batches += 1

        avg_loss = total_loss / n_batches
        avg_scale_loss = [v / n_batches for v in scale_loss_sum]
        avg_scale_acc = [v / n_batches for v in scale_acc_sum]
        overall_acc = avg_scale_acc[-1] if self.apply_mrl else avg_scale_acc[0]
        return avg_loss, avg_scale_loss, overall_acc, avg_scale_acc

    def fit(self):
        for e in range(1, self.num_epochs + 1):
            train_loss, train_sw_loss, train_overall_acc, train_sw_acc = self.train_one_epoch(e)
            val_loss, val_sw_loss, val_overall_acc, val_sw_acc = self.validate_one_epoch(e)

            self.epoch_list.append(e)
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            for i, sl in enumerate(train_sw_loss):
                self.scale_loss_history[i].append(sl)
            self.train_acc_history.append(train_overall_acc)
            self.val_acc_history.append(val_overall_acc)
            for i, acc_val in enumerate(train_sw_acc):
                self.scale_train_acc_history[i].append(acc_val)
            for i, acc_val in enumerate(val_sw_acc):
                self.scale_val_acc_history[i].append(acc_val)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                state_dict = {}
                for k, v in self.bert_model.state_dict().items():
                    state_dict["bert." + k] = v
                if self.apply_mrl:
                    for k, v in self.mrl_projection.state_dict().items():
                        state_dict["mrl_projection." + k] = v
                for k, v in self.shared_classifier.state_dict().items():
                    state_dict["shared_classifier." + k] = v
                torch.save(state_dict, "emotion_classifier_model.pth")

            print(f"\nEpoch {e} Summary:")
            print(f" Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f" Train Acc:  {train_overall_acc:.4f} | Val Acc: {val_overall_acc:.4f}")

        self.plot_loss()
        self.plot_accuracy()

    def plot_loss(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.epoch_list, self.train_loss_history, label='Train Loss', marker='o')
        plt.plot(self.epoch_list, self.val_loss_history, label='Val Loss', marker='o')
        for i, scale in enumerate(self.m_scales):
            plt.plot(self.epoch_list, self.scale_loss_history[i], label=f"Train Loss (Scale {scale})", linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig("losses_plot.png")
        plt.close()

    def plot_accuracy(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.epoch_list, self.train_acc_history, label='Train Acc (Largest)', marker='o')
        plt.plot(self.epoch_list, self.val_acc_history, label='Val Acc (Largest)', marker='o')
        for i, scale in enumerate(self.m_scales):
            plt.plot(self.epoch_list, self.scale_train_acc_history[i], label=f"Train Acc (Scale {scale})", linestyle='--')
            plt.plot(self.epoch_list, self.scale_val_acc_history[i], label=f"Val Acc (Scale {scale})", linestyle=':')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig("accuracies_plot.png")
        plt.close()