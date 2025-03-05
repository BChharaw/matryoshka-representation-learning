import torch
import numpy as np
from torch import nn
from torch.nn.functional import softmax
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score
)
from data_loader import Data_Loader
from transformers import AutoModel
import torch.nn.functional as F  # Needed for F.linear

class Evaluator:
    def __init__(
        self,
        model_path="emotion_classifier_model.pth",
        model_name="bert-base-uncased",
        num_classes=6,
        scales=[16, 64, 128, 256, 512, 768],
        device="cuda",
        class_names=None,
        max_samples=5
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.num_classes = num_classes
        self.scales = scales
        self.device = device
        self.class_names = class_names if class_names else [f"Class {i}" for i in range(num_classes)]
        self.max_samples = max_samples

        # Load test set (via your Data_Loader)
        loader = Data_Loader(model_name=model_name)
        _, _, self.testloader = loader.get_dataset()
        self.tokenizer = loader.tokenizer

        # Build the model to match the revised MRL architecture:
        # BERT + shared mrl_projection + shared_classifier.
        self._build_model()
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)

        # We'll store predictions/probs for each scale
        self.scale_preds = [[] for _ in scales]
        self.scale_probs = [[] for _ in scales]
        self.targets = []

    def _build_model(self):
        # Build a container model that reflects the revised training architecture.
        self.bert = AutoModel.from_pretrained(self.model_name)
        max_scale = max(self.scales)
        self.mrl_projection = nn.Linear(768, max_scale)
        self.shared_classifier = nn.Linear(max_scale, self.num_classes)
        self.model = nn.Module()
        self.model.bert = self.bert
        self.model.mrl_projection = self.mrl_projection
        self.model.shared_classifier = self.shared_classifier

    def _forward(self, input_ids, attention_mask):
        out = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = out.last_hidden_state[:, 0, :]
        # Project to a vector of size [batch, max_scale]
        proj_out = self.model.mrl_projection(cls_out)
        outputs = []
        # For each scale d, use the first d dimensions from both the projection
        # and the classifier's weight matrix to enforce a nested structure.
        for d in self.scales:
            nested_rep = proj_out[:, :d]
            nested_weight = self.model.shared_classifier.weight[:, :d]
            logits = F.linear(nested_rep, nested_weight, self.model.shared_classifier.bias)
            outputs.append(logits)
        return outputs

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc="Evaluating"):
                inp = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self._forward(inp, mask)
                for i, scale_out in enumerate(outputs):
                    preds = scale_out.argmax(dim=1).cpu().numpy()
                    probs = softmax(scale_out, dim=1).cpu().numpy()
                    self.scale_preds[i].extend(preds)
                    self.scale_probs[i].extend(probs)
                self.targets.extend(labels.cpu().numpy())

        self.targets = np.array(self.targets)
        self.scale_probs = [np.array(sp) for sp in self.scale_probs]

    def log_metrics(self):
        print("\n=== Evaluation Metrics ===")
        for i, scale in enumerate(self.scales):
            acc = accuracy_score(self.targets, self.scale_preds[i])
            print(f"\nScale {scale}")
            print(f" Accuracy: {acc:.4f}")
            print(" Classification Report:")
            print(classification_report(self.targets, self.scale_preds[i], target_names=self.class_names))

    def plot_confusion_matrices(self, output_file="confusion_matrices.png"):
        n_scales = len(self.scales)
        cols = 3
        rows = (n_scales + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, scale in enumerate(self.scales):
            cm = confusion_matrix(self.targets, self.scale_preds[i])
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.class_names, yticklabels=self.class_names,
                ax=axes[i]
            )
            acc = accuracy_score(self.targets, self.scale_preds[i])
            axes[i].set_title(f"Scale {scale}\nAcc={acc:.3f}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def sample_predictions(self):
        texts_collected = []
        labels_collected = []
        collected = 0
        for batch in self.testloader:
            if "text" not in batch:
                break
            for i in range(len(batch["text"])):
                if collected >= self.max_samples:
                    break
                texts_collected.append(batch["text"][i])
                labels_collected.append(batch["labels"][i].item())
                collected += 1
            if collected >= self.max_samples:
                break

        if not texts_collected:
            print("\nNo raw text found in test data to sample.")
            return

        inputs = self.tokenizer(
            texts_collected,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        outs = self._forward(inputs["input_ids"], inputs["attention_mask"])
        preds_per_scale = []
        for scale_out in outs:
            preds_per_scale.append(scale_out.argmax(dim=1).cpu().numpy())

        print("\n=== Sample Predictions ===")
        for i, txt in enumerate(texts_collected):
            gt = self.class_names[labels_collected[i]]
            pred_str = {}
            for sc_idx, sc in enumerate(self.scales):
                pred_str[sc] = self.class_names[preds_per_scale[sc_idx][i]]
            print(f"\nText {i}: {txt}")
            print(f"  Ground Truth: {gt}")
            print(f"  Predictions:")
            for sc in self.scales:
                print(f"    Scale {sc}: {pred_str[sc]}")

    def run_tests(self):
        self.evaluate()
        self.log_metrics()
        self.plot_confusion_matrices()
        self.sample_predictions()