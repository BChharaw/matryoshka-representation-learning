# test.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc,
)
from dataset import get_dataset
from transformers import AutoModel

# -----------------------------
#   Define Model (same as train)
# -----------------------------
class MRLLayer(nn.Module):
    def __init__(self, num_classes, m):
        super(MRLLayer, self).__init__()
        self.m = m
        for doll in m:
            setattr(self, f"mrl_classifier_{doll}", nn.Linear(doll, num_classes))

    def forward(self, x):
        return [getattr(self, f"mrl_classifier_{doll}")(x[:, :doll]) for doll in self.m]

class EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, m, apply_mrl=True):
        super(EmotionClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.apply_mrl = apply_mrl
        self.m = m

        if apply_mrl:
            self.classifier = MRLLayer(num_classes, m)
        else:
            self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)  # list if MRL is used, otherwise single tensor

# -----------------------------
#   Evaluate Helper
# -----------------------------
def evaluate_model(model, testloader, device, scales):
    """
    Returns: 
      scale_preds: list (len = len(scales)) of all predicted labels,
      scale_probs: list (len = len(scales)) of softmax probabilities,
      targets:     list of ground-truth labels
    """
    scale_preds = [[] for _ in scales]
    scale_probs = [[] for _ in scales]
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(testloader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            if isinstance(outputs, torch.Tensor):
                # Wrap single output in a list to unify logic
                outputs = [outputs]

            # For each scale, gather predictions + probabilities
            for i, scale_out in enumerate(outputs):
                preds = scale_out.argmax(dim=1)
                probs = nn.functional.softmax(scale_out, dim=1)
                scale_preds[i].extend(preds.cpu().numpy())
                scale_probs[i].extend(probs.cpu().numpy())

            targets.extend(labels.cpu().numpy())

    scale_probs = [np.array(sp) for sp in scale_probs]
    targets = np.array(targets)
    return scale_preds, scale_probs, targets

# -----------------------------
#   Plotting Utils
# -----------------------------
def plot_confusion_matrices(scales, scale_preds, targets, class_names, fig_title="Confusion Matrices"):
    """
    Plots confusion matrices in subplots (2x3 for 6 scales, etc.).
    """
    num_scales = len(scales)
    cols = 3
    rows = (num_scales + cols - 1) // cols  # e.g. 6 -> 2 rows
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, scale in enumerate(scales):
        preds_i = scale_preds[i]
        cm = confusion_matrix(targets, preds_i)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names)
        acc = accuracy_score(targets, preds_i)
        axes[i].set_title(f"Scale {scale}\nAcc={acc:.3f}")
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(fig_title, fontsize=16, y=1.02)
    fig.tight_layout()
    plt.savefig("confusion_matrices.png")
    plt.close()


def plot_roc_curves(scales, scale_probs, targets, num_classes, fig_title="ROC Curves"):
    """
    Plots ROC curves in subplots for each scale.
    """
    cols = 3
    rows = (len(scales) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # One-hot encode targets
    one_hot_targets = np.eye(num_classes)[targets]

    for i, scale in enumerate(scales):
        ax = axes[i]
        probs = scale_probs[i]  # shape [N, num_classes]
        # Compute macro-average
        fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
        for c in range(num_classes):
            fpr_dict[c], tpr_dict[c], _ = roc_curve(one_hot_targets[:, c], probs[:, c])
            roc_auc_dict[c] = auc(fpr_dict[c], tpr_dict[c])

        # Macro-average
        all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
        mean_tpr /= num_classes
        macro_roc_auc = auc(all_fpr, mean_tpr)

        # Plot each class
        for c in range(num_classes):
            ax.plot(fpr_dict[c], tpr_dict[c], label=f"C{c} AUC={roc_auc_dict[c]:.3f}", alpha=0.6)
        ax.plot(all_fpr, mean_tpr, label=f"Macro AUC={macro_roc_auc:.3f}", color='navy', linestyle='--')
        ax.plot([0,1], [0,1], linestyle='--', color='gray')
        ax.set_title(f"Scale {scale}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc="lower right", fontsize=8)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(fig_title, fontsize=16, y=1.02)
    fig.tight_layout()
    plt.savefig("roc_curves.png")
    plt.close()


def plot_matryoshka_schematic(scales):
    """
    A toy schematic showing each "doll" scale as a nested shape.
    This is purely conceptual. In real use, you could visually show your
    data or embeddings if they’re images/heatmaps.
    """
    plt.figure(figsize=(5,5))
    plt.title("Matryoshka Schematic")
    
    # We’ll draw circles with increasing size to represent each scale
    # with labels like “16, 64, 128, ...”
    # The biggest scale is the last in `scales`.
    # We do a radial offset for each scale to look nested.
    center = (0,0)
    for i, doll_size in enumerate(scales[::-1]):
        circle = plt.Circle(center, radius=0.2*(i+1), fill=False, linewidth=2, label=f"Scale {doll_size}")
        plt.gca().add_patch(circle)
    
    plt.gca().axis('scaled')
    plt.gca().set_xlim(-1.5, 1.5)
    plt.gca().set_ylim(-1.5, 1.5)
    plt.legend(loc="upper right")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("matryoshka_schematic.png")
    plt.close()


def show_sample_predictions(testloader, model, device, scales, class_names, max_samples=5):
    """
    Display actual data as it’s meant to be viewed, plus
    predictions from each scale. If text, decode tokens; if images, show images.
    Below is a simple text-based approach.
    """
    # For demonstration, we just gather a few samples
    collected_texts = []
    collected_labels = []
    predictions_per_scale = [[] for _ in scales]

    model.eval()
    with torch.no_grad():
        samples_collected = 0
        for batch in testloader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            
            # If you have a tokenizer, decode here. We'll show IDs as strings:
            for i in range(input_ids.size(0)):
                if samples_collected >= max_samples:
                    break
                text_ids = input_ids[i].tolist()
                text_str = f"Token IDs: {text_ids[:10]}..."  # truncated
                collected_texts.append(text_str)
                collected_labels.append(labels[i].item())
                samples_collected += 1
            
            if samples_collected >= max_samples:
                break

    # Now forward pass on these collected samples
    input_ids_tensor = torch.tensor([batch["input_ids"][0].tolist() for _ in range(max_samples)])  # placeholder
    # Real approach: you'd want the exact tokens you collected above
    # For the snippet, we do a naive approach:
    input_ids_tensor = torch.tensor([eval(txt.replace("Token IDs:", "").replace("...", "")) 
                                     for txt in collected_texts]).to(device)
    attention_mask_tensor = (input_ids_tensor != 0).long().to(device)
    outputs = model(input_ids_tensor, attention_mask_tensor)
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]

    for i, scale_out in enumerate(outputs):
        preds = scale_out.argmax(dim=1).cpu().numpy()
        predictions_per_scale[i] = preds

    # Let’s make a textual figure
    plt.figure(figsize=(8,4))
    plt.title("Sample Predictions")
    plt.axis('off')

    lines = []
    for i in range(max_samples):
        raw_text = collected_texts[i]
        gt_label = class_names[collected_labels[i]]
        scale_preds = {}
        for s_idx, sc in enumerate(scales):
            scale_preds[sc] = class_names[predictions_per_scale[s_idx][i]]
        line = (f"Sample {i}\n  Input: {raw_text}\n"
                f"  Ground Truth: {gt_label}\n"
                f"  Scale Preds: {scale_preds}\n")
        lines.append(line)

    summary_text = "\n".join(lines)
    plt.text(0.01, 0.95, summary_text, ha='left', va='top', fontsize=8)
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    plt.close()

# -----------------------------
#   Main Test
# -----------------------------
def test_model(
    model_path="emotion_classifier_model.pth",
    model_name="bert-base-uncased",
    num_classes=6,
    scales=[16, 64, 128, 256, 512, 768],
    device="cuda",
    class_names=None,
):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Get test data
    _, _, testloader = get_dataset()

    # Load model
    model = EmotionClassifier(model_name, num_classes, scales, apply_mrl=True)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Evaluate
    scale_preds, scale_probs, targets = evaluate_model(model, testloader, device, scales)

    # 1) Print classification metrics + confusion matrices
    for i, scale in enumerate(scales):
        cm = confusion_matrix(targets, scale_preds[i])
        acc = accuracy_score(targets, scale_preds[i])
        print(f"\n=== Scale {scale} ===")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(targets, scale_preds[i], target_names=class_names))

    # 2) Plot confusion matrices as subplots
    plot_confusion_matrices(scales, scale_preds, targets, class_names, fig_title="Confusion Matrices by Scale")

    # 3) Plot multiple ROC curves in subplots
    plot_roc_curves(scales, scale_probs, targets, num_classes, fig_title="ROC Curves by Scale")

    # 4) Show a schematic of the “matryoshka dolls”
    plot_matryoshka_schematic(scales)

    # 5) Show sample predictions on real data
    show_sample_predictions(testloader, model, device, scales, class_names, max_samples=5)


if __name__ == "__main__":
    test_model(
        model_path="emotion_classifier_model.pth",
        model_name="bert-base-uncased",
        num_classes=6,
        scales=[16,64,128,256,512,768],
        device="cuda",
        class_names=["anger", "fear", "joy", "love", "sadness", "surprise"],
    )