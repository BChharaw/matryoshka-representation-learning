import torch
from transformers import AutoModel
from torch import nn, Tensor


class EmotionClassifier(nn.Module):
    """
    Emotion classification model supporting Matryoshka Representation Learning (MRL).
    """
    def __init__(self, model_name, num_classes, m=[16, 64, 128, 256, 512, 768], apply_mrl=True):
        super(EmotionClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)  # Pretrained transformer (e.g., BERT)
        self.m = m  # List of granularities

        # Use either the MRL classifier or a single linear classifier
        self.classifier = MRLLayer(num_classes, m) if apply_mrl else nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the backbone and classifier.

        Args:
            input_ids (Tensor): Tokenized input IDs.
            attention_mask (Tensor): Attention mask for the inputs.

        Returns:
            List[Tensor]: List of logits for each granularity (if MRL is used).
        """
        # Extract hidden states from the transformer backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Use the CLS token embedding (shape: [batch_size, hidden_dim])
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Pass the CLS token embedding through the classifier
        return self.classifier(cls_output)


class MRLLayer(nn.Module):
    """
    Layer for computing logits for multiple nested granularities.
    """
    def __init__(self, num_classes, m=[16, 64, 128, 256, 512, 768]):
        super(MRLLayer, self).__init__()
        self.m = m  # List of granularities

        # Define a separate linear layer for each granularity
        self.layers = nn.ModuleDict({str(d): nn.Linear(d, num_classes) for d in m})

        # Initialize weights for each linear layer
        for layer in self.layers.values():
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass for MRLLayer.

        Args:
            x (Tensor): Input embeddings (shape: [batch_size, max_dim]).

        Returns:
            List[Tensor]: Logits for each granularity.
        """
        logits = []

        # Iterate over granularities and compute logits
        for d in self.m:
            truncated = x[:, :d]  # Truncate embedding to the first `d` dimensions
            logits.append(self.layers[str(d)](truncated))

        return logits


class MRLLoss(nn.Module):
    """
    Loss function for Matryoshka Representation Learning (MRL).
    """
    def __init__(self, cm):
        """
        Args:
            cm (Tensor): Weights for scaling the loss of each granularity.
        """
        super(MRLLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.cm = cm / cm.sum()  # Normalize class weights

    def forward(self, nested_preds, targets):
        """
        Compute the loss for all granularities.

        Args:
            nested_preds (List[Tensor]): Predictions for each granularity.
            targets (Tensor): Ground truth labels.

        Returns:
            Tensor: Aggregated loss across all granularities.
        """
        # Compute cross-entropy loss for each granularity
        losses = torch.stack([self.ce_loss(pred, targets) for pred in nested_preds])

        # Apply weights and sum the losses
        total_loss = (losses * self.cm).sum()

        # Debugging: Print losses for each granularity
        # print("Granularity Losses:", losses.detach().cpu().numpy())

        return total_loss