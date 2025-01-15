import torch
from transformers import AutoModel
from typing import List
from torch import nn, Tensor


class EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, m = [16, 64, 128, 256, 512, 768], apply_mrl = True):
        super(EmotionClassifier, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.m = m

        if apply_mrl:
            self.classifier = MRLLayer(num_classes, m)
        else:
            self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_output)
        return logits
    

class MRLLoss(nn.Module):
    def __init__(self, cm: List[int]):
        super(MRLLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, nested_preds: Tensor, ground_truth: Tensor, cm: Tensor = None):
        if cm is None:
            cm = torch.ones_like(ground_truth[0])

        mrl_losses = torch.stack([self.ce_loss(pred, ground_truth) for pred in nested_preds])
        loss = mrl_losses * cm
        
        return loss.sum()
    

class MRLLayer(nn.Module):
    def __init__(self, num_classes, m = [16, 64, 128, 256, 512, 768]):
        super(MRLLayer, self).__init__()

        self.m = m

        for doll in m:
            setattr(self, f"mrl_classifier_{doll}", nn.Linear(doll, num_classes))

    def forward(self, x):
        if isinstance(self.m, list):
            logits = [getattr(self, f"mrl_classifier_{doll}")(x[:, :doll]) for doll in self.m]
        elif isinstance(self.m, int):
            logits = getattr(self, f"mrl_classifier_{self.m}")(x[:, :self.m])
        else:
            raise ValueError("m should be either a list or an integer")
        
        return logits