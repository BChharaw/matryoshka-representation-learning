import torch
from architecture import Architecture
from data_loader import Data_Loader
from visualize import Evaluator

train = True
test = True

ds_loader = Data_Loader(model_name="bert-base-uncased", max_length=256, batch_size=32, split="split")
train_ds, val_ds, test_ds = ds_loader.get_dataset()

model = Architecture(model_name="bert-base-uncased", num_classes=6, m_scales=[16, 64, 128, 256, 512, 768], apply_mrl=True, trainloader=train_ds, valloader=val_ds, cm=torch.tensor([1, 1, 1, 1, 1, 1], 
    dtype=torch.float), lr=1e-5, num_epochs=6, device="cuda")

if train:
    model.fit()
tester = Evaluator(model_path="emotion_classifier_model.pth", model_name="bert-base-uncased", num_classes=6, scales=[16, 64, 128, 256, 512, 768], device="cuda",
class_names=["sadness", "joy", "love", "anger", "fear", "surprise"])

if test:
    tester.run_tests()
