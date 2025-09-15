# train.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import ModelNetCustomDataset
from models import MultiLayerTransformerClassifier
from utils import (
    accuracy_fn, train_step, test_step, plot_acc_losses, get_selected_classes, device
)
from checkpoint import get_or_train  

# ----- data & classes -----
all_classes, poor_classes, selected_classes = get_selected_classes()
NUM_CLASSES = len(selected_classes)

train_dataset = ModelNetCustomDataset(train=True,  classes=selected_classes, debug=False)
test_dataset  = ModelNetCustomDataset(train=False, classes=selected_classes, debug=False)

BATCH_SIZE = 32
EPOCHS = 20
INPUT_SHAPE = (200, 3)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ----- model & train loop (unchanged) -----
def train_model(model, epochs, verbose=True):
    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4)

    epoch_iterator = tqdm(range(epochs)) if verbose else range(epochs)
    for epoch in epoch_iterator:
        if verbose:
            print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_step(
            model=model, data_loader=train_dataloader,
            loss_fn=loss_fn, optimiser=optimiser,
            accuracy_fn=accuracy_fn, device=device, verbose=verbose
        )

        test_loss, test_acc = test_step(
            model=model, data_loader=test_dataloader,
            loss_fn=loss_fn, accuracy_fn=accuracy_fn,
            device=device, verbose=verbose
        )

        train_losses.append(train_loss); train_accs.append(train_acc)
        test_losses.append(test_loss);   test_accs.append(test_acc)

    if not verbose:
        print(f"Final: Train Acc: {train_accs[-1]:.2f}%, Test Acc: {test_accs[-1]:.2f}%")
    return train_accs, test_accs, train_losses, test_losses

# ----- load-or-train via checkpoint helper -----
FORCE_RETRAIN = False
MODEL_CFG = {
    "num_classes": NUM_CLASSES,
    "embed_dim": 256, "num_heads": 16, "num_layers": 2,
    "p_drop": 0.1, "ffn_dim_multiplier": 4,
    "selected_classes": list(selected_classes),
}

if __name__ == "__main__":
    # get_or_train will:
    #  - load ckpt if exists,
    #  - fallback to legacy weights (.pt) and fabricate minimal history,
    #  - otherwise train then save a new ckpt.
    attention_model, history, artifact_path, model_id = get_or_train(
        MODEL_CFG,
        epochs=EPOCHS,
        train_fn=train_model,        # uses the loop above
        test_loader=test_dataloader, # for quick eval in legacy case
        save_dir="saved_models",
        prefix="attention",
        force_retrain=FORCE_RETRAIN,
    )
    print(f"Model ID: {model_id} | artifact: {artifact_path}")

    # Plot using your existing util
    plot_acc_losses(
        model_name="Multi Layer Attention Classifier",
        train_accs=history["train_accs"],
        test_accs=history["test_accs"],
        train_losses=history["train_losses"],
        test_losses=history["test_losses"],
        epochs=history["epochs"],
    )
