import os
import torch
from src.utils import directories as dir_helper

CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "crn_best.pth")


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_loss,
    epochs_no_improve,
    is_best=True,
) -> None:
    dir_helper.validate_dir(CHECKPOINT_DIR)
    if is_best:
        checkpoint_path = BEST_CHECKPOINT_PATH
    else:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"crn_checkpoint_{epoch}.pth")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "epochs_no_improve": epochs_no_improve,
        },
        checkpoint_path,
    )
    if is_best:
        print(f"💾 BEST Checkpoint salvato in '{checkpoint_path}'")
    else:
        print(f"💾 Checkpoint salvato in '{checkpoint_path}'")


def load_checkpoint(
    model,
    optimizer,
    device,
    from_best=True,
) -> tuple[int, float, int]:
    dir_helper.validate_dir(CHECKPOINT_DIR)
    dir = BEST_CHECKPOINT_PATH if from_best else None

    if os.path.isfile(dir):
        checkpoint = torch.load(dir, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return (
            checkpoint["epoch"],
            checkpoint["best_loss"],
            checkpoint["epochs_no_improve"],
        )
    return 0, float("inf"), 0
