import torch 
import torchvision.transforms.functional as TF
import torch.nn as nn
from dataset import Cardataset
from torch.utils.data import DataLoader

def save_checkpoint(state,filename="my_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state,filename)


def load_checkpoint(checkpoint,model):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_mask,
    valid_dir,
    valid_mask,
    batch_size,
    train_transform,
    val_transform,
    num_workers = 1,
    pin_memory = True,):

    train_ds = Cardataset(
        image_dir = train_dir,
        mask_dir = train_mask,
        transform = train_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers=num_workers,
        pin_memory = pin_memory,
        shuffle = True,
    )

    valid_ds = Cardataset(
        image_dir = valid_dir,
        mask_dir = valid_mask,
        transform = val_transform,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size = batch_size,
        num_workers=num_workers,
        pin_memory = pin_memory,
        shuffle = False,
    )
    return train_loader,valid_loader


def check_accuracy(loader,model,device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y.sum())/((preds+y).sum()+1e-8))
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


