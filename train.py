import torch 
import torchvision.transforms.functional as TF
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import torch.optim as optim 
from Unet_model import Unet


from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy
)

# parameters

learning_rate = 1e-4
Device = "cuda"
batch_size = 8
num_epochs = 3 
num_workers = 2
image_height = 160
image_width = 240
pin_memory = True 
load_model = False 
train_dir = "data/train/"
train_mask = "data/train_masks/"
valid_dir = "data/valid/"
valid_mask = "data/valid_masks/"


def train_fn(loader , model , optimizer , loss_fn,scaler):
    loop = tqdm(loader)
    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device  = Device)
        targets = targets.to(device=Device)
        print(targets.shape)
        
        
        with torch.cuda.amp.autocast():
            predictions = model(data)
            prediction = predictions
            print(predictions.shape)
            loss = loss_fn(predictions , targets)
            
    
    # backward 

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # update tqdm loop 

        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(

        [
            A.Resize(height = image_height , width = image_width),
            A.Rotate(limit = 35 , p = 1.0 ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),

        ],
    )

    val_transform = A.Compose(

        [
            A.Resize(height = image_height , width = image_width),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),

        ],
    )

    model = Unet(in_channels = 3 , out_channels = 1).to(Device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)

    train_loader,val_loader = get_loaders(
        train_dir,
        train_mask,
        valid_dir,
        valid_mask,
        batch_size,
        train_transform,
        val_transform
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        train_fn(train_loader,model,optimizer,loss_fn,scaler)
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader,model,device=Device)




if __name__ == "__main__":
    main()
