from model import UNet
from data import SegData
from torch import nn
from tqdm import tqdm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import random
import metric
import utils


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def acc_fn(pred, mask):
    with torch.no_grad():
        pred = (pred > 0.5).float()
        return (pred == mask).sum() / torch.numel(pred)


def train_epoch(model, data_loader, optimizer, criterion, device):
    total_loss = 0.0
    acc = 0.0
    dice_socre = 0.0

    loader = tqdm(data_loader)
    for img, mask in loader:
        img = img.to(device)
        mask = mask.float().to(device).unsqueeze(dim=1)
        pred = model(img)

        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        tmp_acc = acc_fn(pred, mask)
        tmp_dice = metric.dice_score(pred, mask)
        loader.set_postfix_str(str(tmp_acc.item()))
        acc += tmp_acc.item()
        dice_socre += tmp_dice.item()

    total_loss /= len(data_loader)
    acc /= len(data_loader)
    dice_socre /= len(data_loader)

    return total_loss, acc, dice_socre


def valid(model, data_loader, crition, device):
    model.eval()
    total_loss = 0.0
    acc = 0.0
    dice_score = 0.0
    
    loader = tqdm(data_loader)
    for img, mask in loader:
        img = img.to(device)
        mask = mask.float().to(device).unsqueeze(dim=1)
        pred = model(img)

        loss = crition(mask, pred)
        total_loss += loss.item()

        tmp_acc = acc_fn(pred, mask)
        tmp_dice = metric.dice_score(pred, mask)
        acc += tmp_acc.item()
        dice_score += tmp_dice.item()

    total_loss = total_loss / len(data_loader)
    acc = acc / len(data_loader)
    dice_score = dice_score / len(data_loader)
    model.train()
    return total_loss, acc, dice_score


if __name__ == '__main__':
    setup_seed(20)

    data_root = './data'
    train_transform = A.Compose([
        A.Resize(height=160, width=240),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=160, width=240),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    train_data = SegData(data_root, train=True, transform=train_transform)
    test_data = SegData(data_root, False, val_transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16)

    device = 'cuda'
    activaion = 'relu'

    model = UNet(3, 1, activation=activaion).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    
    hist_dice = []
    epochs = 5
    for epoch in range(epochs):
        train_loss, train_acc, train_dice = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc, valid_dice = valid(model, test_loader, loss_fn, device)
        print(f"epoch: {epoch}, train loss: {train_loss}, train dice: {train_dice}, val loss: {val_loss}, val dice: {valid_dice}")
        hist_dice.append(valid_dice)

    utils.save_ckp(f'ckps/{activaion}_unet.pth', {'model': model.state_dict()})
    utils.write_list(hist_dice, f'metric_logs/{activaion}_hist.dice')
