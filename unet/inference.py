from model import UNet
import utils
from data import SegData
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import os


val_transform = A.Compose([
    A.Resize(height=160, width=240),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])
test_data = SegData('./data', False, val_transform)
device = 'cuda'

infer_times = 3
for i in range(infer_times):
    img_idx = random.randint(0, len(test_data))

    activations = ['relu', 'gelu', 'silu', 'pysilu']

    img, mask = test_data[img_idx]
    plot_imgs = [img.numpy(), mask.unsqueeze(dim=0).numpy()]
    img = img.to(device).unsqueeze(dim=0)
    
    for act in activations:
        model = UNet(3, 1, activation=act).to(device)
        utils.load_ckp(os.path.join('ckps', f"{act}_unet.pth"), model)


        pred = model(img)
        plot_imgs.append(pred.squeeze(dim=0).detach().cpu().numpy())

    plot_titles = ['Image', 'True Mask']
    for act in activations:
        plot_titles.append(f"{act} pred")
    utils.sub_plot(plot_imgs, plot_titles, f'imgs/infer_{i}.png')