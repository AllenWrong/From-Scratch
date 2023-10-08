import torch
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np



def save_ckp(ckp_path, states):
    torch.save(states, ckp_path)
    print('Save checkpoint to', ckp_path)


def load_ckp(ckp_path, model, optimizer=None):
    print("Load checkpoint from", ckp_path)
    states = torch.load(ckp_path)
    model.load_state_dict(states['model'])
    if optimizer is not None:
        optimizer.load_state_dict(states['opt'])


def plot(ys, save_img_path: str):
    if isinstance(ys, list):
        _ = plt.plot(list(range(len(ys))), ys)
        plt.savefig(save_img_path)
        plt.show()

    if isinstance(ys, pd.DataFrame):
        ys.plot()
        plt.savefig(save_img_path)


def write_list(obj: list, file: str):
    print('write list to', file)
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def load_list(file: str):
    print('load list from', file)
    with open(file, 'rb') as f:
        return pickle.load(f)


def show_activation_curve(f):
    xs = torch.linspace(-10, 10, 1000)
    ys = f(xs)
    xs = xs.numpy()
    ys = ys.numpy()
    plt.plot(xs, ys)
    plt.title('Function curve of pysilu activation.')
    plt.savefig('imgs/pysilu_curve.png')
    plt.show()


def sub_plot(imgs: list, plot_titles, save_img_path: str):
    sub_num = len(plot_titles)

    fig, axs = plt.subplots(1, sub_num, figsize=(15, 3))

    for i, ax in enumerate(axs):
        ax.imshow(np.transpose(imgs[i], (1, 2, 0)))
        ax.set_title(plot_titles[i])

    plt.tight_layout()
    plt.savefig(save_img_path)
    plt.show()
