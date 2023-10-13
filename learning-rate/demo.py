import numpy as np
import matplotlib.pyplot as plt
from optim import *
from utils import F
import utils
from argparse import ArgumentParser


def train(x, y, epochs, loss_fn, opt, grad_fn):
    loss = []
    coord = [(x, y)]
    for i in range(epochs):
        loss_val = loss_fn(x, y)
        loss.append(loss_val)
        grad_x, grad_y = grad_fn(x, y)
        x, y = opt.forward([x, y], [grad_x, grad_y])
        coord.append((x, y))
        if i % 50 == 0:
            print(f"epoch {i}, loss: {loss_val}")
    return loss, coord


def one_exp(
    loss_fn,
    grad_fn,
    opt,
    epochs,
    p_color,
    r_min,
    r_max,
    x = None,
    y = None,
    save_img_path=None,
):
    if x is None:
        x = np.random.uniform(low=r_min, high=r_max)
    if y is None:
        y = np.random.uniform(low=r_min, high=r_max)
    print(f'using x={x}, y={y}')
    loss, coord = train(x, y, epochs=epochs, loss_fn=loss_fn, opt=opt, grad_fn=grad_fn)
    x, y, loss_val = utils.contour_data(r_min, r_max, loss_fn)

    plt.figure(figsize=(12, 8))
    contour = plt.contourf(x, y, loss_val, 50, alpha=1.0, cmap='jet')
    cbar = plt.colorbar(contour)
    cbar.set_label('Loss Value', rotation=270, labelpad=15)
    
    for i, (x_c, y_c) in enumerate(coord):
        plt.plot(x_c, y_c, '+', color=p_color)
    plt.title(f"Final Loss: {loss[-1]}")
    if save_img_path is not None:
        plt.savefig(save_img_path)
    plt.show()



def get_parser(parser: ArgumentParser):
    parser.add_argument(
        '--opt',
        required=True,
        type=str,
        help='Optimizer name, [SGD, Momentum, Adam]'
    )
    parser.add_argument(
        '--loss_fn',
        required=True,
        type=str,
        help='Loss function name.'
    )
    parser.add_argument(
        '--lr',
        default=1e-3,
        type=float
    )
    parser.add_argument(
        '--beta1',
        default=None,
        help='momentum',
        type=float
    )
    parser.add_argument(
        '--beta2',
        default=None,
        help='second momentum',
        type=float
    )
    parser.add_argument(
        '--epochs',
        default=200,
        type=int
    )
    parser.add_argument(
        '--r_min',
        default=-10,
        type=float,
        help='The min limit of the contour of x-axis and y-axis'
    )
    parser.add_argument(
        '--r_max',
        default=10,
        type=float,
        help='The max limit of the contour of x-axis and y-axis'
    )
    parser.add_argument(
        '--init_x',
        default=None,
        type=float
    )
    parser.add_argument(
        '--init_y',
        default=None,
        type=float
    )
    parser.add_argument(
        '--p_color',
        default='white',
        type=str
    )
    return parser


if __name__ == "__main__":
    args = get_parser(ArgumentParser()).parse_args()
    # {loss_fn}_{opt}_{lr}_{m}_{epochs}
    save_path = f'./imgs/opts/{args.loss_fn}_{args.opt}_{args.lr}'
    if args.beta2 is not None and args.beta2 is not None:
        save_path = save_path + f"_{args.beta1}_{args.beta2}"
        opt = eval(args.opt)(lr=args.lr, beta1=args.beta1, beta2=args.beta2)
    elif args.beta1 is not None:
        save_path = save_path + f"_{args.beta1}"
        opt = eval(args.opt)(lr=args.lr, beta=args.beta1)
    else:
        opt = eval(args.opt)(lr=args.lr)

    save_path = save_path + f"_{args.epochs}.png"

    loss_fn = eval(f'F.{args.loss_fn}')
    grad_fn = eval(f"F.{args.loss_fn}_grad")
    one_exp(loss_fn, grad_fn, opt, args.epochs, args.p_color,
            args.r_min, args.r_max, args.init_x, args.init_y, save_path)