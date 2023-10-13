from utils import F, plot_f_contour
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument(
    '--fn',
    required=True,
    type=str,
    help='Loss function name.'
)
parser.add_argument(
    '--r_min',
    default=-10,
    type=float
)
parser.add_argument(
    '--r_max',
    default=10,
    type=float
)
args = parser.parse_args()


save_path = f'./imgs/{args.fn}_contour.png'
plot_f_contour(eval(f"F.{args.fn}"), args.r_min, args.r_max, save_path)
