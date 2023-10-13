import numpy as np
import matplotlib.pylab as plt


class F:
    @staticmethod
    def loss1(x, y):
        return x ** 2 + y ** 2
    
    @staticmethod
    def loss1_grad(x, y):
        return 2*x, 2*y
    
    @staticmethod
    def loss2(x, y):
        return 0.5 * x**2 + 10 * y**2 + x + 2 * y
    
    @staticmethod
    def loss2_grad(x, y):
        return 1 * x + 1, 20 * y + 2
    
    @staticmethod
    def rastrigin(x, y):
        return 20 + x**2 - 10 * np.cos(2*np.pi*x) \
               + y**2 - 10 * np.cos(2*np.pi*y)
    
    @staticmethod
    def rastrigin_grad(x, y):
        grad_x = 2*x + 10 * np.sin(2*np.pi*x) * 2*np.pi
        grad_y = 2*y + 10 * np.sin(2*np.pi*y) * 2*np.pi
        return grad_x, grad_y

    @staticmethod
    def ackley(x, y):
        return -20 * np.exp(-0.2 * np.sqrt(0.5*(x**2 + y**2))) - \
               np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + 20 + np.e
    
    @staticmethod
    def ackley_grad(x, y):
        it = np.sqrt((x**2+y**2) / 2)
        grad_x = -20 * np.exp(-0.2 * it) * (-0.1 * x / it) - \
                 np.exp(np.cos(2*np.pi*x) / 2 * (-np.sin(2*np.pi*x)*2*np.pi))
        grad_y = -20 * np.exp(-0.2 * it) * (-0.1 * y / it) - \
                 np.exp(np.cos(2*np.pi*y) / 2 * (-np.sin(2*np.pi*y)*2*np.pi))
        return grad_x, grad_y

    @staticmethod
    def himmelblau(x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    @staticmethod
    def himmelblau_grad(x, y):
        it1 = 2 * (x**2 + y - 11)
        it2 = 2 * (x + y**2 - 7)
        grad_x = it1 * 2 * x +  it2
        grad_y = it1 + it2 * 2 * y
        return grad_x, grad_y

    @staticmethod
    def schaffer_n2(x, y):
        return 0.5 + ((np.sin(x**2 - y**2))**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))

    @staticmethod
    def booth(x, y):
        return (x + 2*y - 7)**2 + (2*x + y - 5)**2
    
    @staticmethod
    def booth_grad(x, y):
        grad_x = 2 * (x + 2*y - 7) + 4 * (2*x + y - 5)
        grad_y = 4 * (x + 2*y - 7) + 2 * (2*x + y - 5)
        return grad_x, grad_y

    @staticmethod
    def six_hump_camel(x, y):
        return (4 - 2.1*x**2 + (x**4)/3) * x**2 + x*y + (-4 + 4*y**2) * y**2
    
    @staticmethod
    def six_hump_camel_grad(x, y):
        ...

    @staticmethod
    def mccormick(x, y):
        return (x-y)**2 - 1.5*x + 2.5*y +1
    
    @staticmethod
    def mccormick_grad(x, y):
        return 2*(x-y) - 1.5, 2.5 - 2*(x-y)
    
    @staticmethod
    def complex(x, y):
        item1 = (x**3 -3*x**2 + 3*y**2 - y**3)**2
        # item2 = (x**2 + y**2 - 4)**2
        return item1 + 0.1*np.cos(5*x) + 0.1*np.sin(5*y)
    
    @staticmethod
    def complex_grad(x, y):
        item = 2*(x**3 -3*x**2 + 3*y**2 - y**3)
        grad_x = item * (3 * x**2 - 6 * x) - 0.5*np.sin(5*x)
        grad_y = item * (6*y - 3*y**2) + 0.5*np.cos(5*y)
        return grad_x, grad_y


def contour_data(r_min, r_max, loss_fn, step=0.1):
    x_axis = np.arange(r_min, r_max, step)
    y_axis = np.arange(r_min, r_max, step)
    
    x, y = np.meshgrid(x_axis, y_axis)
    loss_val = loss_fn(x, y)
    return x_axis, y_axis, loss_val


def plot_f_contour(loss_fn, r_min, r_max, save_img_path=None):
    plt.figure(figsize=(12, 8))
    x_axis, y_axis, loss_val = contour_data(r_min, r_max, loss_fn, 0.2)    
    contour = plt.contourf(x_axis, y_axis, loss_val, 50, alpha=1.0, cmap='jet')

    cbar = plt.colorbar(contour)
    cbar.set_label('Loss Value', rotation=270, labelpad=15)
    if save_img_path is not None:
        plt.savefig(save_img_path)
    plt.show()


fns = [
    'loss1', 'loss2', 'rastrigin', 'mccormick',
    'ackley', 'booth', 'complex', 'himmelblau'
]