# Explore the Behavior of Optimizer

more complex loss functions and interesting ideas are welcomed.

## Table of Contents
- [Explore the Behavior of Optimizer](#explore-the-behavior-of-optimizer)
  - [Table of Contents](#table-of-contents)
  - [TODO](#todo)
  - [Some Complex function(x, y) visualization.](#some-complex-functionx-y-visualization)
    - [Formular of the functions](#formular-of-the-functions)
  - [Experiment Result](#experiment-result)
    - [The behavior of different optimizer on those functions.](#the-behavior-of-different-optimizer-on-those-functions)
    - [Different beta value of adam in loss2](#different-beta-value-of-adam-in-loss2)
    - [Explore of adam in himmelblau function](#explore-of-adam-in-himmelblau-function)
    - [Explore Adam in rastrigin function](#explore-adam-in-rastrigin-function)
    - [Different steps in rastrigin of SGD](#different-steps-in-rastrigin-of-sgd)
    - [Different init value of complex](#different-init-value-of-complex)
    - [Different lr of SGD in mccormick](#different-lr-of-sgd-in-mccormick)
  - [Usage](#usage)
  - [Citation](#citation)
  - [Acknowledgments](#acknowledgments)
  - [Contact With Me](#contact-with-me)
  - [License](#license)


## TODO
- stable the adam optimzier. Exist many numberical problem.
- add Adamw optimizer
- add nestrove optimizer
- generate gif
- document

## Some Complex function(x, y) visualization.

In this repository, I will use the following complex functions to explore the behavior of deffierent optimizer.

| loss1 | loss2 | booth | rastrigin |
|-----|-----|-----|-----|
| <img src='./imgs/loss_fn_1_contour.png'> | <img src='./imgs/loss_fn_2_contour.png'> | <img src='./imgs/booth_contour.png'> | <img src='./imgs/rastrigin_contour.png'> |

| ackley | complex | himmelblau | mccormick |
|-----|-----|-----|-----|
| <img src='./imgs/ackley_contour.png'> | <img src='./imgs/complex_contour.png'> | <img src='./imgs/himmelblau_contour.png'> | <img src='./imgs/mccormick_contour.png'> |

### Formular of the functions

**loss1**

$$f(x, y) = x^2 + y^2$$

**loss2**

$$f(x, y)=0.5x^2 + 10y^2 + x + 2y$$

**booth**:

$$f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2$$

**rastrigin**：

$$f(x, y) = 20 + x^2 - 10 \cos(2\pi x) + y^2 - 10 \cos(2\pi y)$$

**ackley**：

$$f(x, y) = -20 \exp\left(-0.2 \sqrt{0.5(x^2 + y^2)}\right) - \exp\left(0.5(\cos(2\pi x) + \cos(2\pi y))\right) + 20 + e$$

**complex**:

$$f(x, y) = (x^3 - 3x^2 + 3y^2 - y^3)^2 + 0.1\cos(5x) + 0.1\sin(5y)$$

**himmelblau**：

$$f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$$

**mccormick**:

$$f(x, y) = \sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1$$


## Experiment Result

### The behavior of different optimizer on those functions.

**optimizer config**
- config for loss1, loss2, booth:  `lr=0.01, epochs=200, init_x=5, init_y=5, beta in Momentum is 0.9, beta1 in Adam is 0.9, beta2 in Adam is 0.999`.
- config for rastrigin:  `lr=0.01, epochs=300, init_x=5, init_y=5, beta in Momentum is 0.9, beta1 in Adam is 0.9, beta2 in Adam is 0.999`.
- config for ackley:  `lr=0.01, epochs=400, init_x=7.5, init_y=7.5, beta in Momentum is 0.9, beta1 in Adam is 0.9, beta2 in Adam is 0.999`.
- config for complex:  `lr=0.001, epochs=200, init_x=4.3, init_y=-1.5, beta in Momentum is 0.9, beta1 in Adam is 0.9, beta2 in Adam is 0.999`.
- config for himmelblau:  `lr=0.001, epochs=200, init_x=7.5, init_y=7.5, beta in Momentum is 0.9, beta1 in Adam is 0.9, beta2 in Adam is 0.999`.
- config for mccormick: `lr=0.1, epochs=200, init_x=7.5, init_y=7.5` for SGD.  `lr=0.01, epochs=200, init_x=7.5, init_y=7.5, beta in Momentum is 0.9` for Momentum.

|  | SGD | Momentum | Adam |
|-----|-----|-----|-----|
| loss1 |<img src='./imgs/opts/loss1_SGD_0.01_200.png'> | <img src='./imgs/opts/loss1_Momentum_0.01_200.png'> | <img src='./imgs/opts/loss1_Adam_0.01_200.png'> |
| loss2 |<img src='./imgs/opts/loss2_SGD_0.01_200.png'> | <img src='./imgs/opts/loss2_Momentum_0.01_200.png'> | <img src='./imgs/opts/loss2_Adam_0.01_200.png'> |
| booth |  <img src='./imgs/opts/booth_SGD_0.01_200.png'> |  <img src='./imgs/opts/booth_Momentum_0.01_200.png'> |  <img src='./imgs/opts/booth_Adam_0.01_200.png'> |
| rastrigin | <img src='./imgs/opts/rastrigin_SGD_0.01_300.png'> | <img src='./imgs/opts/rastrigin_Momentum_0.01_300.png'> | <img src='./imgs/opts/rastrigin_Adam_0.01_300.png'> |
| ackley | <img src='./imgs/opts/ackley_SGD_0.01_400.png'> | <img src='./imgs/opts/ackley_Momentum_0.01_400.png'> | Fail |
| complex | <img src='./imgs/opts/complex_SGD_0.001_200.png'> | <img src='./imgs/opts/complex_Momentum_0.001_200.png'> | <img src='./imgs/opts/complex_Adam_0.001_200.png'> |
| himmelblau | <img src='./imgs/opts/himmelblau_SGD_0.001_200.png'> | <img src='./imgs/opts/himmelblau_Momentum_0.001_200.png'> | <img src='./imgs/opts/himmelblau_Adam_0.001_200.png'> |
| mccormick | <img src='./imgs/opts/mccormick_SGD_0.1_200.png'> | <img src='./imgs/opts/mccormick_Momentum_0.01_200.png'> | TBD |


**As you can see, Adam optimizer does not work well for those functions. But does this means that adam is not suitable for simple functions? Let's tune it's beta parameters in the following content.**

### Different beta value of adam in loss2
`loss2` is a simple function. But Adam works badly. Let's tune the beta parameters of adam to see if Adam can work better. **I guess it can.**

**1. Just tune the beta1 to see what happens**
Let: `lr=1e-2, beta2=0.999, epochs=200`

| beta1=0.5 | beta1=0.6 | beta1=0.7 | beta1=0.8 | beta1=0.9 |
|-----|-----|-----|-----|-----|
| <img src='./imgs/tune_adam/loss2_Adam_0.01_0.5_0.999_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.6_0.999_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.7_0.999_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.8_0.999_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.9_0.999_200.png'> |

**beta1=0.8** works well. Let's peforming a fine-grained tuning.
| beta1=0.8 | beta1=0.81 | beta1=0.805 | beta1=0.85 |
|-----|-----|-----|-----|
| <img src='./imgs/tune_adam/loss2_Adam_0.01_0.8_0.999_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.81_0.999_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.805_0.999_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.85_0.999_200.png'> |

**beta1=0.81** works well. It can be seen that the beta1 control the existing trend. If beta1 is large, it will along the old trend. If beta1 is small, it will do adjust according the current gradient.

**2. Let tune the beta2 to see if there are any surprises.**
Let: `lr=1e-2, beta1=0.81, epochs=200`

|beta2 = 0.8| beta2=0.9 | beta2=0.99 | beta2=0.999 | beta2=0.9999 |
|-----|-----|-----|-----|-----|
| <img src='./imgs/tune_adam/loss2_Adam_0.01_0.81_0.8_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.81_0.9_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.81_0.99_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.81_0.999_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.81_0.9999_200.png'> |

My explanation is beta2 controls the step size of optimizer, determining how far it will move along the trend. seems that `beta2=0.999` is a nice choice.

Util now, as we can see, adam moves so slowly. How about increase it's learning rate? Let `beta1=0.81, beta2=0.999, epochs=200`.

| lr=0.01 | lr=0.03 |
|----|----|
| <img src='./imgs/tune_adam/loss2_Adam_0.01_0.81_0.999_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.03_0.81_0.999_200.png'> |

We can see that increase learning rate is not a better choice.

**3. Increase epochs.**
Let `lr=0.01, beta1=0.81, beta2=0.999`.

| epochs=200 | epochs=400 | epochs=800 | epochs=1000 |
|----|----|----|----|
| <img src='./imgs/tune_adam/loss2_Adam_0.01_0.81_0.999_200.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.81_0.999_400.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.81_0.999_800.png'> | <img src='./imgs/tune_adam/loss2_Adam_0.01_0.81_0.999_1000.png'> |

**We make it.** My guess is right. But this also proves that, no matter how we tune the parameters, Adam converges slower than SGD on simple problems. Perhaps there is a better way to make Adam work better in such straightforward cases, or perhaps not, but I don't know it for now.

### Explore of adam in himmelblau function
The above experiment result show that Adam does not converge in himmelblau function. So the first I want to lower the learning rate.

**1. lower learning rate**
Let `beta1=0.9, beta2=0.999, epochs=200`.
| lr=1e-4 | lr=3e-4 | lr=1e-5 | lr=3e-5 |
|----|----|----|----|
| <img src='./imgs/tune_adam/himmelblau_Adam_0.0001_200.png'> | <img src='./imgs/tune_adam/himmelblau_Adam_0.0003_200.png'> | <img src='./imgs/tune_adam/himmelblau_Adam_1e-05_200.png'> | <img src='./imgs/tune_adam/himmelblau_Adam_3e-05_200.png'> |

Learning rate has an important influence on this optimizationi process. 1e-4 is a better choice.

### Explore Adam in rastrigin function

"The Rastrigin function has many local optima, making it a challenging optimization problem. Now, let's see if Adam can optimize this problem effectively."

**1. Increase Epochs**
The above experimental result shows that Adam can reach the optimal point if we increase epochs. Let's give it a try. Let: 'lr=0.01, epochs=300, init_x=5, init_y=5, beta1 in Adam is 0.9, beta2 in Adam in 0.999.'

| epochs=300 | epochs=600 |
|----|----|
| <img src='./imgs/tune_adam/rastrigin_Adam_0.01_300.png'> | <img src='./imgs/tune_adam/rastrigin_Adam_0.01_600.png'> |

As we can see, increasing the epochs is not helpful. Another singal that experimental result shows is Adam may not have enough momentum to reach the optimal. Let's tune the beta1 parameter.

**2. Increase beta1**

Let: `lr=0.01, epochs=300, init_x=5, init_y=5, beta2=0.999.`

| beta1=0.91 | beta1=0.92 | beta1=0.93 | beta1=0.94 |
|----|----|----|----|
| <img src='./imgs/tune_adam/rastrigin_Adam_0.01_0.91_0.999_300.png'> | <img src='./imgs/tune_adam/rastrigin_Adam_0.01_0.92_0.999_300.png'> | <img src='./imgs/tune_adam/rastrigin_Adam_0.01_0.93_0.999_300.png'> | <img src='./imgs/tune_adam/rastrigin_Adam_0.01_0.94_0.999_300.png'> |

| beta1=0.95 | beta1=0.96 | beta1=0.97 | beta1=0.98 |
|----|----|----|----|
| <img src='./imgs/tune_adam/rastrigin_Adam_0.01_0.95_0.999_300.png'> | <img src='./imgs/tune_adam/rastrigin_Adam_0.01_0.96_0.999_300.png'> | <img src='./imgs/tune_adam/rastrigin_Adam_0.01_0.97_0.999_300.png'> | <img src='./imgs/tune_adam/rastrigin_Adam_0.01_0.98_0.999_300.png'> |

It seems that `beta1=0.98` is a better choice. Let's give it a fine-grained try. 

| beta1=0.98 | beta1=0.983 | beta1=0.985 | beta1=0.99 |
|----|----|----|----|
| <img src='./imgs/tune_adam/rastrigin_Adam_0.01_0.98_0.999_300.png'> | <img src='./imgs/tune_adam/rastrigin_Adam_0.01_0.983_0.999_300.png'> | <img src='./imgs/tune_adam/rastrigin_Adam_0.01_0.985_0.999_300.png'> | <img src='./imgs/tune_adam/rastrigin_Adam_0.01_0.99_0.999_300.png'> |

The above table experiments proves that Adam may not have enough momentum to reach the optimal is correct. **Another noteworthy detail is just a slight change of beta1 can make a significant difference.**


### Different steps in rastrigin of SGD

### Different init value of complex

### Different lr of SGD in mccormick

## Usage

**1. Clone the repository**

```bash
git clone git@github.com:AllenWrong/From-Scratch.git
cd From-Scratch/learning-rate
```

**2. Run the Following command to have a first try**

```bash
python demo.py \
  --opt SGD \
  --loss_fn loss1 \
  --lr 1e-2 \
  --epochs 200 \
  --r_min -10 \
  --r_max 10 \
  --init_x 5 \
  --init_y 5
```

The output images are saved in `./imgs` directory.

**3. Run the following command to plot a function contour**

```bash
python show_contour.py \
  --fn loss1 \
  --rmin -10 \
  --rmax 10
```

## Citation

```
@misc{Explore the Behavior of Optimizer,
  author = {Zhongchao, Guan},
  title = {Explore the Behavior of Optimizer},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AllenWrong/From-Scratch/learning-rate}},
}
```

## Acknowledgments

- Some content is created with the assistance of ChatGPT.
- 

## Contact With Me

If you are interested in my project or you want to know more about the from scratch series, **follow me on github.**

If you have some ideas youd like to bring to life, **please email me.**

- 📧Email me: [gg884691896@gmail.com](mailto:gg884691896@gmail.com)
- Follow me on [LinkedIn](https://www.linkedin.com/in/zhongchao-guan-aa3288194/).

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/AllenWrong/From-Scratch/learning-rate)
