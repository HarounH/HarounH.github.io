---
title: (1/3) single fp32 perceptron
author: haroun7
date: 2023-09-25 11:33:00 +0800
math: true
mermaid: true
---

Today's [ANNs](https://en.wikipedia.org/wiki/Artificial_neural_network) are made by stacking wide layers of [perceptrons](https://en.wikipedia.org/wiki/Perceptron) - which are inspired by the humble [neuron](https://en.wikipedia.org/wiki/Neuron). In this post we'll dig into a single perceptron. Code is available at [github](https://github.com/HarounH/smol)

# introduction: single fp32 perceptron

Typically, a perceptron is defined as a weighted sum of inputs. In our case, we'll have simplify further - we have just 2 inputs: a variable $x$ and a constant, the number $1$. We'll use $\sigma$ to denote our activation function. For our purposes, a sigmoid function will do. $\hat{y}$ denotes the "output" of a perceptron. Note that unlike brain neurons which emit [all-or-one signals](https://en.wikipedia.org/wiki/All-or-none_law), the perceptron we use outputs a fp32 number. Hence the section title "fp32 perceptron". Mathematically, we can write the perceptron as:

$$
    \hat{y} = \sigma (wx + b)
$$

Where $w$ and $b$ are learnt by optimizing some loss $L = \Sigma_i\ l_i$ over a dataset $ D =  (x_i, y_i)_{i=0}^n $. Code for everything will be available on [this github repo](https://github.com/HarounH/smol).

We'll be generating some really basic toy data as follows

$$
    y_i = \mathbb{1}_{x_i > t}\ \forall i \in \{ 0, 1, 2... n \}
$$

And learning the parameters $w$ and $b$ using Adam. The hyperparameters of our system are:
- dataset size, $n$
- learning rate, $lr$
- batch size as a fraction of dataset size, $batch_size$
- whether parameter $b$ is learnt or set to $1.0$

# learnable bias is important
**Claim**: If we use a constant bias (say 1.0), we make it impossible for the network to win in a range of situations.

Let's look at two situations and see what value of $w$ would be "correct":

<table class="center">
    <tr>
        <th>situation</th>
        <th>ideal $w$</th>
    </tr>
    <tr>
        <td>$t = 0$</td>
        <td>$\infty$</td>
    </tr>
    <tr>
        <td>$t \neq 0$</td>
        <td>$\frac{-1}{t}$</td>
    </tr>
</table>

The first case is obviously impossible with floating point arithmetic and a finite number of steps. To make matters worse, the second step is ALSO super hard because we use sigmoid - as $w$ increasing per step during stochastic gradient descent, we'll see the sigmoid "saturate", i.e., increasing $w$ further offers little benefit, and consequently, the gradients flowing back to $w$ are small. Thus, we'd expect to learn a $w$ that always underestimates the ideal $w$. Indeed, this is the case - in Figure 1, we plot the signed error after 10 epochs (estimated threshold - actual threshold) against learning rate used. We find that error for when bias is learnt (orange line) goes to 0 while error for constant bias gets stuck at a certain amount.

![signed error v/s lr](/assets/images/perceptron1/bias_type_lr.png){:style="display:block; margin-left:auto; margin-right:auto; width:50%; height:50%"}
Figure 1: Signed Error (y-axis) v/s Learning Rate (x-axis) for two types of bias (learnt and constant). We find that constant bias leads to systematically under-estimating our

# learning over epochs
In our experiments we used 10 epochs for each setting, expecting that 10 epochs would be over-kill for a problem like this. However, it was an overkill only under some combinations of hyper-parameters. To demonstrate, Figure 2 and 3 are 3d surface plots of error (z) v/s hyper-parameters, varying over epochs.


![elb](/assets/images/perceptron1/movie_bias_True_elb.gif){:style="display:block; margin-left:auto; margin-right:auto; width:50%; height:50%"}
Figure 2: Error v/s learning rate and batch size. High learning rate and middle range of batch size converge faster than other points on the surface.

![elb](/assets/images/perceptron1/movie_bias_True_eln.gif){:style="display:block; margin-left:auto; margin-right:auto; width:50%; height:50%"}
Figure 3: Error v/s learning rate and n. Unlike Figure 2, it seems that choice of $n$ doesn't really matter here (likely because we have more than sufficient data points to make model good)

For those of you who don't like 3D surface plot GIFs, I've also made figures 4(a-c)

|||
|:----:|:-----:|:----:|
![elb](/assets/images/perceptron1/epoch_vs_batch_size.png){:style="display:block; margin-left:auto; margin-right:auto"} | ![elb](/assets/images/perceptron1/epoch_vs_lr.png){:style="display:block; margin-left:auto; margin-right:auto"} | ![elb](/assets/images/perceptron1/epoch_vs_n.png){:style="display:block; margin-left:auto; margin-right:auto"}
| 4(a) | 4(b) | 4(c) |

Figure 4(a-c): Change of error (more blue is more error) over epochs (X-axis) for various (4a) batch size, (4b) learning rate and (4c) dataset size. We see that reducing learning or increasing batch size make convergence slower.


# Acknowledgements
Thank you [Tong](http://xiaotong.me/) for your discussions along the way!