---
title: (2/3) single fp32 perceptron
author: haroun7
date: 2023-10-13 11:33:00 +0800
math: true
mermaid: true
---

# introduction
In the previous post we looked at empirical behavior of a single perceptron. In this post we'll **manually** dig into a single perceptron because, often, nothing beats writing out the math yourself. We know the closed form for BCE loss over mini-batches, and also the closed form for gradient descent. It is as below, where $f(x; \theta)$ is the perceptron, $x,y,i,l,L$ have their usual meanings, while $\theta$ represents $w$ and $b$.

$$
\begin{align*}
L(\{ x_i, y_i \}_i , f(x ; \theta)) &= \sum_i l(y_i, f(x_i ; \theta))
\\  &= - \sum_i \; y_i \log(\sigma(wx_i + b)) + (1 - y_i) \log(1 - \sigma(wx_i + b))
\\  &= - \sum_i \; y_i \log(\sigma(wx_i + b)) + (1 - y_i) \log(\sigma(-(wx_i + b)))
\end{align*}
$$

Note that in the above loss function, we should use mean rather than summation, but for the purposes of this blog post, that is okay. The gradients we're interested in are $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$. We'll represent them as $\frac{\partial L}{\partial \theta}$, and they can be derived as follows:

$$
\begin{align*}
\frac{\partial L}{\partial \theta} =& \sum_i \frac{\partial l(y_i, f(x_i; \theta))}{\partial f(x_i;\theta)} \frac{\partial f(x_i;\theta)}{\partial wx_i + b} \frac{\partial wx_i + b}{\partial \theta}
\\ =& \sum_i \left( - (\frac{y_i}{f(x_i)}) - \frac{1 - y_i}{1 - f(x_i)} \right) \left( f(x_i) (1-f(x_i) ) \right) \frac{\partial wx + b}{\partial \theta}
\\ =& \sum_i \left( - \frac{y_i - y_i f(x_i) - f(x_i) + f(x_i)y_i}{f(x_i) (1-f(x_i))} \right)\left( f(x_i) (1-f(x_i) ) \right) \frac{\partial wx_i + b}{\partial \theta}
\\ =& \sum_i -(y_i - f(x_i))\frac{\partial wx_i + b}{\partial \theta}
\end{align*}
$$


Then, the parameter update equations, using learning rate $\eta > 0$ are:

$$
\begin{align*}
w_{t+1} \leftarrow& w_t + \eta \sum_i x_i \left(y_i - \sigma(w_t x_i + b_t)\right)
\\ b_{t+1} \leftarrow& b_t + \eta \sum_i \left(y_i - \sigma(w_t x_i + b_t)\right)
\end{align*}
$$

How does this affect the predictions? If batch size is 1, we expect that the prediction should become more correct on the next iteration. We also see how this argument falls apart for larger batch sizes - each update will be dominated by the majority class/behavior. Further, in higher dimensions, we have as many "dimensions" as the size of $w$ - and each dimension has its own gradient update that depends on other dimensions.

> TODO: perheps we could try some tricks to make updates from one data point not affect the others... we leave that thought to a different day :)

# examples
Lets walk through some examples. In particular, we'll deal with cases where $n=1,2,4$ and we'll look at what the updates could be for different batch sizes.

## $n=1$
With just one data point (call it $x,y$), the batch size has to 1, the prediction at $t+1$ is:

$$
\begin{align*}
f(x ; w_{t+1}, b_{t+1}) =& \sigma(w_{t+1} x + b_{t+1})
\\ =& \sigma((w_{t} + \eta x(y-\sigma(w_tx+b_t)))x + b_{t} + \eta (y-\sigma(w_tx+b_t)))
\\ =& \sigma(w_tx + b_t+ \eta(x^2 + 1)(y-\sigma(w_tx+b_t)))
\end{align*}
$$

Notice that since $0 \lt \sigma(.) \lt 1$, if $y=1$ then $f(x; w_{t+1}, b_{t+1}) \gt f(x; w_{t}, b_{t})$ and if $y=0$ then $f(x; w_{t+1}, b_{t+1}) \lt f(x; w_{t}, b_{t})$

![f(x; \theta_t) and f(x; \theta_t+1)](/assets/images/perceptron2/sigmas.png){:style="display:block; margin-left:auto; margin-right:auto; width:80%; height:80%"}
Figure 1: Plot comparing $sigmoid(x)$ (gray), v/s $sigmoid(x - k * sigmoid(x))$ (dashed lines) and $sigmoid(x - k (1-sigmoid(x)))$ (dotted lines). We see that deviation from gray line increases in a seemingly non-linear manner wrt $k$. We also see that large values of $k$ affect the shape of output function significantly.

imho, this behavior will be more interesting if we look at larger values of $n$ and higher dimensionality. I believe, that is how we would end up at [neural tangent kernels](https://en.wikipedia.org/wiki/Neural_tangent_kernel).

> Digression on **Constant Bias** - in our previous post, we did a quick study of constant bias. If we had a constant bias here, then $f$ at time $t+1$ in terms of time $t$ would be as below. Also considering that the dataset used previous was $x \sim N(0, 1)$, these updates would correspond to cases where $k<1$ in Figure 1.

$$
\begin{align*}
f(x ; w_{t+1}, b_{t+1}) =& \sigma(w_{t+1} x + b_{t+1})
\\ =& \sigma((w_{t} + \eta x(y-\sigma(w_tx+b_t)))x + b_{t})
\\ =& \sigma(w_tx + b_t+ \eta x^2 (y-\sigma(w_tx+b_t)))
\end{align*}
$$

## $n>1$

> NOTE: Starting this section, we'll start using a little bit of vector notation. For a primer on vector inputs to perceptrons, please refer to the ["Definition" section in perceptron wiki](https://en.wikipedia.org/wiki/Perceptron#Definition) and ["Steps" section](https://en.wikipedia.org/wiki/Perceptron#Steps) for how we update equation below came to be.

In the case of $n=1$, our gradients were one of two forms depending on $y$. As the batch size increases, the possible compositions of a batch increase since each item can be $y \in \\{ 0,1 \\} $. The number of combinations appears exponential in batch size, but there could be duplicates. However, intuitively, there should be at least be a linear number of combinations possible. Below, we'll use $B_t$ to denote the datapoints that were in batch at time $t$. The perceptron at time $t+1$ in terms of time $t$ is:

$$
\begin{align*}
\textit{Let } \epsilon_{t}^{i} =&\ y_i-\sigma(w_t^T x_i+b_t)\textit{    Then:}
\\ w_{t+1} =&\  w_{t} + \eta \left( \sum_{i\in B_t} x_i\epsilon_t^i \right)
\\ b_{t+1} =&\  b_{t} + \eta \left( \sum_{i\in B_t} \epsilon_t^i \right)
\\ f(x ; w_{t+1}, b_{t+1}) =&\  \sigma(w_{t+1}^T x + b_{t+1})
\\ =&\  \sigma((w_{t} + \eta \left( \sum_{i\in B_t} x_i \epsilon_t^i \right) )^T x + b_{t} + \eta \left( \sum_{i\in B_t} \epsilon_t^i \right))
\\ =&\  \sigma(w_t^T x + b_t + \eta\left( \sum_{i\in B_t}(x_i^T x + 1)\epsilon_t^i \right))
\end{align*}
$$


We can also think of these updates geometrically - $\left[\textbf{w}_t\ b_t\right]\left[ \genfrac{}{}{0pt}{}{\textbf{x}}{1} \right] = 0$ represents a decision boundary that is a hyperplane over $\textbf{x}$ space, and the gradient update is a transformation of the hyperplane, and update to $w_t$ and $b_t$ represents the transformation. We also note that $\left[\textbf{w}_t\ b_t\right]$ represents the normal to the hyper-plane.

Alternatively, the parameters $w_t$ and $b_t$ can be thought of as coordinates in some space with a loss function as the third axis, and we're moving in that space in a greedy way (gradient descent). In the figures below, we plot the trajectory of our network in weight-bias space, with each circle indicating a step.


![Weights and Biases per step as learning rate changes](/assets/images/perceptron2/trajectory_lr.png)
Figure 2: Weights and Biases per step as learning rate changes. Orange line indicates weight and bias such that $weight * threshold + bias = 0$. As we expect, too small an lr fails to converge to the correct solution. Larger lr has a lot of noise - its easy to just pick a final epoch and stop, but we could definitely end up at a non-optimal value.

![Weights and Biases per step as batch size changes](/assets/images/perceptron2/trajectory_bs.png)
Figure 3: Weights and Biases per step as batch size changes. Orange line indicates weight and bias such that $weight * threshold + bias = 0$. The intuition is much like Figure 2 except that instead of small learning rate, we have large batch size that prevents convergence. Why does this happen? The expected loss is the same for all batch sizes, right? There are some references [1](https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu), [2](https://stats.stackexchange.com/questions/316464/how-does-batch-size-affect-convergence-of-sgd-and-why), [3](https://wandb.ai/ayush-thakur/dl-question-bank/reports/What-s-the-Optimal-Batch-Size-to-Train-a-Neural-Network---VmlldzoyMDkyNDU#:~:text=However%2C%20for%20a%20non%2Dconvex,converge%20on%20the%20optimal%20solution.). In our case, we notice that the learning curves are tending towards the "correct" solution, so perhaps its because we simply did not do enough steps.

> NOTE: unpopular opinion: SGD is sample inefficient as evidenced by Figure 2 and 3 - many choices of hyper-parameters simply don't converge as fast as they could.


# on floats
So far we've treated these numbers are real numbers rather than their implementation - floating point numbers. For those familiar with [numerical stability](https://en.wikipedia.org/wiki/Numerical_stability), issues around floating point numbers and the gradient update step we described earlier are somewhat clear - we run risks of [underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow#:~:text=The%20term%20arithmetic%20underflow%20(also,central%20processing%20unit%20(CPU).) and other numerical errors. That said, SGD as an algorithm is numerically stable.

In this section, we will ignore that our algorithm is dealing with real numbers. Instead, we treat floats as a bunch of bits. Consequently, arithmetic is just a collection of weird circuits. For example, 32 bit floating point operations are described [here](https://digitalsystemdesign.in/floating-point-multiplication/).