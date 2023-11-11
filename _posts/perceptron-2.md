---
title: (2/3) single fp32 perceptron
author: haroun7
date: 2023-09-26 11:33:00 +0800
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



## $n>1$, vector input
> NOTE: I take a small jump for vector inputs. Please refer to the pdefinition section in perceptron wiki](https://en.wikipedia.org/wiki/Perceptron#Definition) and [Steps section](https://en.wikipedia.org/wiki/Perceptron#Steps) for how we update equation below came to be.

In the case of $n=1$, our gradients and update of weights were a coin flip. As the batch size increases, the possible compositions of a batch increase exponentially since each item can be $y \in \\{ 0,1 \\} $. We'll use $B_t$ to denote the datapoints that were in batch at time $t$. The function at time $t+1$ in terms of stuff at time $t$ is:

$$
\begin{align*}
f(x ; w_{t+1}, b_{t+1}) =& \sigma(w_{t+1}^T x + b_{t+1})
\\ =& \sigma((w_{t} + \eta \left( \sum_{i\in B_t} x_i(y_i-\sigma(w_t^T x_i+b_t)) \right) )^T x + b_{t} + \eta \left( \sum_{i\in B_t} (y_i-\sigma(w_t^T x_i+b_t)) \right))
\\ =& \sigma(w_t^T x + b_t + \eta\left( \sum_{i\in B_t}(x_i^T x + 1)(y_i-\sigma(w_t^T x_i+b_t)) \right))
\end{align*}
$$

This is a tricky situation to deal with because there are two relationships to care about: (a) the composition of $B_t$ (b) whether or not $x \in B_t$. However, we can still focus on the third term within the sigmoid, $G=\left( \sum_{i\in B_t}(x_i^T x + 1)\epsilon_i^t \right)$ where $\epsilon_i^t=y_i-\sigma(w_t^T x_i+b_t)$. Note that we drop $\eta$ - we know that small enough learning rate will simply move the final update closer to What matters is whether that is positive or negative - we want it to be positive if the $y$ for $x$ is 1, otherwise we want it to be negative.

Although the case where $x\in B_t$ is less meaningful in today's world where we don't train for more than an epoch, we still want to understand how the prediction on a training dataset evolves depending on what the batch is composed of.

$$
\begin{align*}
G &= \left( \sum_{i\in B_t}(x_i x + 1)\epsilon_i^t \right)
\\ =& (x_j^T x_j + 1)\epsilon_j + \left( \sum_{i \in B_t \setminus \{ j \} }(x_i^T x_j + 1)\epsilon_i^t \right)
\end{align*}
$$

# on floats