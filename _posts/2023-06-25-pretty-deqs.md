---
title: hello differential equations (2/2)
author: haroun7
date: 2023-06-25 11:33:00 +0800
math: true
mermaid: true
---
In the previous part we did a shoddy ramp-up to differential equations (both ordinary and partial). In this more involved post, we'll look closely at Gray Scott equation, which can be used to generate very pretty patterns. In particular, I want to understand Figure 1 - which demonstrates phase change.

First off, some quick googling brought me to [this great reference with a simulator.](https://itp.uni-frankfurt.de/~gros/StudentProjects/Projects_2020/projekt_schulz_kaefer/#:~:text=The%20Gray%2DScott%20system%20is,used%20up%2C%20while%20is%20produced.). I encourage playing with the simulator and making some observations for yourself!

<figure style="text-align: center;">
$$
\begin{align*}
\frac{\partial u}{\partial t} &= D_u \nabla^2 u - uv^2 + F(1-u) \\
\frac{\partial v}{\partial t} &= D_v \nabla^2 v + uv^2 - (F+k)v
\end{align*}
$$
<figcaption>Gray-Scott differential equations: $u$ and $v$ are observables, $D_u$ and $D_v$ are coefficients of diffusion, $F$ is Feedrate, $k$ is killrate.</figcaption>
</figure>

# Arc 1: Phases Exists
One behavior we observe is that a small change in parameters can cause the output image to change significantly. To further convince yourself, try the default mitosis setting in the simulator linked earlier, then reduce kill rate to 0.061.

> NOTE: If you're trying to implement your own simulator, be very very careful about hyper-parameters and exact definition of primitives such as laplacian :)

<figure style="text-align: left;">
  <img src="{{ site.baseurl }}/assets/images/ped_2/figure1_phase.png" alt="Alt Text">
  <figcaption>Figure 1: Concentration of component after 9000 steps as feedrate (X-axis) and killrate (Y-axis) are changed. Notice that there are clear conditions that lead to one of two degenerate solutions. Further, behavior near the boundary seems quite unpredicable, but are also where some beautiful patterns form. </figcaption>
</figure>

# Arc 2: Mitosis and Why
One cool pattern is mitosis, which you can see in Figure 2. For specific combinations of hyperparameters, we observe that a single ball of concentrate separates into multiple balls, and the process continues. Figure 2 shows that a 25% deviation in either feedrate or killrate can change the pattern altogether - too small a kill rate and the component takes over the entire screen.

<figure style="text-align: center;">
  <img src="{{ site.baseurl }}/assets/images/ped_2/evolution.png" alt="Alt Text">
  <figcaption>Figure 2: Rows from top to bottom are increasing number of Euler steps. Feed and killrates per column are specified at the top. We see that if kill rate is 0.045, mitosis does not happen at all. However, even at killrate=0.045, mitosis happens at some specific feedrate (=0.025). If feedrate is too low, then the component just diffuses slowly. Meanwhile if feedrate is too high, component remains connected. </figcaption>
</figure>

To dig further into this problem, we plot states at a much larger frequency in Figure 3, but also include killrate=0.075. The main takeaway from Figure 3 is that changes are relatively slow. Splitting from C shape to two blobs takes 10 periods, splitting into 4 blobs takes another 10 periods. Similarly, Figure 2 needs about 1-1.5 periods for each split by 2.

<figure style="text-align: center;">
  <img src="{{ site.baseurl }}/assets/images/ped_2/early_iterations.png" alt="Alt Text">
  <figcaption>(Bonus) Figure 3: Like figure 2, but for killrate=0.075 as well. </figcaption>
</figure>

To investigate what changes in one step when hyperparameters change, in Figure 4, we take a fixed step (an intermediate state from a path that does mitosis), and look at the three terms in $\frac{dv}{dt}$ - i.e., $D_v \nabla^2 v$, $uv^2$ and $(F+k)v$. Note that feedrate and killrate only directly affect the last term. We expect that changes in feedrate and killrate would not affect the first two terms, and would linearly affect the last term.


<figure style="text-align: center;">
  <img src="{{ site.baseurl }}/assets/images/ped_2/early_iterations.png" alt="Alt Text">
  <figcaption>Figure 4: Top to bottom row: killrates=0.045, 0.06, 0.75. Each with feedrate=0.02. Left to Right: (a): $v_t$, (b) $D_v \nabla^2 v$, (c) $uv^2$, (d) $(F+k)v$ and (e) $v_{t+1}$. Notice that as expected, we don't see any major differences. If we look really hard, (d) looks a little brighter in the lowest row. </figcaption>
</figure>

The last thing that we can look at is the effect on mitosis pattern if we change the feedrate or killrate in between. To simulate this, we take an intermediate state from the mitosis configuration (feedrate=0.025 and killrate=0.06), then perform three runs: set killrate to 0.045, 0.06 and 0.075. The results are in Figure 5.


<figure style="text-align: center;">
  <img src="{{ site.baseurl }}/assets/images/ped_2/change_in_between.png" alt="Alt Text">
  <figcaption>Figure 5: Top to bottom row: increasing number of euler integration steps. Left to right: killrates=0.045, 0.06, 0.75.</figcaption>
</figure>

# Arc 3: Wrapping it up
imho, there's some interesting problems here and I definitely don't understand the Gray Scott patterns. There might be other differential equations that have these unique properties too. However, thus far, I have not identified any study that is fundamental to these differential equations. In the next part, we'l look at stochastic differential equations really quick and then move onto machine learning.