# Clustering sliding windows by an iterative multiscale $k$-medoids selection heuristic under a cyclical invariant distance

Denote $[p]=\lbrace 0,1,\ldots,p-1\rbrace$.

Given a time series $\lbrace x(t)\rbrace_{t\in[m]}$ and a window length $w$, the sliding windows generated from $x(t)$ of length $w$ are the vectors

$$
x_i=[x(i),x(i+1),\ldots,x(i+w-1)]^T, \quad\text{where}\quad i\in[m-w+1].
$$

Denote the corresponding $z$-normalized windows by $\lbrace z_i\rbrace_{i\in[m-w+1]}$.

We aim to detect recurrent patterns encoded in the time series by selecting $k$ representative windows from $\lbrace z_i\rbrace_{i\in[m-w+1]}$. 

To this end, we use an iterative multiscale $k$-medoids selection heuristic motivated by the geometry under the cyclic-shift-invariant distance

$$
d(x,y):=\min_{s\in[w]}
\left(
\sum_{r\in[w]} \bigl(x(r+s)-y(r)\bigr)^2
\right)^{1/2},
\qquad x,y\in\mathbb{R}^w,
$$

where $r+s$ is interpreted modulo $w$.

## Detailed implementation

* Required input: all z-normalized windows $`\lbrace z_i \rbrace_{i\in [m-w+1]}`$, the number of selected windows $`M`$ at each iteration, maximum iterations $`t_{\max}`$, and convergence tolerance $`\varepsilon`$.
* Final output: $`k`$ medoids $`\lbrace y_j^{t+1} \rbrace_{j\in[k]}`$ and a partition of $`M`$ windows from $`\lbrace z_i \rbrace_{i\in [m-w+1]}`$ into their $`k`$ Voronoi cells under $`d`$.

### Step 1: Initialize $k$ candidate windows

Set $t=0$. Compute the pairwise distance matrix

$$
D=\bigl[d(z_i,z_j)\bigr]_{i,j\in[m-w+1]}.
$$

Then use $k$-means++ under the distance $d$ to select $k$ initial candidate windows

$$
\lbrace y_j^t\rbrace_{j\in[k]}\subset \lbrace z_i\rbrace_{i\in[m-w+1]}.
$$

### Step 2: Select windows from popular $d$-distance ranges

For each bin-width $`\eta>0`$, define a sequence of level sets for each $`x\in \mathbb{R}^w`$ by

$$I_{p,\eta}(x) = \lbrace z_i: d(z_i, x) \in [p\eta, (p+1)\eta ) \rbrace, \quad \forall p\in \mathbb{N}$$

so that $`\lbrace I_{p,\eta}(x) \rbrace_{p\in \mathbb{N}}`$ forms a partition of the z-normalized windows.

For each $`j\in [k]`$ and each bin-width $\eta \in \mathcal{B}=\lbrace 2^{-5}, 2^{-4}, 2^{-3}, 2^{-2},2^{-1}\rbrace,$ 
collect the $`\frac{1}{\eta}`$-most populated level set among $`\lbrace I_{p,\eta}(y_j^t) \rbrace_{p\ge 0}`$, and denote their union by $`S^t_{\eta}(y_j^t)`$.
Then construct the intersection

$$S^t(y_j^t):= \bigcap_{\eta\in \mathcal{B}}S^t_{\eta}(y_j^t).$$

This multi-scale approach aims to make the selection robust against different bin-widths.


### Step 3: Select windows with the smallest local variation in $d$-distance

Select $`M`$ sliding windows from

$$S^t = \bigcup_{j\in[k]} S^t(y_j^t)$$

with the smallest average absolute consecutive difference in $`d(\cdot, y_j^t)`$, i.e., these $`M`$ windows correspond to an arbitrary minimizer of the following problem

$$C^t := \mathop{\arg\min}_{C \subset S^t \setminus \lbrace z_0 \rbrace, |C|=M} \quad \sum_{z_p \in C} \frac{1}{k} \sum_{j\in[k]} \Big| d_G(z_p, y_j^t) - d_G(z_{p-1}, y_j^t) \Big|$$

### Step 4: Voronoi cell partition under $d$

Partition the selected subset $`C^t`$ using Voronoi cells of $`y_j^t`$'s, where the Voronoi cells are given by

$$V_j^t := \lbrace z_i \in C^t : d(z_i, y_j^t) \le d(z_i, y_{q}^t),\ \forall q \neq j \rbrace,$$

breaking ties arbitrarily.


### Step 5: Update medoids

Within each Voronoi cell $`V_j^t`$, compute a new medoid $`y_j^{t+1}`$ by

$$y^{t+1}_j \in \mathop{\arg\min}_{z_i \in V_j^t} \sum_{z_p \in V_j^t} d_G(z_p, z_i)$$

### Step 6: Convergence check

If

$$\sum_{j\in [k]} d_G(y_j^{t+1}, y_j^t) \le \varepsilon$$

or $`t \ge t_{\max}`$, stop iterations and report the final $`k`$ medoids $`\lbrace y_j^{t+1} \rbrace_{j\in [k]}`$ and their Voronoi cell members. Otherwise, let $`t \leftarrow t+1`$ and repeat Step 2 to Step 6.
