TODOs
- [x] make the repo private
- [x] explain in the README.md exactly what I have to do to run
      (no notebooks, just scripts)

RESEARCH QUESTIONS
- [x] weird learning behavior: exploding gradients
- [x] nr of parameters / memory (15 GB)
- [ ] hyena / low-rank functions


# Intervals / Low-rank

Assume we have a function
$$
    f: [T] \times [T] \to \mathbb R^d.
$$
For the moment $d=1$. Think of $f$ as encoding information
about the ``interval-behaviour'' of some time series 
$$
    x: [T] \to \mathbb R^e.
$$.
So for every $s < t$ we get some 'description' of what
the time series does on the interval.
($f(s,t) := (x(t) - x(s), \max_{s\le r\le t} x(r), \sum_{s\le r\le t} x(r))$).

One could do naive attention on $f$, as follows
$$
    \sum_{s,t} e^{ f(s,t)^\top q } v(s,t), 
$$
for some ''value'' $v: [T] \times [T] \to \mathbb R^d$ and ''query'' $q \in \mathbb R^d$. This costs $\mathcal O(T^2)$.
If we want to do self-attention, it becomes $\mathcal O(T^4)$.

[TODO convince yourself that one could also do this with hyena,
at this high cost.]\

If d = 2, we can write the function as:
$$
    f(x,y) = [O_1, O_2] = [f_a(x,y), f_b(x,y)]
$$
and continue the following process, but for now, let us build a case for d = 1 for simplicity.
<!-- If we want to apply attention or Hyena to $f$, we need to -->
<!-- do $\mathcal O(T^2)$ calculations. If we want to do self-attention -->
<!-- it becomes $\mathcal O(T^4)$. Goal: speed this up. -->


Then, the first hyena convolution
(Usually we think of the interval $[T] := \{1, \ldots, T\}$,
now we can think of doing convolution over $[T]^2$)
$$
    (h \ast f)(t,t')
    = \sum_{(s,s')\in [T]^2} h(t-s,t'-s') f(s, s').
$$

Why does FFT not work?
Mathematically, we know
$$
    \hat{(h \ast f)}
    =
    \hat{h} \hat{f}
$$
But FFT only works in one dimension.
[TODO I think it is because FFT really depends on the cyclic group structure of $\mathbb Z / T \mathbb Z$.]


Assume that $f$ is **low-rank**.
In the rank-1 case this just means
$$
    f(t, t') = f_1(t) f_2(t'),
$$
for some functions $f_i: [T] \to \mathbb R^d$.
(For example, with the time series above,
we could take $f(s,t) := e^{x(t)-x(s)}$).
Assume also
$$
    h(t, t') = h_1(t) h_2(t').
$$

Then
$$
    (h \ast f)(t,t')
    = \sum_{(s,s')\in [T]^2} h_1(t-s) h_2(t'-s') f_1(s) f_2(s')
    = \left( \sum_{s\in [T]} h_1(t-s) f_1(s) \right) \left( \sum_{s'\in [T]} h_2(t'-s') f_2(s') \right)
    =: F_1(t) F_2(t'),
$$
and $F_1, F_2$ can be computed at a total cost of $\mathcal O(2 T \log(T))$.

Now continue with the first hyena multiplication
$$
    f(t,t') (h \ast f)(t,t') = f_1(t) f_2(t') F_1(t) F_2(t')
    = k_1(t) k_2(t'),
$$
where $k_i(t)=f_i(t) F_i(t)$.
And finally, the second hyena convolution can be computed at a cost of $\mathcal O(2 T \log(T))$
(if the filter is rank-1 as well).

[TODO] What happens if $f$ is symmetric, i.e.
$$
    f(t,t') = f(t',t).
$$
Does this simplify the calculations?

[TODO cook up articial examples where
attention to intervals is relevant,
and/or find real-world examples]

[TODO do the calculations when
the function $f$ and the filters are rank-k]

$$
    f(t,t') = f_1(t) f_2(t') + f_3(t) f_4(t').
$$

$$
    h(t,t') = h_1(t) h_2(t') + h_3(t) h_4(t').
$$

Then
$$
    (h \ast f)(t,t')
    = \sum_{(s,s')\in [T]^2} \left( h_1(t-s) h_2(t'-s') + h_3(t-s) h_4(t'-s') \right) \left( f_1(s) f_2(s') + f_3(s) f_4(s') \right)\\
    = 
    \sum_{(s,s')\in [T]^2} 
    \Big(
    h_1(t-s) h_2(t'-s') f_1(s) f_2(s')
    +
    h_1(t-s) h_2(t'-s') f_3(s) f_4(s') \\
    +
    h_3(t-s) h_4(t'-s') f_1(s) f_2(s')
    +
    h_3(t-s) h_4(t'-s') f_3(s) f_4(s')
    \Big)\\
    =
    \sum_{s\in [T]} h_1(t-s) f_1(s) \sum_{s'\in [T]} h_2(t'-s') f_2(s')
    +
    ...\text{ three more }.
    % = \left( \sum_{s\in [T]} h_1(t-s) f_1(s) \right) \left( \sum_{s'\in [T]} h_2(t'-s') f_2(s') \right)
    % =: F_1(t) F_2(t'),
$$


Example f rank 5, all h's rank 3.
Then: after first hyena convolution, we get rank 15.
After the product, we get rank 75.
After the next convolution we get rank 225.

For rank n function and rank m filter, we get rank $n*m$.

Proof:

$$
    f(t,t') = \sum_{j=1}^{n} f_{2j-1}(s) f_{2j}(s')
$$

$$
    h(t,t') = \sum_{i=1}^{m} h_{2i-1}(t) h_{2i}(t')
$$

as we know:

$$
    (h \ast f)(t,t')
    = \sum_{(s,s')\in [T]^2} h(t-s,t'-s') f(s, s').
$$

Therefore for n rank function and m rank filter:

$$
    (h \ast f)(t,t') = \sum_{(s,s')\in [T]^2} \sum_{i=1}^{m} h_{2i-1}(t-s) h_{2i}(t'-s') \sum_{j=1}^{n} f_{2j-1}(s) f_{2j}(s') 
$$

$$
    (h \ast f)(t,t') = \sum_{i=1}^{m} \sum_{j=1}^{n} \left( \sum_{s\in [T]} h_{2i-1}(t-s) f_{2j-1}(s) \right) \left( \sum_{s'\in [T]} h_{2i}(t'-s') f_{2j}(s') \right)
$$


Therefore, m combinations of n rank, i.e. m times n unique pairs. 

Hence proved that for n rank function and m rank filter, we get a hyena convolution output .

Let us try to implement hyena convolutions for low rank functions - \
For this, we assume a function f 
$$
    f: [T] \times [T] \to \mathbb R^d.
$$
Over the dataset with the following details:

1. Time series data of the prices of one of the given forex pairs: AUDUSD or EURUSD. 
2. Function with two inputs t_1 and t_2 represting two timestamps.
3. Output of the function being the percentage price difference between the interval t_1 and t_2 giving a rank 2 function.

Source(Data): https://www.axiory.com/trading-tools/metatrader-historical-data

Therefore:

$$
    f(t_1, t_2) = \frac{p(t_2) - p(t_1)}{p(t_1)} * 100
$$
where, \
p(t) = price at time t

We can write the function f (a low rank function) as:

$$
    f(t_1, t_2) = f_1(t_1).f_2(t_2) + f_3(t_1).f_4(t_2)
$$

where,

$$
    f_1(t_1) = \frac{1}{p(t_1)}
$$

$$
    f_2(t_2) = 100* p(t_2)
$$

$$
    f_3(t_1) = -1
$$

$$
    f_4(t_2) = 100
$$

$$
    (h \ast f)(t,t') = \sum_{s\in [T]} h_1(t-s) f_1(s) \sum_{s'\in [T]} h_2(t'-s') f_2(s')\\ 
    + \sum_{s\in [T]} h_3(t-s) f_1(s) \sum_{s'\in [T]} h_4(t'-s') f_2(s')\\ 
    + \sum_{s\in [T]} h_1(t-s) f_3(s) \sum_{s'\in [T]} h_2(t'-s') f_4(s')\\ 
    + \sum_{s\in [T]} h_3(t-s) f_3(s) \sum_{s'\in [T]} h_4(t'-s') f_4(s')
$$


# Systematic approach to creating low-rank interval functions

We saw the example, for any $n \ge 1$,
$$
    f^{[1^n]}(s,t) := \sum_{s \le i < t} (x_i)^n.
$$
This is low-rank: because we can write it as
$$
    f^{[1^n]}(s,t)
    = f^{[1^n]}(0,t) - f^{[1^n]}(0,s).
    = f^{[1^n]}(0,t) 1(s) - 1(t) f(0,s).
$$

Let us try
$$
    f^{[1][1]}(s,t)
    := \sum_{s \le i_1 < i_2 < t} x_{i_1} x_{i_2}
    := \sum_{i_2=s}^{t} x_{i_2} \sum_{i_1=s}^{i_2} x_{i_1}.
$$

Then can one check, by hand,
$$
    f^{[1][1]}(0,t) - f^{[1][1]}(0,s)
    =
    \sum_{0 \le i_1 < i_2 < t} x_{i_1} x_{i_2}
    -
    \sum_{0 \le i_1 < i_2 < s} x_{i_1} x_{i_2}
    =
    \sum_{s \le i_1 < i_2 < t} x_{i_1} x_{i_2}
    +
    \sum_{i_1 < s \le i_2 < t} x_{i_1} x_{i_2}
    =
    f^{[1][1]}(s,t)
    +
    \sum_{i_1 < s} x_{i_1} \cdot \sum_{s \le i_2 < t} x_{i_2}
    =
    f^{[1][1]}(s,t)
    +
    f^{[1]}(0,s) f^{[1]}(s,t).
    =
    f^{[1][1]}(s,t)
    +
    f^{[1]}(0,s) ( f^{[1]}(0,t) - f^{[1]}(0,s) )
$$
since for $0 \le i_1 < i_2 < s$ either
- $i_1 < i_2 < s$
- $s \le i_1 < i_2 < t$
- $i_1 < s \le i_2 < t$.

Summarizing:
$$
    f^{[1][1]}(s,t)
    =
    f^{[1][1]}(0,t) - f^{[1][1]}(0,s)
    -
    f^{[1]}(0,s) f^{[1]}(0,t) 
    +
    f^{[1]}(0,s) f^{[1]}(0,s).
$$

What about
$$
    \sum_{0 \le i_1 < i_2 < ... < i_k < t} x^{\alpha_1}_{i_1} ... x^{\alpha_k}_{i_k} ??
$$

see:
https://link.springer.com/article/10.1007/s10440-020-00333-x
and
https://arxiv.org/abs/2009.08443