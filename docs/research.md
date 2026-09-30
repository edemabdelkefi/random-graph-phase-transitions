# Critical graph scaling and systemic payment risk

## Critical-window observables

The critical graph experiment uses

$$G(n,p),\qquad p=n^{-1}+\lambda n^{-4/3},$$

with $n\in\{250,1000,4000\}$, five offsets and 200 independent realizations per cell. Ordered component sizes are $L_1\ge L_2\ge\cdots$. The reported observables are

$$n^{-2/3}L_1,\qquad
\chi=\frac1n\sum_iL_i^2,\qquad n^{-1/3}\chi.$$

Susceptibility is the expected component size of a uniformly sampled vertex, conditional on the realized graph. Excluding the largest component gives $\chi_{finite}=\sum_{i\ge2}L_i^2/(n-L_1)$, set to zero when the graph is connected. The cycle surplus is $m-n+K$, where $K$ is the number of components. It is the dimension of the undirected graph's cycle space, not a count of all simple cycles.

This scaling is motivated by [Aldous, Brownian Excursions, Critical Random Graphs and the Multiplicative Coalescent](https://escholarship.org/uc/item/5s77j58b). Standard errors describe independent Monte Carlo realizations. The 5% and 95% quantiles describe graph-to-graph dispersion; they are not confidence intervals for a mean. Finite-size curves need not collapse exactly or prove the limiting law.

## Financial balance sheets

For each expected degree, 32 independent weighted networks have 80 banks. Each undirected edge is present independently with probability $c/(n-1)$ and receives a Gamma(2, 0.5) weight. The symmetric weight matrix is $W$ with zero diagonal. Set $s_i=\sum_jW_{ij}$ and let $\bar s$ be the average positive strength.

For nonisolated banks, total obligations and interbank liabilities are

$$\bar p_i=s_i/\bar s,\qquad L_{ij}=\beta W_{ij}/\bar s,\qquad \beta=0.65.$$

Isolated banks have unit obligations and no interbank liabilities. Relative liabilities are $\Pi_{ij}=L_{ij}/\bar p_i$. Consequently, nonisolated banks owe 65% internally and 35% to an outside creditor; isolated banks owe 100% outside. Symmetry gives equal aggregate incoming and outgoing interbank face claims for every bank.

Baseline external assets satisfy

$$a_i^0=\bar p_i-\sum_jL_{ji}+\kappa\bar p_i,\qquad \kappa=0.08.$$

All banks therefore begin solvent, with equity $\kappa\bar p_i$. This construction holds interbank share and relative capital fixed among connected banks while allowing bank size and outside asset composition to depend on degree. Comparing expected degrees consequently mixes several structural changes.

## Correlated external stress

Eight shock vectors per network use a common Gaussian factor and independent idiosyncratic factors. For severity $s>0$,

$$f_i=\operatorname{clip}\left(s+0.12[\sqrt{0.35}Z+\sqrt{0.65}Z_i],0,0.95\right),\qquad
a_i=a_i^0(1-f_i).$$

The latent factors have correlation 0.35; clipping changes the correlation of realized losses. The zero-severity cell sets every loss to zero as a deterministic solvency control. It is not the $s\downarrow0$ limit of the dispersed positive-severity design. Common standardized factors are reused across severities and expected-degree cases to reduce comparison noise. Networks within each degree remain independently generated.

## Clearing, uniqueness and numerical certification

The clearing vector solves

$$p^*=T(p^*)=\min\{\bar p,a+\Pi^Tp^*\},$$

where the minimum is componentwise. This represents limited liability and proportional payment priority, following [Eisenberg and Noe, Systemic Risk in Financial Systems](https://ms.mcmaster.ca/tom/Research%20Papers/EiseNoe01.pdf).

The row sums of $\Pi$ are at most $\beta<1$. Since componentwise clipping is nonexpansive,

$$\lVert T(p)-T(q)\rVert_1\le\lVert\Pi^T\rVert_1\lVert p-q\rVert_1
\le\beta\lVert p-q\rVert_1.$$

The map therefore has a unique fixed point. Iterating from $\bar p$ decreases monotonically; iterating from zero increases monotonically. The experiment computes both solutions. With residual $r=\lVert T(p)-p\rVert_1$, the a posteriori error bound is

$$\lVert p-p^*\rVert_1\le r/(1-\beta).$$

The solver stops only when this certificate is at most $10^{-10}$. The guarantee relies on a strictly positive outside debt share; the API rejects systems outside that sufficient contraction condition. This is more restrictive than general clearing-system existence theory but makes this experiment auditable.

The direct-only reference assumes all interbank counterparties pay in full: $p^{direct}=\min\{\bar p,a+\Pi^T\bar p\}$. It records defaults caused by the external shock before payment feedback. Clearing can add defaults and creditor losses.

## Local derivatives and spectral amplification

Away from default switching boundaries, let $D$ be the defaulted banks and $S$ the solvent banks. Solvent payments are fixed at face value. The default payments satisfy

$$p_D=(I-\Pi_{DD}^T)^{-1}(a_D+\Pi_{SD}^T\bar p_S).$$

Hence $\partial p_D/\partial a_D=(I-\Pi_{DD}^T)^{-1}$ and the other derivative entries are zero on this unchanged default set. Its spectral radius is below one, and its L1 resolvent norm is bounded by $1/(1-\beta)$. Finite-difference tests compare this derivative with independently recomputed payments.

Let $v_i=1-\sum_j\Pi_{ij}$ be each bank's outside debt share. Outside loss in original outside claims is

$$\mathcal L=\frac{v^T(\bar p-p^*)}{v^T\bar p}.$$

The unnormalized marginal outside loss per dollar of external asset depletion is $v^TJ$. Without bankruptcy costs, $v^Tp+\sum_i\mathrm{equity}_i=\sum_i a_i$: interbank transfers cancel in aggregate. Payment shortfalls can amplify internally without creating extra aggregate resource destruction. The experiment distinguishes this model from fire-sale or bankruptcy-cost mechanisms.

## Clustered tail-risk inference

Within each degree/severity cell, 32 networks each carry eight related shocks, for 256 scenarios. Empirical 95% expected shortfall averages the largest $\lceil0.05\cdot256\rceil=13$ outside losses. Direct loss, clearing loss and their difference are saved for every scenario.

A 2,000-draw bootstrap resamples whole networks and keeps all eight shocks together. Intervals describe uncertainty over the network-and-shock design. They are pointwise, and 32 independent clusters limit precision. Cross-degree intervals alone do not test a paired topology effect.

The results also retain per-graph critical-window observables, every financial scenario, default-set sensitivities, upper/lower agreement, fixed-point certificates, complete protocols, environment versions and SHA-256 checksums. All financial exposures and shocks are synthetic; neither calibrated bank default probabilities nor an empirical systemic-risk forecast is claimed.
