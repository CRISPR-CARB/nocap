# Distribution Estimation
Given a symbolic probability distribution output by an identification algorithm, how do we actually compute it using data?

* Need to use some nonparametric estimation technique for continuous variables
  * KDE
  * Doubly robust estimators

* Gaussian KDE
  * Use a multivariate Gaussian KDE to estimate the joint distribution based on the observations/samples
  * Can a Gaussian kernel be used properly if the underlying data is not Gaussian / can we make a non-parametric assumption and still use a Gaussian kernel?
    * KDE is a non-parametric technique, sum's up a bunch of small Gaussian kernels to produce a smooth estimation of the actual probability density function
  * How to choose bandwidth?
    * Silverman's rule of thumb
    * Scott's rule (relies upon standard deviation)
    * Sheather-Jones method ("best overall choice for non-Gaussian data because it captures multi-peaked distributions")
* Conditional multivariate KDE
  * Supports estimating conditional probabilities.
    * Requires smoothness assumptions in outcome and conditioning variables
  * Can use joint-KDE ratio, marginalize out variables of the joint and use conditional prob rule.
    * Can be problematic
* Continuous variables marginalization
  * Need to use integration, use SciPy quadrature or nquad

## [Building Efficient Estimators](https://alejandroschuler.github.io/mci/4-building-efficient-estimators.html)
* Different efficient estimators may behave the same asymptotically, but might wildly vary with smaller sample sizes and when assumptions are not met
  * Can build efficient estimators without making any scientifically meaningful statistical assumptions

## [G-computation](https://marginaleffects.com/chapters/gcomputation.html)
* If we are not interested in computing distributions, and simply computing expectations based on a symbolic probability distribution, that is an easier problem to solve
  * With this we can compute ATE
* G-computation is a method to do so
  * Fit statistical model controlling for confounder
  * Use fitted model to predict counterfactuals
  * Compare counterfactual with actual for ATE
* Can combine G-computation and IPW for doubly-robust estimation
* How to derive expectation expressions from symbolic probability expression?
  * Have to convert to some expectation

$$
p_{\mathrm{do}}(y\mid z) = \int p(y\mid x,z)p(x\mid z)\,dx \\
\mathbb E_{\mathrm{do}(z)}[h(Y)]
=
\int h(y)\,p_{\mathrm{do}}(y\mid z)\,dy \\
\begin{aligned}
\mathbb E_{\mathrm{do}(z)}[h(Y)]
&=
\int h(y)
\left[
\int p(y\mid x,z)p(x\mid z)\,dx
\right]dy\\
&=
\int\int
h(y)p(y\mid x,z)p(x\mid z)
\,dy\,dx\\
&=
\mathbb E_{X\sim p(x\mid z)}
\left[
\mathbb E_{Y\sim p(y\mid X,z)}
[h(Y)]
\right].
\end{aligned}
$$