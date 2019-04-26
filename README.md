# TrackedDistributions

[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://research.pages.invenia.ca/TrackedDistributions.jl/)
[![Build Status](https://gitlab.invenia.ca/research/TrackedDistributions.jl/badges/master/build.svg)](https://gitlab.invenia.ca/research/TrackedDistributions.jl/commits/master)
[![Coverage](https://gitlab.invenia.ca/research/TrackedDistributions.jl/badges/master/coverage.svg)](https://gitlab.invenia.ca/research/TrackedDistributions.jl/commits/master)



This package is unfortunately needed to combine Distributions.jl with Flux.
Although we can do this:

```
julia> using Distributions
julia> using Flux
julia> Normal(0, 1)
julia> Normal(Flux.Tracker.TrackedReal(0), Flux.Tracker.TrackedReal(1))
Distributions.Normal{Flux.Tracker.TrackedReal{Int64}}(μ=0 (tracked), σ=1 (tracked))
```

We can't do this to the MvNormal equivalent:
```
julia> MvNormal(zeros(2)), ones(2)
(ZeroMeanDiagNormal(
dim: 2
μ: [0.0, 0.0]
Σ: [0.0 0.0; 0.0 0.0]
)
, [1.0, 1.0])
julia> MvNormal(Flux.Tracker.TrackedArray(zeros(2)), Flux.Tracker.TrackedArray(ones(2)))
ERROR: MethodError: no method matching Distributions.MvNormal(::TrackedArray{…,Array{Float64,1}}, ::TrackedArray{…,Array{Float64,1}})

TrackedDistributions extends Distributions such that it can work with TrackedArrays, such as given in the examples

```
# A note of expectations
This is a pretty barebones package, and is pretty limited in terms coverage of the distributions in Distributions. By "limited", there is one distribution covered at present...
However, it can be extended to include other relevant Distributions of interest, while a more robust solution of integrating Flux/Zygote and Distributions can be found. 

# Examples



```
d = TMVDiagonalNormal(Flux.Tracker.TrackedArray(zeros(2)), Flux.Tracker.TrackedArray(ones(2)))
```

We can get the logpdf as normal, except now returning a Tracked Array:

```
logpdf(d, [1., 1.])
-3.973212349645958 (tracked)
```

It can also be sampled:

```
using Random
rng = Random.seed!(1)
Tracked 2-element Array{Float64,1}:
 0.808112526181959
 1.0394600105212195
```

And get the KL divergence:
```
d1 = TMVDiagonalNormal(Flux.Tracker.TrackedArray(zeros(2)), Flux.Tracker.TrackedArray(ones(2)))
d2 = TMVDiagonalNormal(Flux.Tracker.TrackedArray(ones(2)), Flux.Tracker.TrackedArray(ones(2)))
kl_q_p(d1, d2)
0.1353352832366128 (tracked)
kl_q_p(d1, d1)
0.0 (tracked)
```

Most of the machinery of distributions should carry over:

```
mean(d1)
Tracked 2-element Array{Float64,1}:
 0.0
 0.0

cov(d1)
Tracked 2×2 Array{Float64,2}:
7.38906  0.0    
0.0      7.38906

```
Note here the following (because typically we want to work with log σ)
```
exp.(1.0).^2 == 7.3890560989306495
true
```






```
