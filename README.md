# TrackedDistributions

[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://doc.invenia.ca/invenia/TrackedDistributions.jl/master)
[![Build Status](https://gitlab.invenia.ca/invenia/TrackedDistributions.jl/badges/master/build.svg)](https://gitlab.invenia.ca/invenia/TrackedDistributions.jl/commits/master)
[![Coverage](https://gitlab.invenia.ca/invenia/TrackedDistributions.jl/badges/master/coverage.svg)](https://gitlab.invenia.ca/invenia/TrackedDistributions.jl/commits/master)



This package is unfortunately needed to combine Distributions.jl with Flux.
Althoguh we can do this:

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
```
