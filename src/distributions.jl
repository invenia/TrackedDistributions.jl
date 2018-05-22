import Flux.Tracker:
    TrackedArray,
    TrackedReal

const TrArray = Union{Array{<:Real}, TrackedArray}
const TrReal = Union{<:Real, TrackedReal}

abstract type AbstractTMVDiagonalNormal <: ContinuousMultivariateDistribution end
abstract type AbstractTDiagonalNormal <: ContinuousUnivariateDistribution end

# Distributions.rand(rng::AbstractRNG, d::AbstractTMVDiagonalNormal) = Distributions._rand!(rng, d, Vector{eltype(d)})(length(d))
# Distributions.rand(rng::AbstractRNG, d::AbstractTMVDiagonalNormal, n::Int) = Distributions._rand!(rng, d, Matrix{eltype(d)}(length(d), n))
# Distributions.rand!(rng::AbstractRNG, d::AbstractTMVDiagonalNormal, x::VecOrMat) = Distributions._rand!(rng, d, x)



struct TMVDiagonalNormal{T<:TrArray} <: AbstractTMVDiagonalNormal
    μ::T
    logσ::T
    function TMVDiagonalNormal{T}(μ::T, logσ::T) where T<:TrArray
        @assert size(μ) == size(logσ)
        @assert ndims(μ) == 1
        new(μ, logσ)
    end
end

TMVDiagonalNormal(μ::T, logσ::T) where {T<:Real} = TMVDiagonalNormal([μ], [logσ]) # Convert to multivariate case
TMVDiagonalNormal(μ::T, logσ::T) where {T<:TrArray} = TMVDiagonalNormal{T}(μ, logσ)

Base.convert(::Type{<:MvNormal}, d::TMVDiagonalNormal{<:Array{<:Real}}) = MvNormal(d.μ, exp.(d.logσ))
#Base.convert(::Type{<:MvNormal}, d::TMVDiagonalNormal{<:TrackedArray}) = MvNormal(d.μ.data, exp.(d.logσ.data))

# TTMVDiagonalNormal: https://juliastats.github.io/Distributions.jl/latest/extends.html
Base.length(d::TMVDiagonalNormal) = length(d.μ)
sampler(d::TMVDiagonalNormal) = d
#Distributions._rand!(d::TMVDiagonalNormal, x::AbstractArray) = Distributions._rand!(convert(MvNormal, d), x)
Distributions.rand(rng, d::TMVDiagonalNormal{<:Array{<:Real}}) = Distributions.rand(rng, convert(MvNormal, d))
Distributions.rand(rng, d::TMVDiagonalNormal{<:Array{<:Real}}, n::Int) = Distributions.rand(rng, convert(MvNormal, d), n)
Distributions._logpdf(d::TMVDiagonalNormal, x::AbstractArray) = Distributions._logpdf(convert(MvNormal, d), x)
Base.mean(d::TMVDiagonalNormal) = d.μ
Base.var(d::TMVDiagonalNormal) = exp.(2 * d.logσ)
Distributions.entropy(d::TMVDiagonalNormal, args...) = Distributions.entropy(convert(MvNormal, d), args...)
Base.cov(d::TMVDiagonalNormal) = diagm(Distributions.var(d))

Distributions._rand!(d::TMVDiagonalNormal, x::VecOrMat) = Distributions._rand!(Base.GLOBAL_RNG, convert(MvNormal, d), x)
Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal, x::VecOrMat) = Distributions._rand!(rng, convert(MvNormal, d), x)

Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal, x::AbstractMatrix) = Distributions._rand!(rng, convert(MvNormal, d), x)
Distributions._rand!(d::TMVDiagonalNormal, x::AbstractMatrix) = Distributions._rand!(Base.GLOBAL_RNG, convert(MvNormal, d), x)D
Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal, x::AbstractVector) = Distributions._rand!(rng, convert(MvNormal, d), x)
Distributions._rand!(d::TMVDiagonalNormal, x::AbstractVector) = Distributions._rand!(Base.GLOBAL_RNG, convert(MvNormal, d), x)

# For sampling with TrackedArrays, we want to use the reparmaterisation trick:
Distributions.rand(rng, d::TMVDiagonalNormal{TrackedArray}) = sample(rng, d)
Distributions.rand(rng, d::TMVDiagonalNormal{TrackedArray}, n::Int) = sample(rng, d, n)

Distributions._rand!(d::TMVDiagonalNormal{<:TrackedArray}, x::VecOrMat) = error("Not Implemented Error")
Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal{<:TrackedArray}, x::VecOrMat) = error("Not Implemented Error")

Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal{<:TrackedArray}, x::AbstractMatrix) = error("Not Implemented Error")
Distributions._rand!(d::TMVDiagonalNormal{<:TrackedArray}, x::AbstractMatrix) = error("Not Implemented Error")
Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal{<:TrackedArray}, x::AbstractVector) = error("Not Implemented Error")
Distributions._rand!(d::TMVDiagonalNormal{<:TrackedArray}, x::AbstractVector) = error("Not Implemented Error")


"""
    kl_q_p(q::DiagonalNormal, p::DiagonalNormal)

    Compute KL divergence between multidimenional diagonal Gaussian distributions
    with the given means and log-sigmas: KL(N(μ_q, logσ_q) || N(μ_p, logσ_p)).
    Reference: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function
"""
function kl_q_p(q::TMVDiagonalNormal, p::TMVDiagonalNormal)
    μ_q, logσ_q = q.μ, q.logσ
    μ_p, logσ_p = p.μ, p.logσ
    @assert(size(μ_q) == size(μ_p))
    @assert(ndims(μ_q) == 1)
    D = length(μ_q)
    return - 0.5 * D + 0.5 .* sum( 2 .* (logσ_p .- logσ_q) .+ exp.(2 .* (logσ_q .- logσ_p)) .+ ( (μ_q .- μ_p).^2 ./ exp.(2 .* logσ_p) ) )
end
#
#
"""
    rand(rng, d::DiagonalNormal)

    Used for the Reparametrization trick.  Note we pass in rng.
    TODO: Convert to sampler and rand
"""
function sample(rng::AbstractRNG, d::TMVDiagonalNormal)
    μ = d.μ
    logσ = d.logσ
    return μ .+ exp.(logσ) .* randn(rng, Float64, size(μ))
end

Distributions.rand(rng, d::TMVDiagonalNormal{TrackedArray{T, N, Array{T, N}}}) where {T<:Real, N} = sample(rng, d)
Distributions.rand(d::TMVDiagonalNormal{TrackedArray{T, N, Array{T, N}}}) where {T<:Real, N} = error("Seeding Enforced. use rand(rng, d)")
Distributions.rand(rng::AbstractRNG, d::TMVDiagonalNormal{TrackedArray{T, N, Array{T, N}}}, n::Int) where {T<:Real, N} = error("Not Implemented Error")

Distributions._logpdf(d::TMVDiagonalNormal{TrackedArray{T, N, Array{T, N}}}, x::AbstractArray) where {T<:Real, N} = log_pdf(d, x)

function log_pdf(d::TMVDiagonalNormal{TrackedArray{T, 1, Array{T, 1}}}, x::AbstractArray) where {T<:Real}
    μ = d.μ
    logσ = d.logσ
    n = 1
    D = length(μ)
    return -.5 * ( n * (D * log(2π) + 2 * sum(logσ)) + sum((x .- μ).^2 .* exp.(-2 .* logσ)) )
end
