import Tracker: TrackedArray, TrackedReal

# const TrArray = Union{Array{<:Real}, TrackedArray}
# const TrArray1 = Union{Array{<:Real, 1}, TrackedArray{<:Real, 1, <:AbstractArray{<:Real, 1}}}
# const TrArray2 = Union{Array{<:Real, 2}, TrackedArray{<:Real, 1, <:AbstractArray{<:Real, 2}}}

#const TrArray{T, N} = Union{Array{T, N} where {T<:Real} where {N<:Int}, TrackedArray{T, N, AbstractArray{T, N}} where {T<:Real} where {N<:Int}}
#TrArray = TrArray{T, N} where {T, N}
const TrReal = Union{<:Real, TrackedReal}

abstract type AbstractTDiagonalNormal <: ContinuousUnivariateDistribution end
abstract type AbstractTMVDiagonalNormal <: ContinuousMultivariateDistribution end

# Distributions.rand(rng::AbstractRNG, d::AbstractTMVDiagonalNormal) = Distributions._rand!(rng, d, Vector{eltype(d)})(length(d))
# Distributions.rand(rng::AbstractRNG, d::AbstractTMVDiagonalNormal, n::Int) = Distributions._rand!(rng, d, Matrix{eltype(d)}(length(d), n))
# Distributions.rand!(rng::AbstractRNG, d::AbstractTMVDiagonalNormal, x::VecOrMat) = Distributions._rand!(rng, d, x)

struct TMVDiagonalNormal{T<:AbstractArray} <: AbstractTMVDiagonalNormal
    μ::T
    logσ::T
    function TMVDiagonalNormal{T}(μ::T, logσ::T) where T<:AbstractArray{S, 1} where {S}
        @assert size(μ) == size(logσ)
        new(μ, logσ)
    end
    function TMVDiagonalNormal{T}(μ::T, logσ::T) where T<:AbstractArray{S, 2} where {S}
        @assert size(μ,2) == 1
        @assert size(logσ,2) == 1
        TMVDiagonalNormal(dropdims(μ, dims=2), dropdims(logσ, dims=2))
    end
end

TMVDiagonalNormal(μ::T, logσ::T) where {T<:Real} = TMVDiagonalNormal{Array{T, 1}}([μ], [logσ]) # Convert to multivariate case
TMVDiagonalNormal(μ::T, logσ::T) where {T<:AbstractArray} = TMVDiagonalNormal{T}(μ, logσ)

# https://github.com/JuliaLang/julia/pull/26601
Base.broadcastable(t::AbstractTMVDiagonalNormal) = Ref(t)

Base.convert(::Type{<:MvNormal}, d::TMVDiagonalNormal{<:Array{<:Real}}) = MvNormal(d.μ, exp.(d.logσ))
#Base.convert(::Type{<:MvNormal}, d::TMVDiagonalNormal{<:TrackedArray}) = MvNormal(d.μ.data, exp.(d.logσ.data))

# TTMVDiagonalNormal: https://juliastats.github.io/Distributions.jl/latest/extends.html
Base.length(d::TMVDiagonalNormal) = length(d.μ)
sampler(d::TMVDiagonalNormal) = d
#Distributions._rand!(d::TMVDiagonalNormal, x::AbstractArray) = Distributions._rand!(convert(MvNormal, d), x)
Distributions.rand(rng::AbstractRNG, d::TMVDiagonalNormal{<:Array{<:Real}}) = Distributions.rand(rng, convert(MvNormal, d))
Distributions.rand(rng::AbstractRNG, d::TMVDiagonalNormal{<:Array{<:Real}}, n::Int) = Distributions.rand(rng, convert(MvNormal, d), n)
Distributions._logpdf(d::TMVDiagonalNormal, x::AbstractArray) = Distributions._logpdf(convert(MvNormal, d), x)
Distributions.mean(d::TMVDiagonalNormal) = d.μ
Distributions.var(d::TMVDiagonalNormal) = exp.(2 * d.logσ)
Distributions.entropy(d::TMVDiagonalNormal, args...) = Distributions.entropy(convert(MvNormal, d), args...)
Distributions.cov(d::TMVDiagonalNormal) = diagm(0 => Distributions.var(d))

Distributions._rand!(d::TMVDiagonalNormal, x::VecOrMat) = Distributions._rand!(Random.GLOBAL_RNG, convert(MvNormal, d), x)
Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal, x::VecOrMat) = Distributions._rand!(rng, convert(MvNormal, d), x)

Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal, x::AbstractMatrix) = Distributions._rand!(rng, convert(MvNormal, d), x)
Distributions._rand!(d::TMVDiagonalNormal, x::AbstractMatrix) = Distributions._rand!(Random.GLOBAL_RNG, convert(MvNormal, d), x)D
Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal, x::AbstractVector) = Distributions._rand!(rng, convert(MvNormal, d), x)
Distributions._rand!(d::TMVDiagonalNormal, x::AbstractVector) = Distributions._rand!(Random.GLOBAL_RNG, convert(MvNormal, d), x)

# For sampling with TrackedArrays, we want to use the reparmaterisation trick:
Distributions.rand(rng::AbstractRNG, d::TMVDiagonalNormal{<:TrackedArray}) = sample(rng, d)
Distributions.rand(rng::AbstractRNG, d::TMVDiagonalNormal{<:TrackedArray}, n::Int) = sample(rng, d, n)

Distributions._rand!(d::TMVDiagonalNormal{<:TrackedArray}, x::VecOrMat) = error("Not Implemented Error")
Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal{<:TrackedArray}, x::VecOrMat) = error("Not Implemented Error")

Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal{<:TrackedArray}, x::AbstractMatrix) = error("Not Implemented Error")
Distributions._rand!(d::TMVDiagonalNormal{<:TrackedArray}, x::AbstractMatrix) = error("Not Implemented Error")
Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal{<:TrackedArray}, x::AbstractVector) = error("Not Implemented Error")
Distributions._rand!(d::TMVDiagonalNormal{<:TrackedArray}, x::AbstractVector) = error("Not Implemented Error")

function data(
    d::TrackedDistributions.TMVDiagonalNormal{T}
) where {N, S<:Real, T<:TrackedArray{S, N, Array{S,N}}}
    TrackedDistributions.TMVDiagonalNormal(
        Array{Float64}(mean(d).data),
        Array{Float64}(log.(sqrt.(var(d)).data)),
    )
end

# Getters
logσ(d::TMVDiagonalNormal) = d.logσ

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
    sample(rng, d::DiagonalNormal)

    Used for the Reparametrization trick.  Note we pass in rng.
    TODO: Convert to sampler and rand
"""
function sample(rng::AbstractRNG, d::TMVDiagonalNormal)
    μ = d.μ
    logσ = d.logσ
    return μ .+ exp.(logσ) .* randn(rng, Float64, size(μ))
end

Distributions.rand(rng::AbstractRNG, d::TMVDiagonalNormal{TrackedArray{T, N, Array{T, N}}}) where {T<:Real, N} = sample(rng, d)
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
