import Flux.Tracker:
    TrackedArray,
    TrackedReal


const TrArray = Union{Array{<:Real}, TrackedArray}
const TrReal = Union{Array{<:Real}, TrackedReal}

struct TDiagonalNormal{T<:TrReal} <: ContinuousUnivariateDistribution
    μ::T
    logσ::T
    function TDiagonalNormal{T}(μ::T, logσ::T) where T<:TrReal
        @assert size(μ) == size(logσ)
        @assert ndims(μ) == 1
        new(μ, logσ)
    end
end

struct TMVDiagonalNormal{T<:TrArray} <: ContinuousMultivariateDistribution
    μ::T
    logσ::T
    function TMVDiagonalNormal{T}(μ::T, logσ::T) where T<:TrArray
        @assert size(μ) == size(logσ)
        @assert ndims(μ) == 1
        new(μ, logσ)
    end
end

TDiagonalNormal(μ::T, logσ::T) where {T<:TrReal} = TMVDiagonalNormal{T}(μ, logσ)
TMVDiagonalNormal(μ::T, logσ::T) where {T<:TrArray} = TMVDiagonalNormal{T}(μ, logσ)

Base.convert(::Type{<:MvNormal}, d::TMVDiagonalNormal{<:Array{<:Real}}) = MvNormal(d.μ, exp.(d.logσ))

# TTMVDiagonalNormal: https://juliastats.github.io/Distributions.jl/latest/extends.html
Base.length(d::TMVDiagonalNormal) = length(d.μ)
sampler(d::TMVDiagonalNormal) = error("Not Implemented Error")
Distributions._rand!(d::TMVDiagonalNormal, x::AbstractArray) = Distributions._rand!(convert(MvNormal, d), x)
Distributions._logpdf(d::TMVDiagonalNormal, x::AbstractArray) = Distributions._logpdf(convert(MvNormal, d), x)
Base.mean(d::TMVDiagonalNormal) = d.μ
Base.var(d::TMVDiagonalNormal) = exp.(2 * d.logσ)
Distributions.entropy(d::TMVDiagonalNormal, args...) = Distributions.entropy(convert(MvNormal, d), args...)
Base.cov(d::TMVDiagonalNormal) = diagm(Distributions.var(d))

# Additional

Distributions._rand!(d::TMVDiagonalNormal, x::AbstractVector) = Distributions._rand!(convert(MvNormal, d), x)
Distributions._rand!(d::TMVDiagonalNormal, x::AbstractMatrix) = Distributions._rand!(convert(MvNormal, d), x)
Distributions.rand(rng::AbstractRNG, d::TMVDiagonalNormal) = Distributions._rand!(rng, d, Vector{eltype(d)})(length(d))
Distributions.rand(rng::AbstractRNG, d::TMVDiagonalNormal, n::Int64) = Distributions._rand!(rng, d, Matrix{eltype(d)}(length(d), n))
Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal, x::AbstractVector) = Distributions._rand!(rng, convert(MvNormal, d), x)
Distributions._rand!(rng::AbstractRNG, d::TMVDiagonalNormal, x::AbstractMatrix) = Distributions._rand!(rng, convert(MvNormal, d), x)
