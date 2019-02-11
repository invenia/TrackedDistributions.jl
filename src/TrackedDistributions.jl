__precompile__()
module TrackedDistributions

using Reexport

export
    TDiagonalNormal,
    TMVDiagonalNormal,
    kl_q_p,
    data,
    logσ

using LinearAlgebra
import LinearAlgebra: diagm
using Random
@reexport using Distributions

include("./distributions.jl")



end
