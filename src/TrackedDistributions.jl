__precompile__()
module TrackedDistributions

using Reexport

export
    TDiagonalNormal,
    TMVDiagonalNormal,
    kl_q_p,
    data,
    logσ

using Compat: dropdims
using Compat.LinearAlgebra
import Compat.LinearAlgebra: diagm
using Compat.Random
@reexport using Distributions

include("./distributions.jl")



end
