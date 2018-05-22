__precompile__()
module TrackedDistributions

using Reexport

export
    TDiagonalNormal,
    TMVDiagonalNormal,
    TrArray,
    kl_q_p

@reexport using Distributions

include("./distributions.jl")



end
