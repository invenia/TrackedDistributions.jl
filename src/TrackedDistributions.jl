__precompile__()
module TrackedDistributions

using Reexport

export
    TDiagonalNormal,
    TMVDiagonalNormal,
    kl_q_p,
    data

@reexport using Distributions

include("./distributions.jl")



end
