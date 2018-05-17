__precompile__()
module TrackedDistributions

using Reexport

export
    TDiagonalNormal,
    TMVDiagonalNormal

@reexport using Distributions

include("./distributions.jl")



end
