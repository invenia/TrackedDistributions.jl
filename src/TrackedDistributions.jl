__precompile__()
module TrackedDistributions

using Reexport

export
    TDiagonalNormal,
    TMVDiagonalNormal,
    TrArray

@reexport using Distributions

include("./distributions.jl")



end
