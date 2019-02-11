using Documenter, TrackedDistributions

makedocs(;
    modules=[TrackedDistributions],
    format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
    repo="https://gitlab.invenia.ca/invenia/TrackedDistributions.jl/blob/{commit}{path}#L{line}",
    sitename="TrackedDistributions.jl",
    authors="Alex Robson",
    assets=[
        "assets/invenia.css",
        "assets/logo.png",
    ],
    strict = true,
    checkdocs = :exports,
)
