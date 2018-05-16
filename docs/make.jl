using Documenter, TrackedDistributions

makedocs(;
    modules=[TrackedDistributions],
    format=:html,
    pages=[
        "Home" => "index.md",
    ],
    repo="https://gitlab.invenia.ca/invenia/TrackedDistributions.jl/blob/{commit}{path}#L{line}",
    sitename="TrackedDistributions.jl",
    authors="Alex Robson",
    assets=[
        "assets/invenia.css",
        "assets/logo.png",
    ],
)
