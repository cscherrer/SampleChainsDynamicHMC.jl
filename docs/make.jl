using SampleChainsDynamicHMC
using Documenter

DocMeta.setdocmeta!(SampleChainsDynamicHMC, :DocTestSetup, :(using SampleChainsDynamicHMC); recursive=true)

makedocs(;
    modules=[SampleChainsDynamicHMC],
    authors="Chad Scherrer <chad.scherrer@gmail.com> and contributors",
    repo="https://github.com/cscherrer/SampleChainsDynamicHMC.jl/blob/{commit}{path}#{line}",
    sitename="SampleChainsDynamicHMC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cscherrer.github.io/SampleChainsDynamicHMC.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cscherrer/SampleChainsDynamicHMC.jl",
)
