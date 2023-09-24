using Documenter, SciMLSensitivity

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
using Plots

include("pages.jl")

makedocs(sitename = "SciMLSensitivity.jl",
    authors = "Chris Rackauckas et al.",
    modules = [SciMLSensitivity],
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:missing_docs],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/SciMLSensitivity/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/SciMLSensitivity.jl.git";
    push_preview = true)
