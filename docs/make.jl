using Documenter, DiffEqSensitivity

include("pages.jl")

makedocs(
    sitename = "DiffEqSensitivity.jl",
    authors="Chris Rackauckas et al.",
    clean = true,
    doctest = false,
    modules = [DiffEqSensitivity],

    format = Documenter.HTML(#analytics = "",
                             assets = ["assets/favicon.ico"],
                             canonical="https://sensitivity.sciml.ai/stable/"),
    pages=pages
)

deploydocs(
   repo = "github.com/SciML/DiffEqSensitivity.jl.git";
   push_preview = true
)
