using Documenter, DiffEqSensitivity

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
using Plots

include("pages.jl")

makedocs(
    sitename = "DiffEqSensitivity.jl",
    authors="Chris Rackauckas et al.",
    clean = true,
    doctest = false,
    modules = [DiffEqSensitivity],

    strict = [
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
         ],

    format = Documenter.HTML(#analytics = "",
                             assets = ["assets/favicon.ico"],
                             canonical="https://sensitivity.sciml.ai/stable/"),
    pages=pages
)

deploydocs(
   repo = "github.com/SciML/DiffEqSensitivity.jl.git";
   push_preview = true
)
