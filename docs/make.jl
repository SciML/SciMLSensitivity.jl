using Documenter, SciMLSensitivity
using DocumenterVitepress

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
using Plots

include("pages.jl")

deploy_config = Documenter.auto_detect_deploy_system()
deploy_decision = Documenter.deploy_folder(deploy_config; repo="github.com/SciML/SciMLSensitivity.jl",
    devbranch="master", devurl="dev", push_preview=true)

makedocs(sitename = "SciMLSensitivity.jl",
    authors = "Chris Rackauckas et al.",
    modules = [SciMLSensitivity],
    clean = true, doctest = false, linkcheck = true,
    warnonly = true,
    format = DocumenterVitepress.MarkdownVitepress(;
            repo="https://github.com/SciML/SciMLSensitivity.jl",
            devbranch="master", devurl="dev",
            deploy_url="https://docs.sciml.ai/SciMLSensitivity/", deploy_decision
        ),
    pages = pages)

deploydocs(repo = "github.com/SciML/SciMLSensitivity.jl.git";
    target="build", # this is where Vitepress stores its output
    branch = "gh-pages",
    devbranch="master",
    push_preview = true)
