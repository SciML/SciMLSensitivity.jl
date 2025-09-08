using Pkg
Pkg.activate(".")

using SciMLSensitivity
using Enzyme

println("Testing if Enzyme rules are working...")

# Create a BacksolveAdjoint object
sensealg = BacksolveAdjoint(autojacvec=EnzymeVJP())

# Test if Enzyme can handle it
function test_func(x)
    # Store the sensealg in a data structure
    data = (value=x[1] + x[2], alg=sensealg)
    return data.value * 2.0
end

x = [1.0, 2.0]
dx = Enzyme.make_zero(x)

try
    result = Enzyme.autodiff(Enzyme.Reverse, test_func, Enzyme.Active, Enzyme.Duplicated(x, dx))
    println("SUCCESS: Enzyme rules work! dx = ", dx)
    println("The fix successfully treats sensealg as constant!")
catch e
    println("FAILED: Still getting error: ", e)
end