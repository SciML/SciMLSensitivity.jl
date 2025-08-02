using ADTypes: AutoFiniteDiff

# Simulate the problematic constructor logic
function test_constructor(; autodiff = true, autojacvec = autodiff, autojacmat = false)
    println("autodiff = ", autodiff, " (type: ", typeof(autodiff), ")")
    println("autojacvec = ", autojacvec, " (type: ", typeof(autojacvec), ")")
    println("autojacmat = ", autojacmat, " (type: ", typeof(autojacmat), ")")
    
    # This should fail when autojacvec is not a boolean
    try
        result = autojacvec && autojacmat
        println("Boolean operation result: ", result)
    catch e
        println("ERROR: ", e)
    end
end

println("Testing with boolean autodiff (should work):")
test_constructor(autodiff = true)

println("\nTesting with AutoFiniteDiff autodiff (should fail):")
test_constructor(autodiff = AutoFiniteDiff())