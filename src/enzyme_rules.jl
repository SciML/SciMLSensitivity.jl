# Enzyme rules for sensitivity algorithms
# 
# Sensitivity algorithms define HOW to compute sensitivities, not WHAT to differentiate.
# They should be treated as inactive (constant) during Enzyme differentiation to prevent
# errors when they are stored in problem structures or other data that Enzyme differentiates through.
#
# This fixes issues like #1225 where passing `sensealg` to ODEProblem constructor would fail
# with Enzyme, while passing it to solve() would work.

import Enzyme: EnzymeRules

# All sensitivity algorithm types should be inactive
EnzymeRules.inactive_type(::Type{<:AbstractSensitivityAlgorithm}) = true

# VJP choice types should also be inactive since they configure how jacobian-vector 
# products are computed within sensitivity algorithms
EnzymeRules.inactive_type(::Type{<:VJPChoice}) = true