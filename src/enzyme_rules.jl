# Enzyme rules for VJP choice types defined in SciMLSensitivity
#
# VJP choice types configure how jacobian-vector products are computed within 
# sensitivity algorithms. They should be treated as inactive (constant) during 
# Enzyme differentiation to prevent errors when they are stored in problem 
# structures or other data that Enzyme differentiates through.
#
# Note: AbstractSensitivityAlgorithm inactive rule is handled in SciMLBase
# to avoid type piracy.

import Enzyme: EnzymeRules

# VJP choice types should be inactive since they configure computation methods
EnzymeRules.inactive_type(::Type{<:VJPChoice}) = true
