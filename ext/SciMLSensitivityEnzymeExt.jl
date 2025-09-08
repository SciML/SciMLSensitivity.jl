module SciMLSensitivityEnzymeExt

using SciMLSensitivity
import EnzymeRules

# Enzyme rules for VJP choice types defined in SciMLSensitivity
#
# VJP choice types configure how jacobian-vector products are computed within 
# sensitivity algorithms. They should be treated as inactive (constant) during 
# Enzyme differentiation.
#
# Note: AbstractSensitivityAlgorithm inactive rule is handled in SciMLBase
# to avoid type piracy.

# VJP choice types should be inactive since they configure how jacobian-vector 
# products are computed within sensitivity algorithms
EnzymeRules.inactive_type(::Type{<:SciMLSensitivity.VJPChoice}) = true

end