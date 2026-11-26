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

# `automatic_sensealg_choice` selects a sensitivity algorithm/VJP by probing
# candidate backends (ReverseDiff, Tracker, ...) inside try/catch. Its result is
# a discrete algorithm choice that never depends differentiably on the numeric
# inputs, so it must be inactive. This matters for forward-over-reverse (nested
# Enzyme), where the outer forward pass would otherwise type-analyze the
# Union-typed probing code and fail with an IllegalTypeAnalysisException.
EnzymeRules.inactive(::typeof(automatic_sensealg_choice), args...) = nothing
EnzymeRules.inactive(::typeof(_enzyme_originator_vjp), args...) = nothing
