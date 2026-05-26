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
import Enzyme.EnzymeCore: MixedDuplicated

# VJP choice types should be inactive since they configure computation methods
EnzymeRules.inactive_type(::Type{<:VJPChoice}) = true

# Workaround for EnzymeAD/Enzyme.jl#3126: Enzyme's `runtime_generic_augfwd`
# (via `create_activity_wrapper` in `jitrules.jl`) constructs
# `MixedDuplicated(primarg, shadowarg)` with `shadowarg::T` rather than
# `Base.RefValue{T}`. `EnzymeCore.MixedDuplicated` only defines
# `MixedDuplicated(::T, ::Base.RefValue{T})`, so the bare-`T` call raises
# `MethodError`. This blocks reverse-mode AD through MTK init paths that
# capture `ODESolution` (and similar mutable SciML types).
#
# Until #3126 is fixed upstream, intercept the unwrapped form here and wrap
# the shadow in `Base.RefValue` so the existing constructor accepts it.
# This is type piracy on `EnzymeCore`; remove this shim once #3126 lands.
@inline MixedDuplicated(x::T, dx::T, check::Bool = true) where {T} =
    MixedDuplicated(x, Base.RefValue(dx), check)
