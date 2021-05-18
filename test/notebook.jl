

using SampleChainsDynamicHMC
using TransformVariables

function ℓ(nt)
	z = nt.x/nt.σ
	return -z^2 - nt.σ - log(nt.σ)
end

t = as((x=asℝ, σ=asℝ₊))

chain = initialize!(dynamicHMC(), ℓ, t)

sample!(chain, 9)
sample!(chain, 90)

chains = initialize!(4, dynamicHMC(), ℓ, t)

sample!(chains, 9)
sample!(chains, 90)

samples(chains)
