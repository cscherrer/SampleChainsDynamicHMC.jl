

using SampleChainsDynamicHMC
using TransformVariables

function ℓ(nt)
	z = nt.x/nt.σ
	return -z^2 - nt.σ - log(nt.σ)
end

t = as((x=asℝ, σ=asℝ₊))

chain = initialize!(dynamicHMC(), ℓ, t)

drawsamples!(chain, 9)
drawsamples!(chain, 90)

chains = initialize!(4, dynamicHMC(), ℓ, t)

drawsamples!(chains, 9)
drawsamples!(chains, 90)

samples(chains)
