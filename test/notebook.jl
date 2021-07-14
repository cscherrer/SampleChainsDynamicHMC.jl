

using SampleChainsDynamicHMC
using TransformVariables
using Test

function ℓ(nt)
	z = nt.x/nt.σ
	return -z^2 - nt.σ - log(nt.σ)
end

t = as((x=asℝ, σ=asℝ₊))

@testset "single chain" begin
	chain = newchain(dynamichmc(), ℓ, t)
	@test chain isa SampleChainsDynamicHMC.DynamicHMCChain
	@test length(chain) == 1

	sample!(chain, 9)
	@test length(chain) == 10

	sample!(chain, 90)
	@test length(chain) == 100
end

@testset "multichain" begin
	chains = newchain(4, dynamichmc(), ℓ, t)
	@test chains isa MultiChain
	chains_chains = getfield(chains, :chains)
	@test length(chains_chains) == 4
	@test all(x -> length(x) == 1, chains_chains)

	sample!(chains, 9)
	@test all(x -> length(x) == 10, chains_chains)
	sample!(chains, 90)
	@test all(x -> length(x) == 100, chains_chains)

	samples(chains)
end


