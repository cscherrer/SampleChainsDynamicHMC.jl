

using SampleChainsDynamicHMC
using TransformVariables
using Logging
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

using LinearAlgebra
using ReverseDiff
using SampleChainsDynamicHMC.LogDensityProblems

@testset "config options" begin
	config = dynamichmc(
		warmup_stages=default_warmup_stages(
			M=Symmetric, # adapt dense positive definite metric
			stepsize_adaptation=DualAveraging(δ=0.9), # target acceptance rate 0.9
			doubling_stages=7, # 7-stage metric adaptation
		),
		reporter=LogProgressReport(), # log progress using Logging
		ad_backend=Val(:ReverseDiff), # use ReverseDiff AD package
	)

	chain = newchain(config, ℓ, t)
	@test length(chain) == 1

	meta = getfield(chain, :meta)
	@test meta.H.κ.M⁻¹ isa Symmetric
	@test meta.H.ℓ isa LogDensityProblems.ReverseDiffLogDensity

	sample!(chain, 9)
	@test length(chain) == 10
	sample!(chain, 90)
	@test length(chain) == 100
end

@testset "reporting" begin
	io = IOBuffer()
	chains = with_logger(SimpleLogger(io)) do
		newchain(dynamichmc(reporter=LogProgressReport()), ℓ, t)
	end
	warmup_log = String(take!(io))
	@test !isempty(warmup_log)

	io = IOBuffer()
	with_logger(SimpleLogger(io)) do
		sample!(chains, 10)
	end
	log = String(take!(io))
	@test !isempty(log)
end
