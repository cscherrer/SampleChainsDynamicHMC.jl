module SampleChainsDynamicHMC

using SampleChains
using LogDensityProblems
using DynamicHMC
using NestedTuples
using ElasticArrays
using StructArrays
using ConcreteStructs
using TransformVariables
using MappedArrays
using Random

export DynamicHMCChain

@concrete struct  DynamicHMCChain{T} <: AbstractChain{T}
    samples     # :: AbstractVector{T}
    logp        # log-density for distribution the sample was drawn from
    info        # Per-sample metadata, type depends on sampler used
    meta        # Metadata associated with the sample as a whole
    state       
    transform
end

samples(chain::DynamicHMCChain) = getfield(chain, :samples)
logp(chain::DynamicHMCChain) = getfield(chain, :logp)

SampleChains.summarize(ch::DynamicHMCChain) = summarize(samples(ch))

function gettransform(chain::DynamicHMCChain)
    return getfield(chain, :transform)
end

info(chain::DynamicHMCChain) = getfield(chain, :info)

export pushsample!
function pushsample!(chain::DynamicHMCChain, Q::DynamicHMC.EvaluatedLogDensity, tree_stats)
    push!(samples(chain), gettransform(chain)(Q.q))
    push!(logp(chain), Q.ℓq)
    push!(info(chain), tree_stats)
end

export step!
function step!(chain::DynamicHMCChain)
    Q, tree_stats = DynamicHMC.mcmc_next_step(getfield(chain, :meta), getfield(chain, :state))
end

function DynamicHMCChain(t::TransformVariables.TransformTuple, Q::DynamicHMC.EvaluatedLogDensity, tree_stats, steps)
    tQq = t(Q.q)
    T = typeof(tQq)
    samples = TupleVector([tQq])
    logp = ElasticVector([Q.ℓq])
    info = ElasticVector(StructArray([tree_stats]))
    meta = steps
    transform = t

    return DynamicHMCChain{T}(samples, logp, info, meta, Q, transform)
end

export initialize!
function initialize!(rng::Random.AbstractRNG, ::Type{DynamicHMCChain}, ℓ, tr, ad_backend)
    P = LogDensityProblems.TransformedLogDensity(tr, ℓ)
    ∇P = LogDensityProblems.ADgradient(ad_backend, P)
    reporter = DynamicHMC.NoProgressReport()

    results = DynamicHMC.mcmc_keep_warmup(
        rng,
        ∇P,
        0;
        reporter = reporter
    )

    steps = DynamicHMC.mcmc_steps(
        results.sampling_logdensity,
        results.final_warmup_state,
    )

    Q = results.final_warmup_state.Q
    Q, tree_stats = DynamicHMC.mcmc_next_step(steps, Q)
    chain = DynamicHMCChain(tr, Q, tree_stats, steps)
end

export initialize!

function initialize!(::Type{DynamicHMCChain}, ℓ, tr, ad_backend=Val(:ForwardDiff))
    rng = Random.GLOBAL_RNG
    return initialize!(rng, DynamicHMCChain, ℓ, tr, ad_backend)
end

export drawsamples!
function drawsamples!(chain::DynamicHMCChain, n=1000)
    @cleanbreak for j in 1:n
        Q, tree_stats = step!(chain)
        pushsample!(chain, Q, tree_stats)
    end 
    return chain
end

end
