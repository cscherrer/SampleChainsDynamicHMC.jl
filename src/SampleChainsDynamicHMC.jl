module SampleChainsDynamicHMC

using Reexport
@reexport using SampleChains
using LogDensityProblems
using DynamicHMC
using NestedTuples
using ElasticArrays
using StructArrays
using ConcreteStructs
using TransformVariables
using MappedArrays
using Random

using TupleVectors: chainvec

export DynamicHMCChain

@concrete struct  DynamicHMCChain{T} <: AbstractChain{T}
    samples     # :: AbstractVector{T}
    logq        # log-density for distribution the sample was drawn from
    info        # Per-sample metadata, type depends on sampler used
    meta        # Metadata associated with the sample as a whole
    state       
    transform
end

function DynamicHMCChain(t::TransformVariables.TransformTuple, Q::DynamicHMC.EvaluatedLogDensity, tree_stats, steps)
    tQq = TransformVariables.transform(t, Q.q)
    T = typeof(tQq)
    samples = chainvec(tQq)
    logq = chainvec(Q.ℓq)
    info = chainvec(tree_stats)
    meta = steps
    transform = t

    return DynamicHMCChain{T}(samples, logq, info, meta, Q, transform)
end

SampleChains.summarize(ch::DynamicHMCChain) = summarize(samples(ch))

function gettransform(chain::DynamicHMCChain)
    return getfield(chain, :transform)
end

function SampleChains.pushsample!(chain::DynamicHMCChain, Q::DynamicHMC.EvaluatedLogDensity, tree_stats)
    push!(samples(chain), transform(gettransform(chain), Q.q))
    push!(logq(chain), Q.ℓq)
    push!(info(chain), tree_stats)
end

function SampleChains.step!(chain::DynamicHMCChain)
    Q, tree_stats = DynamicHMC.mcmc_next_step(getfield(chain, :meta), getfield(chain, :state))
end

function SampleChains.initialize!(rng::Random.AbstractRNG, ::Type{DynamicHMCChain}, ℓ, tr, ad_backend=Val(:ForwardDiff))
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

function SampleChains.initialize!(::Type{DynamicHMCChain}, ℓ, tr, ad_backend=Val(:ForwardDiff))
    rng = Random.GLOBAL_RNG
    return initialize!(rng, DynamicHMCChain, ℓ, tr, ad_backend)
end

function SampleChains.drawsamples!(chain::DynamicHMCChain, n::Int=1000)
    @cleanbreak for j in 1:n
        Q, tree_stats = step!(chain)
        pushsample!(chain, Q, tree_stats)
    end 
    return chain
end

end
