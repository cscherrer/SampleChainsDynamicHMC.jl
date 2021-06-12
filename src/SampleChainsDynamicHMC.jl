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
using TupleVectors

using TupleVectors:chainvec

export dynamicHMC

@concrete struct DynamicHMCChain{T} <: AbstractChain{T}
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

TupleVectors.summarize(ch::DynamicHMCChain) = summarize(samples(ch))

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

@concrete struct DynamicHMCConfig <: ChainConfig{DynamicHMCChain}
    init
    warmup_stages
    algorithm
    reporter
end

# Docs adapted from https://tamaspapp.eu/DynamicHMC.jl/stable/interface/
"""
    dynamicHMC(
        init          = ()
      , warmup_stages = DynamicHMC.default_warmup_stages()
      , algorithm     = DynamicHMC.NUTS()
      , reporter      = DynamicHMC.NoProgressReport()
    )

`init`: a `NamedTuple` that can contain the following fields (all of them optional and provided with reasonable defaults):
- `q`: initial position. Default: random (uniform `[-2,2]` for each coordinate).
- `κ`: kinetic energy specification. Default: Gaussian with identity matrix.
- `ϵ`: a scalar for initial step size, or `nothing` for heuristic finders.

`warmup_stages`: a sequence of warmup stages. See `DynamicHMC.default_warmup_stages` and `DynamicHMC.fixed_stepsize_warmup_stages`; the latter requires an `ϵ` in initialization.

`algorithm`: see `NUTS`. It is very unlikely you need to modify this, except perhaps for the maximum depth.

`reporter`: how progress is reported. By default, verbosely for interactive sessions using the log message mechanism (see `DynamicHMC.LogProgressReport`, and no reporting for non-interactive sessions (see `DynamicHMC.NoProgressReport`).

For more details see https://tamaspapp.eu/DynamicHMC.jl/stable/interface/
"""   
function dynamicHMC(;
          init          = ()
        , warmup_stages = DynamicHMC.default_warmup_stages()
        , algorithm     = DynamicHMC.NUTS()
        , reporter      = DynamicHMC.NoProgressReport()
    )
    DynamicHMCConfig(init, warmup_stages, algorithm, reporter)
end

function SampleChains.newchain(rng::Random.AbstractRNG, config::DynamicHMCConfig, ℓ, tr, ad_backend=Val(:ForwardDiff))
    P = LogDensityProblems.TransformedLogDensity(tr, ℓ)
    ∇P = LogDensityProblems.ADgradient(ad_backend, P)
    reporter = DynamicHMC.NoProgressReport()

    results = DynamicHMC.mcmc_keep_warmup(rng, ∇P, 0; 
          initialization = config.init          
        , warmup_stages  = config.warmup_stages 
        , algorithm      = config.algorithm     
        , reporter       = config.reporter     
        )

    steps = DynamicHMC.mcmc_steps(
        results.sampling_logdensity,
        results.final_warmup_state,
    )
    
    Q = results.final_warmup_state.Q
    Q, tree_stats = DynamicHMC.mcmc_next_step(steps, Q)
    chain = DynamicHMCChain(tr, Q, tree_stats, steps)
end

function SampleChains.newchain(config::DynamicHMCConfig, ℓ, tr, ad_backend=Val(:ForwardDiff))
    rng = Random.GLOBAL_RNG
    return newchain(rng, config, ℓ, tr, ad_backend)
end

function SampleChains.sample!(chain::DynamicHMCChain, n::Int=1000)
    @cleanbreak for j in 1:n
        Q, tree_stats = step!(chain)
        pushsample!(chain, Q, tree_stats)
    end 
    return chain
end

end
