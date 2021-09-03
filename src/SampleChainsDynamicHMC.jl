module SampleChainsDynamicHMC

using Reexport
@reexport using SampleChains
@reexport using DynamicHMC
using LogDensityProblems
using NestedTuples
using ElasticArrays
using StructArrays
using ConcreteStructs
using TransformVariables
using MappedArrays
using Random
using TupleVectors

using TupleVectors:chainvec

export dynamichmc

@concrete mutable struct DynamicHMCChain{T} <: AbstractChain{T}
    samples     # :: AbstractVector{T}
    logq        # log-density for distribution the sample was drawn from
    info        # Per-sample metadata, type depends on sampler used
    meta        # Metadata associated with the sample as a whole
    state       
    transform
    reporter
end

function DynamicHMCChain(t::TransformVariables.TransformTuple, Q::DynamicHMC.EvaluatedLogDensity, tree_stats, steps, reporter = NoProgressReport())
    tQq = TransformVariables.transform(t, Q.q)
    T = typeof(tQq)
    samples = chainvec(tQq)
    logq = chainvec(Q.ℓq)
    info = chainvec(tree_stats)
    meta = steps
    transform = t

    return DynamicHMCChain{T}(samples, logq, info, meta, Q, transform, reporter)
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
    setfield!(chain, :state, Q)
    return Q, tree_stats
end

@concrete struct DynamicHMCConfig <: ChainConfig{DynamicHMCChain}
    init
    warmup_stages
    algorithm
    reporter
    ad_backend
end

# Docs adapted from https://tamaspapp.eu/DynamicHMC.jl/stable/interface/
"""
    dynamichmc(
      ; init          = ()
      , warmup_stages = DynamicHMC.default_warmup_stages()
      , algorithm     = DynamicHMC.NUTS()
      , reporter      = DynamicHMC.NoProgressReport()
      , ad_backend    = Val(:ForwardDiff)
    )

`init`: a `NamedTuple` that can contain the following fields (all of them
optional and provided with reasonable defaults): 
- `q`: initial position. Default: random (uniform `[-2,2]` for each coordinate).
- `κ`: kinetic energy specification. Default: Gaussian with identity matrix.
- `ϵ`: a scalar for initial step size, or `nothing` for heuristic finders.

`warmup_stages`: a sequence of warmup stages. See
`DynamicHMC.default_warmup_stages` and
`DynamicHMC.fixed_stepsize_warmup_stages`; the latter requires an `ϵ` in
initialization. 

`algorithm`: see `DynamicHMC.NUTS`. It is very unlikely you need to modify this,
except perhaps for the maximum depth. 

`reporter`: how progress is reported. This is currently silent by default (see
`DynamicHMC.NoProgressReport`), but this default will likely change in future
releases. 

`ad_backend`: The automatic differentiation backend to use for gradient
computation, specified as either a symbol or a `Val` type with a symbol that
refers to an AD package. See [LogDensityProblems.jl](https://tamaspapp.eu/LogDensityProblems.jl/stable/#Automatic-differentiation)
for supported packages, including `ForwardDiff`, `ReverseDiff`, `Zygote`, and `Tracker`.

For more details see https://tamaspapp.eu/DynamicHMC.jl/stable/interface/
# Example

```jldoctest
julia> using LinearAlgebra, ReverseDiff

julia> config = dynamichmc(
           warmup_stages=default_warmup_stages(
               M=Symmetric, # adapt dense positive definite metric
               stepsize_adaptation=DualAveraging(δ=0.9), # target acceptance rate 0.9
               doubling_stages=7, # 7-stage metric adaptation
           ),
           reporter=LogProgressReport(), # log progress using Logging
           ad_backend=Val(:ReverseDiff), # use ReverseDiff AD package
       );
```
"""
function dynamichmc(;
    init          = ()
  , warmup_stages = DynamicHMC.default_warmup_stages()
  , algorithm     = DynamicHMC.NUTS()
  , reporter      = DynamicHMC.NoProgressReport()
  , ad_backend    = Val(:ForwardDiff)
)
    DynamicHMCConfig(init, warmup_stages, algorithm, reporter, ad_backend)
end

function SampleChains.newchain(rng::Random.AbstractRNG, config::DynamicHMCConfig, ℓ, tr)
    P = LogDensityProblems.TransformedLogDensity(tr, ℓ)
    ∇P = LogDensityProblems.ADgradient(config.ad_backend, P)
    reporter = config.reporter

    results = DynamicHMC.mcmc_keep_warmup(rng, ∇P, 0; 
          initialization = config.init          
        , warmup_stages  = config.warmup_stages 
        , algorithm      = config.algorithm     
        , reporter       = reporter     
        )

    steps = DynamicHMC.mcmc_steps(
        results.sampling_logdensity,
        results.final_warmup_state,
    )
    
    Q = results.final_warmup_state.Q
    Q, tree_stats = DynamicHMC.mcmc_next_step(steps, Q)
    chain = DynamicHMCChain(tr, Q, tree_stats, steps, reporter)
end

function SampleChains.newchain(config::DynamicHMCConfig, ℓ, tr)
    return newchain(Random.GLOBAL_RNG, config, ℓ, tr)
end

function SampleChains.sample!(chain::DynamicHMCChain, n::Int=1000)
    reporter = getfield(chain, :reporter)
    mcmc_reporter = DynamicHMC.make_mcmc_reporter(reporter, n; currently_warmup = false)
    @cleanbreak for j in 1:n
        Q, tree_stats = step!(chain)
        pushsample!(chain, Q, tree_stats)
        DynamicHMC.report(mcmc_reporter, j)
    end 
    return chain
end


end
