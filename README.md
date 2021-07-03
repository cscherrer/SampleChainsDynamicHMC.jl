# SampleChainsDynamicHMC

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cscherrer.github.io/SampleChainsDynamicHMC.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cscherrer.github.io/SampleChainsDynamicHMC.jl/dev)
[![Build Status](https://github.com/cscherrer/SampleChainsDynamicHMC.jl/workflows/CI/badge.svg)](https://github.com/cscherrer/SampleChainsDynamicHMC.jl/actions)
[![Coverage](https://codecov.io/gh/cscherrer/SampleChainsDynamicHMC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/cscherrer/SampleChainsDynamicHMC.jl)

Setting up:
```julia
julia> using SampleChainsDynamicHMC

julia> using TransformVariables

julia> function ℓ(nt)
               z = nt.x/nt.σ
               return -z^2 - nt.σ - log(nt.σ)
       end
ℓ (generic function with 1 method)

julia> t = as((x=asℝ, σ=asℝ₊))
TransformVariables.TransformTuple{NamedTuple{(:x, :σ), Tuple{TransformVariables.Identity, TransformVariables.ShiftedExp{true, Float64}}}}((x = asℝ, σ = asℝ₊), 2)
```

Initialize and take some samples:
```julia
julia> chain = newchain(DynamicHMCChain, ℓ, t)
1-element Chain with schema (x = Float64, σ = Float64)
(x = -0.66±0.0, σ = 0.65±0.0)

julia> sample!(chain, 9)
10-element Chain with schema (x = Float64, σ = Float64)
(x = -0.36±0.38, σ = 1.26±0.69)

julia> sample!(chain, 90)
100-element Chain with schema (x = Float64, σ = Float64)
(x = -0.32±0.68, σ = 1.06±0.66)

julia> chain[1]
(x = -0.660818661864279, σ = 0.6482824278360845)

julia> chain.x[1:10]
10-element ElasticArrays.ElasticVector{Float64, 0, Vector{Float64}}:
 -0.660818661864279
 -0.31966349282522916
 -0.5030732787889958
 -0.27788387641411594
 -0.9287874718868021
 -0.6260927333733151
  0.4303096842134812
 -0.3844104968943612
  0.05987431572954072
 -0.351002647246055
```

Or multiple chains:

```julia
julia> chains = newchain(4, DynamicHMCChain, ℓ, t)
4-element MultiChain with 4 chains and schema (x = Float64, σ = Float64)
(x = -0.38±0.75, σ = 0.76±0.76)

julia> sample!(chains, 9)
40-element MultiChain with 4 chains and schema (x = Float64, σ = Float64)
(x = -0.11±0.73, σ = 0.83±0.8)

julia> sample!(chains, 90)
400-element MultiChain with 4 chains and schema (x = Float64, σ = Float64)
(x = -0.18±0.75, σ = 0.9±0.97)

julia> samples(chains)
400-element TupleVector with schema (x = Float64, σ = Float64)
(x = -0.18±0.75, σ = 0.9±0.97)

julia> getchains(chains) .|> summarize
4-element Vector{NamedTuple{(:x, :σ), Tuple{RealSummary, RealSummary}}}:
 (x = -0.22±0.73, σ = 1.4±0.92)
 (x = -0.031±0.33, σ = 0.46±0.37)
 (x = -0.0214±0.014, σ = 0.07497±0.0044)
 (x = -0.46±1.2, σ = 1.7±1.0)
```

A `MultiChain` is still represented abstractly similarly to a single chain, for easy comptuations:
```julia
julia> chains[1]
(x = -0.36681258114618465, σ = 1.7508963122497017)

julia> chains.x[1:10]
vcat(10-element view(::ElasticArrays.ElasticVector{Float64, 0, Vector{Float64}}, 1:10) with eltype Float64, 0-element view(::ElasticArrays.ElasticVector{Float64, 0, Vector{Float64}}, 1:0) with eltype Float64, 0-element view(::ElasticArrays.ElasticVector{Float64, 0, Vector{Float64}}, 1:0) with eltype Float64, 0-element view(::ElasticArrays.ElasticVector{Float64, 0, Vector{Float64}}, 1:0) with eltype Float64):
 -0.36681258114618465
 -0.09339967949694516
 -0.3089171887973833
 -1.5420534117776032
 -0.10574714292144685
 -0.11312594562766448
 -0.008799704824529742
  0.5209894936643252
 -0.11204122979765113
 -1.100922340370071
 ```
