using TupleVectors
function divergence_indices(chain::DynamicHMCChain)
    inds = findall(isdivergence, SampleChains.info(chain))
end


function divergences(chain::DynamicHMCChain)
    TupleVector(chain[divergence_indices(chain)])
end

using SampleChainsDynamicHMC.DynamicHMC: TreeStatisticsNUTS
using SampleChainsDynamicHMC.DynamicHMC: is_divergent

isdivergence(tree::TreeStatisticsNUTS) = tree.termination.left == tree.termination.right
