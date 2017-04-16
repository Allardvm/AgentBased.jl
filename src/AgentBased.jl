# TODO: Collector, and probably other types, aren't type-stable when using keywords.
# TODO: Type-stability of HDF5 functions should be checked.
# TODO: Implement proper tests.
# TODO: the passthrough function can be more efficient.
# TODO: Skip iter in collectdata it is `nothing` in the generated function.
# TODO: DataWriters close their remote channel to communicate that they are finished. Should
#       implement automatic reopening of the channel when the DataWriter is re-used.

module AgentBased

using HDF5

include("utils.jl")
include("base.jl")
include("datawriter.jl")
include("hdf5writer.jl")

export Reporter, Collector, DataWriter, HDF5Writer, runbatch, collectdata, flushdata

end # module AgentBased
