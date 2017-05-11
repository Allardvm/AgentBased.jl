# TODO: Type-stability of HDF5 functions should be checked.
# TODO: Implement proper tests.
# TODO: the passthrough function can be more efficient.
# TODO: Skip iter in collectdata it is `nothing` in the generated function.
# TODO: Writers close their remote channel to communicate that they are finished. Should
#       implement automatic reopening of the channel when the Writer is re-used.
# TODO: Check that HDF5Writer has the correct supported datatypes.

module AgentBased

using HDF5
using DataFrames

include("utils.jl")
include("base.jl")
include("writer.jl")
include("writer_hdf5.jl")
include("writer_dataframe.jl")
include("runbatch.jl")

export Reporter, Collector, Writer, HDF5Writer, DataFrameWriter, runbatch, collectdata,
       flushdata

end # module AgentBased
