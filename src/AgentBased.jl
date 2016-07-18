module AgentBased

using HDF5
import Base.flush

export Collector, Reporter, simbatch, update, flush

type TypedBuffer{T <: NTuple}
    size::Int # The current number of data entries in the buffer.
    maxsize::Int # The maximum number of data entries in the buffer.
    data::T # Tuple that contains a fixed number of 1-dimensional arrays with length `maxsize`.

    function TypedBuffer(size, maxsize, data)
        @assert size == 0
        @assert maxsize > 0
        @assert typeof(data) <: NTuple{length(data), Array}
        return new(size, maxsize, data)
    end
end

function TypedBuffer(types::Array{DataType,1}, maxsize::Int)
    data = ntuple((idx) -> Array(types[idx]::DataType, maxsize), length(types))
    return TypedBuffer{typeof(data)}(0, maxsize, data)
end

type Reporter{N}
    iter::Function # Returns an iterable when called as iter(model, agents, exp)
    names::Array{ASCIIString,1} # The names of the pieces of data.
    types::Array{DataType,1} # The types of the pieces of data.
    calls::Array{Function,1} # The functions that return the corresponding values.

    function Reporter(iter, names, types, calls)
        @assert length(names) == N
        @assert length(types) == N
        @assert length(calls) == N
        return new(iter, names, types, calls)
    end
end

"""
    Reporter(args...)

Return a `Reporter{N}` that specifies the data that should be collected. Each of `N` elements of
`args` is a `Tuple{ASCIIString, DataType, Function}` that specifies a piece of collected data. The
`ASCIIString` specifies the name of the piece of data and the `DataType` its type. The `Function`
returns the corresponding value when called as `f(model, agents, exp)` if the reporter is used as a
modelreporter and as `f(model, agent, exp)` if the reporter is used as an agentreporter.

# Arguments
* `args::Tuple{ASCIIString, DataType, Function}...`: specifies the data that should be collected.
"""
function Reporter(iter::Function, args::Tuple{ASCIIString, DataType, Function}...)
    N = length(args)
    names = ASCIIString[arg[1] for arg in args]
    types = DataType[arg[2] for arg in args]
    calls = Function[arg[3] for arg in args]
    return Reporter{N}(iter, names, types, calls)
end

type Collector{T,N}
    reporter::Reporter{N} # Specifies the data that should be collected.
    condition::Function # Specifies the conditions under which calls to update the
                        # collector should proceed. Continues if and only if `condition(model,
                        # agents, exp)` returns `true`.
    prepare::Function # Specifies algorithms to run right before the collector collects data.
                      # Called as `prepare(model, agents, exp)`.
    finish::Function # Specifies algorithms that run right after the collector collects data.
                     # Called as `finish(model, agents, exp)`.
    buffer::TypedBuffer{T} # Stores a data entries until the buffer is full.
    writequeue::RemoteRef{Channel{TypedBuffer{T}}} # References the master process' queue to which
                                                   # to send the buffer when it is full.
end

"""
    Collector(reporter, condition, prepare, finish, chunksz)

Return a `Collector{T,N}` that collects data according to the specification provided by the reporter.
When the `agentreporter` is empty, the collector collects data only once for every
call to update it. When it is not empty, the collector collects data once for every individual
agent.

# Arguments
* `reporter::Reporter{M} = Reporter()`: specifies the data that should be collected.
* `condition::Function = () -> true`: specifies the conditions under which calls to update the
collector should proceed. Continues if and only if `condition(model, agents, exp)` returns `true`.
* `prepare::Function = () -> true`: specifies algorithms to run right before the collector collects
data. Called as `prepare(model, agents, exp)`.
* `finish::Function = () -> nothing`: specifies algorithms that run right after the collector
collects data. Called as `finish(model, agents, exp)`.
* `chunksz::Int = 10000`: specifies the number of pieces of data that the collector's buffer can
store before sending it off to the master process to write it to disk.
"""
function Collector{N}(reporter::Reporter{N} = Reporter(), condition::Function = () -> true,
                      prepare::Function = () -> nothing, finish::Function = () -> nothing,
                      chunksz::Int = 15000)
    buffer = TypedBuffer(reporter.types, chunksz)
    T = typeof(buffer).parameters[1]
    writequeue = RemoteRef(() -> Channel{TypedBuffer{T}}(2 * nworkers()), 1)
    return Collector{T,N}(reporter, condition, prepare, finish, buffer, writequeue)
end

"""
    update(collector, model, agents, exp)

Update the `collector` by collecting data according to the specification provided by its fields.
The data collection, condition, prepare, and finish functions specified in the `collector` are
called as f(`model`, `agents`, `exp`). Collect data on individual agents by calling the data
collection functions in the `agentreporter` field of the `collector` once for each element `i_agent`
in `agents` as f(`model`, `i_agent`. `exp`). If the `collector`'s buffer is full, automatically
send the buffered data to the master process to write it to disk.

# Arguments
* `collector::Collector`: the collector to update.
* `model::Any`: an object that contains the variables required to collect data.
* `agents::Array`: an N-dimensional array that contains the agents.
* `exp::Any`: an object that holds the parameters that were used to configure the current run of
the simulation.
"""
@generated function update{T,N}(collector::Collector{T,N}, model, agents, exp)
    return quote
        if collector.condition(model, agents, exp) == true
            collector.prepare(model, agents, exp)
            quote
                for i in collector.reporter()
                    ensureroom(collector)
                    collector.buffer.size += 1

                    # Manually hoist field access from the inner loops.
                    data = collector.buffer.data
                    size = collector.buffer.size
                    calls = collector.reporter.calls

                    # Unroll the loop to avoid type instability due to accessing fields with
                    # heterogenous types.
                    $(ex = :();
                        for idx in 1:N;
                            retype = T.parameters[idx].parameters[1];
                            ex = :($(ex.args...);
                                   @inbounds data[$idx][size] =
                                       calls[$idx](model, agents, exp, i)::$retype);
                        end;
                        ex)
                end
            end
            collector.finish(model, agents, exp)
        end
        return nothing
    end
end

"""
    flush(collector)

Manually send the buffered data in the 'collector' to the master process to write it to disk.
Usually called right before the simulation function returns to ensure that the remaining data
is written to disk.

# Arguments
* `collector::Collector`: the collector whose buffer to flush.
"""
function flush(collector::Collector)
    put!(collector.writequeue, deepcopy(collector.buffer))
    collector.buffer.size = 0
    return nothing
end

function ensureroom(collector::Collector)
    if collector.buffer.size >= collector.buffer.maxsize
        flush(collector)
    end
    return nothing
end

"""
    simbatch(f, expqueue, filename, groupname, writemode, collector)

Run the simulation specified in `f` in parallel by calling it as `f(expqueue[idx], idx, collector)`
for each entry `expqueue[idx]` in `expqueue`. Individual runs of the simulation are automatically
distributed over all available workers by the master process. The master process write the results
to disk as the HDF5 file `filename`, in the group `groupname`, using the specified `writemode`.
Collect data using the `collector`.

# Arguments
* `f::Function`: the simulation function.
* `expqueue::Array{T,1}`: a 1-dimensional array that contains objects that hold the parameters
needed to configure a run of the simulation.
* `filename::ASCIIString`: the HDF5 file to in which to store the data.
* `groupname::ASCIIString`: the group in the HDF5 file in which to store the data.
* `writemode::ASCIIString`: the writemode to use for `filename`. Can be either "r+" (read-write,
preserving any existing contents) or "w" (read-write, destroying any existing contents, if any).
* `collector::Collector`: the collector to use to collect data.
"""
function simbatch{T}(f::Function, expqueue::Array{T,1}, filename::ASCIIString,
                  groupname::ASCIIString, writemode::ASCIIString, collector::Collector)
    hdf5file = h5open(filename, writemode)
    try
        if exists(hdf5file, groupname)
            error("the specified groupname already exists")
        else
            hdf5group = g_create(hdf5file, groupname)
            juliaversion = IOBuffer()
            versioninfo(juliaversion, false)
            attrs(hdf5group)["JuliaVersion"] = takebuf_string(juliaversion)
            attrs(hdf5group)["DateStarted"] = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
        end
        println(now(), ": Queued ", length(expqueue), " experiments on ", nworkers(), " cores")
        pmaphdf5(hdf5group, f, expqueue, collector)
        println(now(), ": All experiments have been completed")
        attrs(hdf5group)["DateFinished"] = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
        close(hdf5group)
    finally
        close(hdf5file)
    end
    return nothing
end

function pmaphdf5{Experiment}(hdf5group::HDF5Group, f::Function, expqueue::Array{Experiment,1},
                              collector::Collector)
    next_expidx::Int64 = 0
    getnext_expidx() = (next_expidx += 1)

    overview::Array{Int64,1} = zeros(Int64, nworkers())
    updateoverview(wpid::Int64, expidx::Int64) =
        (overview[ifelse(nprocs() == 1, 1, wpid - 1)] = expidx)

    @sync begin
        @async begin
            writeoffset = 0
            while true
                try
                    writeoffset = inserthdf5!(hdf5group, writeoffset, collector,
                                              take!(collector.writequeue))
                catch err
                    if isa(err, InvalidStateException) && err.state == :closed
                        break # Resultsqueue is empty and was closed because workers are done.
                    else
                        rethrow(err)
                    end
                end
            end
        end
        @sync begin # Prevents race conditions when a worker's async task throws an error.
            for i_wpid in workers()
                @async begin
                    last_expidx = length(expqueue)
                    cur_expidx::Int64 = getnext_expidx()
                    while cur_expidx <= last_expidx
                        updateoverview(i_wpid, cur_expidx)
                        print("$(now()): Computing experiments $overview...\n")
                        status = remotecall_fetch(i_wpid, f, expqueue[cur_expidx], cur_expidx,
                                                  collector)
                        status != "" && info("Experiment $cur_expidx: $status")
                        cur_expidx = getnext_expidx()
                    end
                    updateoverview(i_wpid, 0)
                    print("$(now()): Computing experiments $overview...\n")
                end
            end
        end
        close(collector.writequeue)
    end
    return nothing
end

@generated function inserthdf5!{T,N}(hdf5group::HDF5Group, writeoffset::Integer,
                                     collector::Collector{T,N}, writequeue::TypedBuffer{T})
    return quote
        if writequeue.size > 0
            newoffset = writeoffset + writequeue.size
            quote
                # Manually hoist field access from the inner loops.
                names = collector.reporter.names
                data = writequeue.data
                size = writequeue.size
                maxsize = writequeue.maxsize

                # Unroll the loop to avoid type instability due to accessing fields with
                # heterogenous types.
                $(ex = :();
                    for idx in 1:N;
                        retype = T.parameters[idx].parameters[1];
                        ex = :($(ex.args...);
                                if exists(hdf5group, names[$idx]);
                                    dset = d_open(hdf5group, names[$idx]);
                                    set_dims!(dset, (newoffset, ));
                                else;
                                    dset = d_create(hdf5group, names[$idx], $retype,
                                                    ((size, ), (-1, )), "chunk", (maxsize, ));
                                end;
                                dset[(writeoffset + 1):newoffset] = sub(data[$idx], 1:size);
                                close(dset);
                        );
                    end;
                    ex)
                end
            writeoffset = newoffset
        end
        return writeoffset
    end
end

end # module AgentBased
