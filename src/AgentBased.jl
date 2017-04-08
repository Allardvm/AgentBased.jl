module AgentBased


using HDF5


include("utils.jl")


export Collector, Reporter, simbatch, collectdata, flushdata


type TypedBuffer{T}
    size::Int # The current number of data entries in the buffer.
    maxsize::Int # The maximum number of data entries in the buffer.
    data::T # Tuple that contains a fixed number of data vectors with length `maxsize`.

    function TypedBuffer{T}(size, maxsize, data) where T
        @assert size == 0
        @assert maxsize > 0
        @assert typeof(data) <: NTuple{length(data),Array}
        return new(size, maxsize, data)
    end
end


function TypedBuffer(types::Vector{DataType}, maxsize::Integer)
    n_datavectors = length(types)
    data = ntuple((idx) -> Vector{types[idx]}(maxsize), n_datavectors)
    return TypedBuffer{typeof(data)}(0, maxsize, data)
end


type Reporter{N}
    iter::Function # Returns an iterable when called as iter(model, agents, exp)
    names::Vector{String} # The names of the pieces of data.
    types::Vector{DataType} # The types of the pieces of data.
    calls::Vector{Function} # The functions that return the corresponding values.

    function Reporter{N}(iter, names, types, calls) where N
        @assert length(names) == N
        @assert length(types) == N
        @assert length(calls) == N
        return new(iter, names, types, calls)
    end
end


"""
    Reporter(args...)

Return a `Reporter{N}` that specifies the data that should be collected. Each of `N` elements of
`args` is a `Tuple{String, DataType, Function}` that specifies a piece of collected data. The
`String` specifies the name of the piece of data and the `DataType` its type. The `Function` in the
tuple should return the corresponding value when called as `f(model, agents, exp, i)`. The function
is called once for each element `i` of the iterable that is returned when calling `iter` as
`iter(model, agents, exp)`. This can be used to, for example, collect data on all individual
agents.

# Arguments
* `iter::Function`: specifies the iterable to be used when collecting data.
* `args::Tuple{String, DataType, Function}...`: specifies the data that should be collected.
"""
function Reporter(iter::Function, args::Tuple{String,DataType,Function}...)
    N = length(args)
    names = String[arg[1] for arg in args]
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
    buffer::TypedBuffer{T} # Stores data entries until the buffer is full.
    writequeue::RemoteChannel{Channel{TypedBuffer{T}}} # References the master process' queue to
                                                       # which to send the buffer when it is full.
end


"""
    Collector(reporter; condition = always_true, prepare = always_nothing, finish = always_nothing,                 chunksz = 15000)

Return a `Collector{T,N}` that collects data according to the specification in the reporter.

# Arguments
* `reporter::Reporter{N}`: specifies the data that should be collected.
* `condition::Function`: specifies the conditions under which the collector should collect data.
    Calls to collect data with the collector will only proceed when `condition(model, agents, exp)` returns `true`. Defaults to a function that always returns `true`.
* `prepare::Function`: specifies a function to run right before the collector collects data, but
    after the check that data should be collected. The function is called as `prepare(model,
    agents, exp)`. Defaults to an empty function.
* `finish::Function`: specifies a function to run right after the collector collected data, but
    only if data was actually collected. The function is called as `finish(model, agents, exp)`.
    Defaults to an empty function.
* `chunksz::Integer`: specifies the number of sets of data that the collector's buffer can store
    before sending it off to the master process to write it to disk.
"""
function Collector{N}(reporter::Reporter{N};
                      condition::Function = always_true, prepare::Function = always_nothing,
                      finish::Function = always_nothing, chunksz::Integer = 15000)
    buffer = TypedBuffer(reporter.types, chunksz)
    T = typeof(buffer).parameters[1]
    writequeue = RemoteChannel(() -> Channel{TypedBuffer{T}}(2 * nworkers()), 1)
    return Collector{T,N}(reporter, condition, prepare, finish, buffer, writequeue)
end


"""
    collectdata(collector, model, agents, exp)

Collect data with the `collector`. If the `collector`'s buffer is full, automatically send the
buffered data to the master process to write it to disk.

# Arguments
* `collector::Collector{T,N}`: the collector to update.
* `model::Any`: an object that contains the variables required to collect data.
* `agents::Array`: a vector that contains the agents.
* `exp::Any`: an object that holds the parameters that were used to configure the current run of
    the simulation.
"""
@generated function collectdata{T,N}(collector::Collector{T,N}, model, agents, exp)
    return quote
        if collector.condition(model, agents, exp) == true
            collector.prepare(model, agents, exp)

            # Hoist from the loop.
            data = collector.buffer.data
            calls = collector.reporter.calls
            for i in collector.reporter.iter(model, agents, exp)
                ensureroom(collector)
                size = (collector.buffer.size += 1)

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

            collector.finish(model, agents, exp)
        end
        return nothing
    end
end


"""
    flushdata(collector)

Send the buffered data in the 'collector' to the master process to write it to disk. Has to be
called right before the simulation function returns to ensure that the remaining data is written
to disk.

# Arguments
* `collector::Collector`: the collector whose buffer to flush.
"""
function flushdata(collector::Collector)
    put!(collector.writequeue, deepcopy(collector.buffer))
    collector.buffer.size = 0
    return nothing
end


function ensureroom(collector::Collector)
    if collector.buffer.size >= collector.buffer.maxsize
        flushdata(collector)
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
* `expqueue::Vector{T}`: a vector that contains objects that hold the parameters needed to
    configure a run of the simulation.
* `filename::String`: the HDF5 file to in which to store the data.
* `groupname::String`: the group in the HDF5 file in which to store the data.
* `writemode::String`: the writemode to use for `filename`. Can be either "r+" (read-write,
    preserving any existing contents) or "w" (read-write, destroying any existing contents, if any).
* `collector::Collector`: the collector to use to collect data.
"""
function simbatch{T}(f::Function, expqueue::Vector{T}, filename::String, groupname::String,
                     writemode::String, collector::Collector)
    hdf5file = h5open(filename, writemode)
    try
        if exists(hdf5file, groupname)
            error("the specified groupname already exists")
        else
            hdf5group = g_create(hdf5file, groupname)
            juliaversion = IOBuffer()
            versioninfo(juliaversion, false)
            attrs(hdf5group)["JuliaVersion"] = String(take!(juliaversion))
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


function pmaphdf5{Experiment}(hdf5group::HDF5Group, f::Function, expqueue::Vector{Experiment},
                              collector::Collector)
    next_expidx = 0
    getnext_expidx() = (next_expidx += 1)

    overview = zeros(Int, nworkers())
    updateoverview(wpid, expidx) = (overview[ifelse(nprocs() == 1, 1, wpid - 1)] = expidx)

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
                    cur_expidx = getnext_expidx()
                    while cur_expidx <= last_expidx
                        updateoverview(i_wpid, cur_expidx)
                        print("$(now()): Computing experiments $overview...\n")
                        status = remotecall_fetch(f, i_wpid, expqueue[cur_expidx], cur_expidx,
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

            # Hoist from the loop.
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
                            dset[(writeoffset + 1):newoffset] = view(data[$idx], 1:size);
                            close(dset);
                    );
                end;
                ex)
            writeoffset = newoffset
        end
        return writeoffset
    end
end


end # module AgentBased
