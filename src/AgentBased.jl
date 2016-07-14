module JuliaABM

using HDF5
import Base.flush

export Collector, Report, simbatch, update, flush

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

type Report{N}
    names::Array{ASCIIString,1} # The names of the pieces of data.
    types::Array{DataType,1} # The types of the pieces of data.
    calls::Array{Function,1} # the functions that return the corresponding values.

    function Report(names, types, calls)
        @assert length(names) == N
        @assert length(types) == N
        @assert length(calls) == N
        return new(names, types, calls)
    end
end

"""
    Report(args...)

Return a `Report` that specifies the data that should be collected. Each element of `args` is a 
`Tuple{ASCIIString, DataType, Function}` that specifies a piece of collected data. The 
`ASCIIString` specifies the name of the piece of data and the `DataType` its type. The `Function`
returns the corresponding value when called as `f(model, agents, exp)` if the report is used as a
modelreport and as `f(model, agent, exp)` if the report is used as an agentreport.

# Arguments
* `args::Tuple{ASCIIString, DataType, Function}...`: specifies the data that should be collected.
"""
function Report(args::Tuple{ASCIIString, DataType, Function}...)
    N = length(args)
    names = [arg[1]::ASCIIString for arg in args]
    types = [arg[2]::DataType for arg in args]
    calls = [arg[3]::Function for arg in args]
    return Report{N}(names, types, calls)
end

type Collector{M,A,T}
    modelreport::Report{M} # Specifies the global model data that should be collected.
    agentreport::Report{A} # Specifies the agent specific data that should be collected.
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
    Collector{M,A,T}(modelreport, agentreport, condition, prepare, finish, chunksz)

Return a `Collector` that collects data according to the specification provided by `modelreport`
and `agentreport`. When the `agentreport` is empty, the collector collects data only once for every
call to update it. When it is not empty, the collector collects data once for every individual
agent.

# Arguments
* `modelreport::Report{M} = Report()`: specifies the global model data that should be collected.
* `agentreport::Report{A} = Report()`: specifies the agent specific data that should be collected.
* `condition::Function = () -> true`: specifies the conditions under which calls to update the
collector should proceed. Continues if and only if `condition(model, agents, exp)` returns `true`.
* `prepare::Function = () -> true`: specifies algorithms to run right before the collector collects
data. Called as `prepare(model, agents, exp)`.
* `finish::Function = () -> nothing`: specifies algorithms that run right after the collector
collects data. Called as `finish(model, agents, exp)`.
* `chunksz::Int = 10000`: specifies the number of pieces of data that the collector's buffer can
store before sending it off to the master process to write it to disk.
"""
function Collector{M,A}(modelreport::Report{M} = Report(), agentreport::Report{A} = Report(),
                        condition::Function = () -> true, prepare::Function = () -> nothing,
                        finish::Function = () -> nothing, chunksz::Int = 10000)
    types = vcat(modelreport.types, agentreport.types)
    buffer = TypedBuffer(types, chunksz)
    T = typeof(buffer).parameters[1]
    writequeue = RemoteRef(() -> Channel{TypedBuffer{T}}(2 * nworkers()), 1)
    return Collector{M,A,T}(modelreport, agentreport, condition, prepare, finish, buffer,
                            writequeue)
end

"""
    update(collector, model, agents, exp)

Update the `collector` by collecting data according to the specification provided by its fields.
The data collection, condition, prepare, and finish functions specified in the `collector` are
called as f(`model`, `agents`, `exp`). Collect data on individual agents by calling the data
collection functions in the `agentreport` field of the `collector` once for each element `i_agent`
in `agents` as f(`model`, `i_agent`. `exp`). If the `collector`'s buffer is full, automatically
send the buffered data to the master process to write it to disk.

# Arguments
* `collector::Collector`: the collector to update.
* `model::Any`: an object that contains the variables required to collect data.
* `agents::Array{Any, N}`: an N-dimensional array that contains the agents.
* `exp::Any`: an object that holds the parameters that were used to configure the current run of
the simulation.
"""
@generated function update{M,A,T}(collector::Collector{M,A,T}, model, agents, exp)
    return quote
        if collector.condition(model, agents, exp) == true
            collector.prepare(model, agents, exp)
            $(if A > 0 # Only compile this part when there is an agentreport.
                quote
                    for i_agent in agents
                        ensureroom(collector)
                        collector.buffer.size += 1

                        # Manually hoist field access from the inner loops.
                        data = collector.buffer.data
                        size = collector.buffer.size
                        modelcalls = collector.modelreport.calls
                        agentcalls = collector.agentreport.calls

                        # Unroll the loop to avoid type instability due to accessing fields with
                        # heterogenous types.
                        $(ex = :();
                          for idx in 1:M;
                              retype = T.parameters[idx].parameters[1];
                              ex = :($(ex.args...);
                                     @inbounds data[$idx][size] =
                                         modelcalls[$idx](model, i_agent, exp)::$retype);
                          end;
                          ex)

                        # Agent report accessed in an unrolled loop to avoid type instability
                        # due to accessing fields with heterogenous types. Access to buffer fields
                        # is offset by `M` to account for the fields of the model report.
                        $(ex = :();
                          for idx in 1:A;
                              offsetidx = idx + M;
                              retype = T.parameters[offsetidx].parameters[1];
                              ex = :($(ex.args...);
                                     @inbounds data[$offsetidx][size] =
                                         agentcalls[$idx](model, i_agent, exp)::$retype);
                          end;
                          ex)
                    end
                end
            elseif M > 0 # Only compile this part when there is a modelreport.
                quote
                    ensureroom(collector)
                    collector.buffer.size += 1

                    # Manually hoist field access from the inner loops.
                    data = collector.buffer.data
                    size = collector.buffer.size
                    modelcalls = collector.modelreport.calls

                    # Unroll the loop to avoid type instability due to accessing fields with
                    # heterogenous types.
                    $(ex = :();
                      for idx in 1:M;
                          retype = T.parameters[idx].parameters[1];
                          ex = :($(ex.args...);
                                 @inbounds data[$idx][size] =
                                     modelcalls[$idx](model, agents, exp)::$retype);
                      end;
                      ex)
                end
            end)
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
* `expqueue::Array{Any,1}`: a 1-dimensional array that contains objects that hold the parameters
needed to configure a run of the simulation.
* `filename::ASCIIString`: the HDF5 file to in which to store the data.
* `groupname::ASCIIString`: the group in the HDF5 file in which to store the data. 
* `writemode::ASCIIString`: the writemode to use for `filename`. Can be either "r+" (read-write,
preserving any existing contents) or "w" (read-write, destroying any existing contents, if any).
* `collector::Collector`: the collector to use to collect data.
"""
function simbatch(f:::Function, expqueue::Array{Any,1}, filename::ASCIIString,
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

function pmaphdf5{Experiment}(hdf5group::HDF5Group, f:::Function, expqueue::Array{Experiment,1},
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
                        status < 0 && info("Experiment $cur_expidx skipped: ", strstatus(status))
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

@generated function inserthdf5!{M,A,T}(hdf5group::HDF5Group, writeoffset::Integer,
                                       collector::Collector{M,A,T}, writequeue::TypedBuffer{T})
    return quote
        if writequeue.size > 0
            newoffset = writeoffset + writequeue.size
            $(if A > 0 # Only compile this part when there is an agentreport.
                quote
                    # Manually hoist field access from the inner loops.
                    m_names = collector.modelreport.names
                    a_names = collector.agentreport.names
                    data = writequeue.data
                    size = writequeue.size
                    maxsize = writequeue.maxsize

                    # Unroll the loop to avoid type instability due to accessing fields with
                    # heterogenous types.
                    $(ex = :();
                      for idx in 1:M;
                          retype = T.parameters[idx].parameters[1];
                          ex = :($(ex.args...);
                                 if exists(hdf5group, m_names[$idx]);
                                     dset = d_open(hdf5group, m_names[$idx]);
                                     set_dims!(dset, (newoffset, ));
                                 else;
                                     dset = d_create(hdf5group, m_names[$idx], $retype,
                                                     ((size, ), (-1, )), "chunk", (maxsize, ));
                                 end;
                                 dset[(writeoffset + 1):newoffset] = sub(data[$idx], 1:size);
                                 close(dset);
                          );
                      end;
                      ex)

                    # Unroll the loop to avoid type instability due to accessing fields with
                    # heterogenous types. Access to buffer fields is offset by `M` to account for
                    # the fields of the model report.
                    $(ex = :();
                      for idx in 1:A;
                          offsetidx = idx + M;
                          retype = T.parameters[offsetidx].parameters[1];
                          ex = :($(ex.args...);
                                 if exists(hdf5group, a_names[$idx]);
                                     dset = d_open(hdf5group, a_names[$idx]);
                                     set_dims!(dset, (newoffset, ));
                                 else;
                                     dset = d_create(hdf5group, a_names[$idx], $retype,
                                                     ((size, ), (-1, )), "chunk", (maxsize, ));
                                 end;
                                 dset[(writeoffset + 1):newoffset] = sub(data[$offsetidx], 1:size);
                                 close(dset);
                          );
                      end;
                      ex)
                end
            elseif M > 0 # Only compile this part when there is a modelreport.
                quote
                    # Manually hoist field access from the inner loops.
                    m_names = collector.modelreport.names
                    data = writequeue.data
                    size = writequeue.size
                    maxsize = writequeue.maxsize

                    # Unroll the loop to avoid type instability due to accessing fields with
                    # heterogenous types.
                    $(ex = :();
                      for idx in 1:M;
                          retype = T.parameters[idx].parameters[1];
                          ex = :($(ex.args...);
                                 if exists(hdf5group, m_names[$idx]);
                                     dset = d_open(hdf5group, m_names[$idx]);
                                     set_dims!(dset, (newoffset, ));
                                 else;
                                     dset = d_create(hdf5group, m_names[$idx], $retype,
                                                     ((size, ), (-1, )), "chunk", (maxsize, ));
                                 end;
                                 dset[(writeoffset + 1):newoffset] = sub(data[$idx], 1:size);
                                 close(dset);
                          );
                      end;
                      ex)
                end
            end)
            writeoffset = newoffset
        end
        return writeoffset
    end
end

end # module JuliaABM
