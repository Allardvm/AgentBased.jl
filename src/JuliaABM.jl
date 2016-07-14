module JuliaABM

using HDF5
import Base.flush

export Collector, Report, simbatch, update!, flush

type TypedBuffer{T <: NTuple}
    size::Int
    maxsize::Int
    data::T

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
    names::Array{ASCIIString,1}
    types::Array{DataType,1}
    calls::Array{Function,1}

    function Report(names, types, calls)
        @assert length(names) == N
        @assert length(types) == N
        @assert length(calls) == N
        return new(names, types, calls)
    end
end

function Report(args...)
    @assert eltype(args) <: Tuple "expected one or more Tuple{ASCIISTring, DataType, Function}"
    N = length(args)
    names = [arg[1]::ASCIIString for arg in args]
    types = [arg[2]::DataType for arg in args]
    calls = [arg[3]::Function for arg in args]
    return Report{N}(names, types, calls)
end

type Collector{M,A,T}
    modelreport::Report{M} # Similar for each agent
    agentreport::Report{A} # Different for each agent
    condition::Function
    prepare::Function
    finish::Function
    buffer::TypedBuffer{T}
    writequeue::RemoteRef{Channel{TypedBuffer{T}}}
end

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

@generated function update!{M,A,T}(collector::Collector{M,A,T}, model, agents, exp)
    ex = quote
        if collector.condition(model, agents, exp) == true
            collector.prepare(model, agents, exp)
            $(if A > 0 # Only compile this part when there is an agentreport.
                quote
                    for i_agent in agents
                        ensureroom!(collector)
                        collector.buffer.size += 1

                        # Manually hoist field access from the inner loops.
                        data = collector.buffer.data
                        size = collector.buffer.size
                        modelcalls = collector.modelreport.calls
                        agentcalls = collector.agentreport.calls

                        # Unroll the loop to avoid type instability due to accessing fields with
                        # heterogenous types.
                        $(unrollex = :();
                          for idx in 1:M;
                              retype = T.parameters[idx].parameters[1];
                              unrollex = :($(unrollex.args...);
                                           @inbounds data[$idx][size] =
                                               modelcalls[$idx](model, i_agent, exp)::$retype);
                          end;
                          unrollex)

                        # Agent report accessed in an unrolled loop to avoid type instability
                        # due to accessing fields with heterogenous types. Access to buffer fields
                        # is offset by `M` to account for the fields of the model report.
                        $(unrollex = :();
                          for idx in 1:A;
                              offsetidx = idx + M;
                              retype = T.parameters[offsetidx].parameters[1];
                              unrollex = :($(unrollex.args...);
                                           @inbounds data[$offsetidx][size] =
                                               agentcalls[$idx](model, i_agent, exp)::$retype);
                          end;
                          unrollex)
                    end
                end
            elseif M > 0 # Only compile this part when there is a modelreport.
                quote
                    ensureroom!(collector)
                    collector.buffer.size += 1

                    # Manually hoist field access from the inner loops.
                    data = collector.buffer.data
                    size = collector.buffer.size
                    modelcalls = collector.modelreport.calls

                    # Unroll the loop to avoid type instability due to accessing fields with
                    # heterogenous types.
                    $(unrollex = :();
                      for idx in 1:M;
                          retype = T.parameters[idx].parameters[1];
                          unrollex = :($(unrollex.args...);
                                       @inbounds data[$idx][size] =
                                           modelcalls[$idx](model, agents, exp)::$retype);
                      end;
                      unrollex)
                end
            end)
            collector.finish(model, agents, exp)
        end
        return nothing
    end
    return ex
end

function flush(collector::Collector)
    put!(collector.writequeue, deepcopy(collector.buffer))
    return nothing
end

function ensureroom!(collector::Collector)
    if collector.buffer.size >= collector.buffer.maxsize
        put!(collector.writequeue, deepcopy(collector.buffer))
        collector.buffer.size = 0
    end
    return nothing
end

"""
    simbatch(f::Function, filename::ASCIIString, groupname::ASCIIString, writemode::ASCIIString,
             expqueue::Array{Experiment,1}, collector::Collector)

Runs the model for each set of parameters in `expqueue`.

Writes the results as CSV to `filename`. `Writemode` specifies whether the results should be
appended to `filename` ("append") or written to a new file ("create").
"""
function simbatch{Experiment}(f::Function, filename::ASCIIString, groupname::ASCIIString,
                              writemode::ASCIIString, expqueue::Array{Experiment,1},
                              collector::Collector)
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
                        status = remotecall_fetch(i_wpid, f, expqueue[cur_expidx], cur_expidx, collector)
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
                                       collector::Collector{M,A,T},
                                       writequeue::TypedBuffer{T})
    ex = quote
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
                    $(unrollex = :();
                      for idx in 1:M;
                          retype = T.parameters[idx].parameters[1];
                          unrollex = :($(unrollex.args...);
                                       if exists(hdf5group, m_names[$idx]);
                                           dset = d_open(hdf5group, m_names[$idx]);
                                           set_dims!(dset, (newoffset, ));
                                       else;
                                           dset = d_create(hdf5group, m_names[$idx], $retype,
                                                                  ((size, ), (-1, )), "chunk",
                                                                  (maxsize, ));
                                       end;
                                       dset[(writeoffset + 1):newoffset] = sub(data[$idx],
                                                                               1:size);
                                       close(dset);
                          );
                      end;
                      unrollex)

                    # Unroll the loop to avoid type instability due to accessing fields with
                    # heterogenous types. Access to buffer fields is offset by `M` to account for
                    # the fields of the model report.
                    $(unrollex = :();
                      for idx in 1:A;
                          offsetidx = idx + M;
                          retype = T.parameters[offsetidx].parameters[1];
                          unrollex = :($(unrollex.args...);
                                       if exists(hdf5group, a_names[$idx]);
                                           dset = d_open(hdf5group, a_names[$idx]);
                                           set_dims!(dset, (newoffset, ));
                                       else;
                                           dset = d_create(hdf5group, a_names[$idx], $retype,
                                                                  ((size, ), (-1, )), "chunk",
                                                                  (maxsize, ));
                                       end;
                                       dset[(writeoffset + 1):newoffset] = sub(data[$offsetidx],
                                                                               1:size);
                                       close(dset);
                          );
                      end;
                      unrollex)
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
                    $(unrollex = :();
                      for idx in 1:M;
                          retype = T.parameters[idx].parameters[1];
                          unrollex = :($(unrollex.args...);
                                       if exists(hdf5group, m_names[$idx]);
                                           dset = d_open(hdf5group, m_names[$idx]);
                                           set_dims!(dset, (newoffset, ));
                                       else;
                                           dset = d_create(hdf5group, m_names[$idx], $retype,
                                                                  ((size, ), (-1, )), "chunk",
                                                                  (maxsize, ));
                                       end;
                                       dset[(writeoffset + 1):newoffset] = sub(data[$idx], 1:size);
                                       close(dset);
                          );
                      end;
                      unrollex)
                end
            end)
            writeoffset = newoffset
        end
        return writeoffset
    end
    return ex
end

end # module JuliaABM
