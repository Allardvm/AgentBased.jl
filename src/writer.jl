abstract type Writer{C,T} end


type TypedBuffer{T<:NTuple{<:Any,Array}}
    size::Int # The current number of data entries in the buffer.
    maxsize::Int # The maximum number of data entries in the buffer.
    data::T # Tuple that contains a fixed number of data vectors with length `maxsize`.
end


@generated function TypedBuffer(collector::Collector{C}, maxsize::Integer) where C
    buffer_datatypes = C.types
    n_datavectors = length(buffer_datatypes)
    T = Tuple{(Vector{buffer_datatypes[i]} for i in 1:n_datavectors)...}

    return :(TypedBuffer{$T}(0, maxsize, ($([:(Vector{$(buffer_datatypes[i])}(maxsize))
                                             for i in 1:n_datavectors]...),)))
end


"""
    collectdata(writer, args...)

Collect data with the collector in `writer`.

When the the `writer`'s local buffer is full, automatically send the buffered data to the master
process to write it to disk.

# Arguments
* `writer::Writer`: the Writer to use.
* `args...`: arguments to pass on to the collector's functions.
"""
@generated function collectdata{C,T}(writer::Writer{C,T}, args...)
    n_calls = length(T.parameters)
    ex = :()
    for call_idx in 1:n_calls
        ex = :($(ex.args...);
               @inbounds data[$call_idx][size] = calls[$call_idx](report_args...))
    end

    return quote
        collector = writer.collector
        if collector.condition(args...) == true
            collector.prepare(args...)

            # Hoist from the loop.
            data = writer.localbuffer.data
            calls = collector.calls

            for report_args in collector.iter(args...)
                ensureroom(writer)
                size = (writer.localbuffer.size += 1)
                $ex
            end

            collector.finish(args...)
        end
        return nothing
    end
end


"""
    flushdata(writer)

Send the `writer`'s locally buffered data to the master process, which will write it to disk.

The flushdata function has to be called right before the simulation function returns. This ensures
that any remaining locally buffered data is send to the master proces before the local buffer is
closed.

# Arguments
* `writer::Writer`: the Writer whose buffer to flush.
"""
function flushdata(writer::Writer)
    if myid() == 1 # Make re-using the buffer safe by forcing a copy when the caller is the master.
        put!(writer.remotequeue, deepcopy(writer.localbuffer))
    else
        put!(writer.remotequeue, writer.localbuffer)
    end
    writer.localbuffer.size = 0
    return nothing
end


function ensureroom(writer::Writer)
    if writer.localbuffer.size >= writer.localbuffer.maxsize
        flushdata(writer)
    end
    return nothing
end
