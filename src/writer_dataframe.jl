type DataFrameWriter{C,T} <: Writer{C,T}
    results::DataFrame
    collector::C # The collector used by the Writer.
    localbuffer::TypedBuffer{T} # Stores data entries until the buffer is full.
    remotequeue::RemoteChannel{Channel{TypedBuffer{T}}} # References the channel on the master
                                                        # to which the buffer should be flushed.
end


"""
    DataFrameWriter(collector; chunksz = 15000, channels_per_worker = 2)

Return a `Writer` that returns the `collector`'s data as a DataFrame.

# Arguments
* `collector::Collector`: the Collector to use to collect data.
* `chunksz::Integer`: specifies the number of sets of data that the collector's buffer can store
    before sending it off to the master process to write it to disk.
* `channels_per_worker::Integer`: specifies the number of channels that the master process should
    allocate for each worker.
"""
function DataFrameWriter(collector::Collector{C};
                         chunksz::Integer = 15000, channels_per_worker::Integer = 2) where C
    localbuffer = TypedBuffer(collector, chunksz)
    buffertype = typeof(localbuffer)
    remotequeue = RemoteChannel(() -> Channel{buffertype}(channels_per_worker * nworkers()), 1)

    df_columns = [Array{coltype}(0) for coltype in C.types]
    df_names = [Symbol(colname) for colname in collector.names]
    results = DataFrame(df_columns, df_names)

    return DataFrameWriter(results, collector, localbuffer, remotequeue)
end


function open_sink(writer::DataFrameWriter)
    return nothing
end


function close_sink(sink::Void, writer::DataFrameWriter)
    return nothing
end


function get_return(writer::DataFrameWriter)
    return writer.results
end


function insert!(sink::Void, writer::DataFrameWriter, buffer::TypedBuffer{T}) where T
    n_datavectors = length(T.parameters)

    # Hoist from the loop.
    results = writer.results
    data = buffer.data
    size = buffer.size
    for datavector_idx in 1:n_datavectors
        @inbounds append!(results[datavector_idx], view(data[datavector_idx], 1:size))
    end

    return nothing
end
