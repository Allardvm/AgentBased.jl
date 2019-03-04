using Distributed


const SUPPORTED_HDF5 = (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float32, Float64)

mutable struct HDF5Writer{C,T} <: Writer{C,T}
    filename::String
    groupname::String
    writemode::String
    writeoffset::Int
    collector::C # The collector used by the Writer.
    localbuffer::TypedBuffer{T} # Stores data entries until the buffer is full.
    remotequeue::RemoteChannel{Channel{TypedBuffer{T}}} # References the channel on the master
                                                        # to which the buffer should be flushed.
end


"""
    HDF5Writer(collector; chunksz = 15000, channels_per_worker = 2)

Return a `Writer` that writes the `collector`'s data to an HDF5 file.

# Arguments
* `filename::String`: the HDF5 file in which to store the data.
* `groupname::String`: the group in the HDF5 file in which to store the data.
* `collector::Collector`: the Collector to use to collect data.
* `chunksz::Integer`: specifies the number of sets of data that the collector's buffer can store
    before sending it off to the master process to write it to disk.
* `channels_per_worker::Integer`: specifies the number of channels that the master process should
    allocate for each worker.
* `writemode::String`: the writemode to use for `filename`. Can be either "r+" (read-write,
    preserving any existing contents) or "w" (read-write, destroying any existing contents, if any).
"""
function HDF5Writer(filename::String, groupname::String, collector::Collector{C};
                    chunksz::Integer = 15000, channels_per_worker::Integer = 2,
                    writemode::String = "w") where C
    for datatype in C.types
        @assert(in(datatype, SUPPORTED_HDF5),
                "collector contains a datatype that isn't supported by HDF5Writer, got: $datatype")
    end
    localbuffer = TypedBuffer(collector, chunksz)
    buffertype = typeof(localbuffer)
    remotequeue = RemoteChannel(() -> Channel{buffertype}(channels_per_worker * nworkers()), 1)
    return HDF5Writer(filename, groupname, writemode, 0, collector, localbuffer, remotequeue)
end


function open_sink(writer::HDF5Writer)
    hdf5file = h5open(writer.filename, writer.writemode)
    if exists(hdf5file, writer.groupname)
        close(hdf5file)
        error("the specified groupname already exists")
    else
        hdf5group = g_create(hdf5file, writer.groupname)
    end
    return hdf5group, hdf5file
end


function close_sink(sink::Tuple{HDF5Group,HDF5File}, writer::HDF5Writer)
    foreach(close, sink)
    return nothing
end


function get_return(writer::HDF5Writer)
    return nothing
end


# Uses a generated function to unroll the loop over the datavector tuple. This avoids the type
# instability that results from iterating over a heterogenous tuple.
@generated function insert!(sink::Tuple{HDF5Group,HDF5File}, writer::HDF5Writer,
                            buffer::TypedBuffer{T}) where T
    n_datavectors = length(T.parameters)

    ex = :()
    for datavector_idx in 1:n_datavectors
        datatype = T.parameters[datavector_idx].parameters[1]

        ex = :($(ex.args...);
            name = names[$datavector_idx];
            if exists(sink[1], name);
                dset = d_open(sink[1], name);
                set_dims!(dset, (newoffset, ));
            else;
                dset = d_create(sink[1], name, $datatype, ((size, ), (-1, )), "chunk",
                                (maxsize, ));
            end;
            dset[(writer.writeoffset + 1):newoffset] = view(data[$datavector_idx], 1:size);
            close(dset);)
    end

    return quote
        if buffer.size > 0
            newoffset = writer.writeoffset + buffer.size

            # Hoist from the loop.
            names = writer.collector.names
            data = buffer.data
            size = buffer.size
            maxsize = buffer.maxsize

            $ex

            writer.writeoffset = newoffset
        end
        return nothing
    end
end
