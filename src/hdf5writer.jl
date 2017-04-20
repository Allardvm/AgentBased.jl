type HDF5Writer{C,T} <: DataWriter{C,T}
    filename::String
    groupname::String
    writemode::String
    writeoffset::Int
    collector::C # The collector used by the DataWriter.
    localbuffer::TypedBuffer{T} # Stores data entries until the buffer is full.
    remotequeue::RemoteChannel{Channel{TypedBuffer{T}}} # References the channel on the master
                                                        # to which the buffer should be flushed.
end


"""
    HDF5Writer(collector; chunksz = 15000, channels_per_worker = 2)

Return a `Collector` that collects data according to the specification in the reporter.

# Arguments
* `filename::String`: the HDF5 file to in which to store the data.
* `groupname::String`: the group in the HDF5 file in which to store the data.
* `collector::Collector`: the Collector to use to collect data.
* `chunksz::Integer`: specifies the number of sets of data that the collector's buffer can store
    before sending it off to the master process to write it to disk.
* `channels_per_worker::Integer`: specifies the number of channels that the master process should
    allocate for each worker.
* `writemode::String`: the writemode to use for `filename`. Can be either "r+" (read-write,
    preserving any existing contents) or "w" (read-write, destroying any existing contents, if any).
"""
function HDF5Writer(filename::String, groupname::String, collector::Collector;
                    chunksz::Integer = 15000, channels_per_worker::Integer = 2,
                    writemode::String = "w")
    localbuffer = TypedBuffer(collector, chunksz)
    buffertype = typeof(localbuffer)
    remotequeue = RemoteChannel(() -> Channel{buffertype}(channels_per_worker * nworkers()), 1)
    return HDF5Writer(filename, groupname, writemode, 0, collector, localbuffer, remotequeue)
end


"""
    runbatch(f, expqueue, filename, groupname, writemode, collector)

Run the simulation specified in `f` in parallel by calling it as `f(expqueue[idx], idx, collector)`
for each entry `expqueue[idx]` in `expqueue`. Individual runs of the simulation are automatically
distributed over all available workers by the master process. The master process write the results
to disk as the HDF5 file `filename`, in the group `groupname`, using the specified `writemode`.
Collect data using the `collector`.

# Arguments
* `f::Function`: the simulation function.
* `expqueue::Vector{T}`: a vector that contains objects that hold the parameters needed to
    configure a run of the simulation.
* `collector::DataWriter`: the collector to use to collect data.
"""
function runbatch{T}(f::Function, expqueue::Vector{T}, writer::HDF5Writer)
    hdf5file = h5open(writer.filename, writer.writemode)
    try
        if exists(hdf5file, writer.groupname)
            error("the specified groupname already exists")
        else
            hdf5group = g_create(hdf5file, writer.groupname)
            juliaversion = IOBuffer()
            versioninfo(juliaversion, false)
            attrs(hdf5group)["JuliaVersion"] = String(take!(juliaversion))
            attrs(hdf5group)["DateStarted"] = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
        end
        println(now(), ": Queued ", length(expqueue), " experiments on ", nworkers(), " cores")
        pmaphdf5(hdf5group, f, expqueue, writer)
        println(now(), ": All experiments have been completed")
        attrs(hdf5group)["DateFinished"] = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
        close(hdf5group)
    finally
        close(hdf5file)
    end
    return nothing
end


function pmaphdf5{Experiment}(hdf5group::HDF5Group, f::Function, expqueue::Vector{Experiment},
                              writer::HDF5Writer)
    next_expidx = 0
    getnext_expidx() = (next_expidx += 1)

    overview = zeros(Int, nworkers())
    updateoverview(wpid, expidx) = (overview[ifelse(nprocs() == 1, 1, wpid - 1)] = expidx)

    @sync begin
        @async begin
            while true
                try
                    inserthdf5!(hdf5group, writer, take!(writer.remotequeue))
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
                                                  writer)
                        status != "" && info("Experiment $cur_expidx: $status")
                        cur_expidx = getnext_expidx()
                    end
                    updateoverview(i_wpid, 0)
                    print("$(now()): Computing experiments $overview...\n")
                end
            end
        end
        close(writer.remotequeue)
    end
    return nothing
end


# Uses a generated function to unroll the loop over the datavector tuple. This avoids the type
# instability that results from iterating over a heterogenous tuple.
@generated function inserthdf5!(hdf5group::HDF5Group, writer::HDF5Writer,
                                remotequeue::TypedBuffer{T}) where T
    n_datavectors = length(T.parameters)

    ex = :()
    for datavector_idx in 1:n_datavectors
        datatype = T.parameters[datavector_idx].parameters[1]

        ex = :($(ex.args...);
               name = names[$datavector_idx];
               if exists(hdf5group, name);
                   dset = d_open(hdf5group, name);
                   set_dims!(dset, (newoffset, ));
               else;
                   dset = d_create(hdf5group, name, $datatype, ((size, ), (-1, )), "chunk",
                                   (maxsize, ));
               end;
               dset[(writer.writeoffset + 1):newoffset] = view(data[$datavector_idx], 1:size);
               close(dset);)
    end

    return quote
        if remotequeue.size > 0
            newoffset = writer.writeoffset + remotequeue.size

            # Hoist from the loop.
            names = writer.collector.names
            data = remotequeue.data
            size = remotequeue.size
            maxsize = remotequeue.maxsize

            $ex

            writer.writeoffset = newoffset
        end
        return nothing
    end
end
