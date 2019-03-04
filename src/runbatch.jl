using Dates, Distributed


"""
    runbatch(sim, writer, expqueue)

Run the simulation in parallel, using the writer to collect data.

Calls the simulation function `sim` in parallel as `f(expqueue[idx], idx, writer)` for each entry
`expqueue[idx]` in `expqueue`. Individual runs of the simulation are automatically distributed over
all available workers by the master process. The master process uses the `writer` to collect and
store the results.

# Arguments
* `sim::Function`: the simulation function.
* `writer::Writer`: the Writer to use to store the data.
* `expqueue::Vector`: a Vector that the arguments for each run of the simulation.
"""
function runbatch(sim::Function, writer::Writer, expqueue::Vector)
    sink = open_sink(writer)
    try
        next_expidx = 0
        getnext_expidx() = (next_expidx += 1)

        overview = zeros(Int, nworkers())

        function updateoverview(pid, expidx)
            overview[ifelse(nprocs() == 1, 1, pid - 1)] = expidx
        end

        @sync begin
            @async begin
                while true
                    try
                        flushedbuffer = take!(writer.remotequeue)
                        insert!(sink, writer, flushedbuffer)
                    catch err
                        if isa(err, InvalidStateException) && err.state == :closed
                            break # Queue is empty and was closed because workers are done.
                        else
                            rethrow(err)
                        end
                    end
                end
            end
            @sync begin # Prevents race conditions when a worker's async task throws an error.
                println(now(), ": Queued ", length(expqueue), " experiments on ", nworkers(), " cores")
                for pid in workers()
                    @async begin
                        last_expidx = length(expqueue)
                        cur_expidx = getnext_expidx()
                        while cur_expidx <= last_expidx
                            updateoverview(pid, cur_expidx)
                            print("$(now()): Computing experiments $overview...\n")
                            remotecall_fetch(sim, pid, expqueue[cur_expidx], cur_expidx, writer)
                            cur_expidx = getnext_expidx()
                        end
                        updateoverview(pid, 0)
                        print("$(now()): Computing experiments $overview...\n")
                    end
                end
            end
            close(writer.remotequeue)
        end
        println(now(), ": All experiments have been completed")
    finally
        close_sink(sink, writer)
    end
    return get_return(writer)
end
