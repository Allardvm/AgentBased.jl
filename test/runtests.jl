# TODO: Add tests that include conditions and iters etc. Make sure to test keywords for all types.

using AgentBased
using Base.Test
using HDF5

@testset "AgentBased.jl" begin


#==================================================================================================
TEST REPORTERS
==================================================================================================#

    floatreturner(x::Float64) = x

    r1 = Reporter("int", Int64, () -> 5)
    r2 = Reporter("float", Float64, floatreturner)

    @testset "Reporter call type stability" begin
        check_call(reporter::Reporter, args...) = reporter.call(args...)

        @test Base.return_types(check_call, Tuple{typeof(r1)}) == [Int64]
        @test Base.return_types(check_call, Tuple{typeof(r2), Float64}) == [Float64]
    end


#==================================================================================================
TEST COLLECTORS
==================================================================================================#

    c1 = Collector(r1, r2)

    @testset "Collector calls type stability" begin
        check_call_1(coll::Collector, args...) = coll.calls[1](args...)
        check_call_2(coll::Collector, args...) = coll.calls[2](args...)
        check_call_both(coll::Collector, args...) = (coll.calls[1](), coll.calls[2](args...))

        @test Base.return_types(check_call_1, Tuple{typeof(c1)}) == [Int64]
        @test Base.return_types(check_call_2, Tuple{typeof(c1), Float64}) == [Float64]
        @test Base.return_types(check_call_both, Tuple{typeof(c1), Float64}) ==
            [Tuple{Int64,Float64}]
    end


#==================================================================================================
TEST TYPEDBUFFERS
==================================================================================================#

    t1 = AgentBased.TypedBuffer(c1, 3)

    @testset "TypedBuffer internals" begin
        @test t1.size == 0
        @test t1.maxsize == 3
        @test length(t1.data) == 2
        @test all(length(t1.data[i]) == 3 for i in length(t1.data))
    end

    @testset "TypedBuffer data type stability" begin
        function check_data(tb::AgentBased.TypedBuffer)
            foo = tb.data[1][1]
            bar = tb.data[2][1]
            return foo, bar
        end

        @test Base.return_types(check_data, Tuple{typeof(t1)}) == [Tuple{Int64,Float64}]
        @test Base.return_types(AgentBased.TypedBuffer, Tuple{typeof(c1), Int}) ==
            [AgentBased.TypedBuffer{Tuple{Vector{Int64},Vector{Float64}}}]
    end


#==================================================================================================
TEST HDF5WRITER
==================================================================================================#

    h1 = HDF5Writer("test.h5", "test", c1)
    h1 = HDF5Writer("test.h5", "test", c1; chunksz = 3, channels_per_worker = 1, writemode = "w")

    @testset "HDF5Writer internals" begin
        coll_type = typeof(h1).parameters[1]
        buff_type = typeof(h1).parameters[2]
        @test all(buff_type.parameters[i].parameters[1] == coll_type.parameters[1].parameters[i]
                for i in 1:length(buff_type.parameters))
    end

    @testset "HDF5Writer data type stability" begin
        function check_data(writer::HDF5Writer)
            foo = writer.localbuffer.data[1][1]
            bar = writer.localbuffer.data[2][1]
            return foo, bar
        end

        @test Base.return_types(check_data, Tuple{typeof(h1)}) == [Tuple{Int64,Float64}]
    end

    @testset "HDF5Writer calls type stability" begin
        check_call_1(writer::HDF5Writer, args...) = writer.collector.calls[1](args...)
        check_call_2(writer::HDF5Writer, args...) = writer.collector.calls[2](args...)
        check_call_both(writer::HDF5Writer, args...) = (writer.collector.calls[1](),
                                                        writer.collector.calls[2](args...))

        @test Base.return_types(check_call_1, Tuple{typeof(h1)}) == [Int64]
        @test Base.return_types(check_call_2, Tuple{typeof(h1), Float64}) == [Float64]
        @test Base.return_types(check_call_both, Tuple{typeof(h1), Float64}) ==
            [Tuple{Int64,Float64}]
    end


#==================================================================================================
TEST COLLECTDATA, ENSUREROOM, AND FLUSHDATA
==================================================================================================#

    @testset "HDF5Writer data collection" begin
        test_iter() = 1:4
        test_r1 = Reporter("iter", Int64, (i) -> i)
        test_r2 = Reporter("float", Float64, (i) -> convert(Float64, i))
        test_c = Collector(test_r1, test_r2; iter = test_iter)
        test_w = HDF5Writer("test.h5", "test", test_c; chunksz = 3, channels_per_worker = 1)

        @sync begin
            @async begin
                collectdata(test_w)
                AgentBased.flushdata(test_w)
            end
            @async begin
                take!(test_w.remotequeue) ==
                    AgentBased.TypedBuffer{Tuple{Vector{Int64},Vector{Float64}}}(
                        3, 3, ([1, 2, 3], [1.0, 2.0, 3.0]))
                take!(test_w.remotequeue) ==
                    AgentBased.TypedBuffer{Tuple{Vector{Int64},Vector{Float64}}}(
                        1, 3, ([4, 2, 3], [4.0, 2.0, 3.0]))
            end
        end
    end


#==================================================================================================
TEST RUNBATCH
==================================================================================================#


#==================================================================================================
TEST INSERTHDF5
==================================================================================================#

end