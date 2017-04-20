immutable Reporter{T,F<:Function}
    name::String
    call::F
end


"""
    Reporter(name, datatype, call)

Return a `Reporter` that specifies a piece of data that should be collected.

# Arguments
* `name::String`: specifies the name of the piece of data.
* `datatype::DataType`: specifies the datatype of the piece of data.
* `call::Function`: specifies a Function that returns the corresponding value when called as
    `f(args...)`.
"""
function Reporter(name::String, datatype::DataType, call::F) where F<:Function
    return Reporter{datatype,F}(name, call)
end


immutable Collector{T,C,F1<:Function,F2<:Function,F3<:Function,F4<:Function}
    names::Vector{String} # Holds the reporter names.
    calls::C # Holds the reporter calls.
    iter::F1 # Returns an iterable when called as iter(model, agents, exp).
    condition::F2 # Specifies the conditions under which calls to update the
                  # collector should proceed. Continues if and only if `condition(model,
                  # agents, exp)` returns `true`.
    prepare::F3 # Specifies algorithms to run right before the collector collects data.
                # Called as `prepare(model, agents, exp)`.
    finish::F4 # Specifies algorithms that run right after the collector collects data.
               # Called as `finish(model, agents, exp)`.
end


"""
    Collector(reporters...; iter = passthrough, condition = always_true, prepare = always_nothing,
              finish = always_nothing)

Return a `Collector` that collects data for the reporters.

# Arguments
* `reporters::Reporter...`: one or more `Reporter`s that specify data that should be collected.
* `iter::Function`: specifies a Function that provides an iterable that provides the arguments
    with which to call the reporters. The function is called as `iter(args...)`. Defaults to a
    function that passes the arguments directly.
* `condition::Function`: specifies the conditions under which the collector should collect data.
    Calls to collect data with the collector will only proceed when `condition(args...)` returns
    `true`. Defaults to a function that always returns `true`.
* `prepare::Function`: specifies a function to run right before the collector collects data, but
    after the check that data should be collected. The function is called as `prepare(args...)`.
    Defaults to an empty function.
* `finish::Function`: specifies a function to run right after the collector collected data, but
    only if data was actually collected. The function is called as `finish(args...)`. Defaults to
    an empty function.
"""
function Collector(reporters::Reporter...;
                   iter::F1 = passthrough, condition::F2 = always_true,
                   prepare::F3 = always_nothing, finish::F4 = always_nothing) where {F1,F2,F3,F4}
    T = Tuple{(typeof(reporter).parameters[1] for reporter in reporters)...}
    C = Tuple{(typeof(reporter).parameters[2] for reporter in reporters)...}
    names = [reporter.name for reporter in reporters]
    calls = ((reporter.call for reporter in reporters)...)
    return Collector{T,C,F1,F2,F3,F4}(names, calls, iter, condition, prepare, finish)
end
