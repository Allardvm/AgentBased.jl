function always_true(_...)
    return true
end


function always_false(_...)
    return false
end


function always_nothing(_...)
    return nothing
end


function passthrough(args...)
    return (args, )
end
