include("parameter-estimation.jl")
using Distributions: Chisq

function likelihood_profile(β, neural_network_parameters, model, timepoints, cpeptide_data, lower_bound, upper_bound, sigma; steps=1000)
    
    loss_minimum = loss(β, (model, timepoints, cpeptide_data, neural_network_parameters))
    nll_minimum = 1/(2 * sigma^2) * loss_minimum
    nll_values = Float64[]
    parameter_values = range(lower_bound, stop=upper_bound, length=steps)

    for β in parameter_values
        loss_value = loss(β, (model, timepoints, cpeptide_data, neural_network_parameters))
        push!(nll_values, 1/(2 * sigma^2) * loss_value)
    end

    return nll_values, nll_minimum, parameter_values
end

function likelihood_profile(β, loss_function, args, lower_bound, upper_bound, sigma; steps=1000)
    
    loss_minimum = loss_function(β, args)
    nll_minimum = 1/(2 * sigma^2) * loss_minimum
    nll_values = Float64[]
    parameter_values = range(lower_bound, stop=upper_bound, length=steps)

    for β in parameter_values
        loss_value = loss_function(β, args)
        push!(nll_values, 1/(2 * sigma^2) * loss_value)
    end

    return nll_values, nll_minimum, parameter_values
end

function find_confidence_intervals(loss_values, loss_minimum, parameter_values; target=:cantelli95)

    if target == :cantelli95
        threshold = loss_minimum + 7.16
    elseif target == :cantelli90
        threshold = loss_minimum + 5.24
    elseif target == :raue95
        threshold = loss_minimum + quantile(Chisq(1), 0.95)
    else
        println("Unknown target $(target). Using raue 95%")
        threshold = loss_minimum + quantile(Chisq(1), 0.95)
    end
    # println(threshold)
    # find the indices of the loss values that are below the threshold
    indices = findall(loss_values .<= threshold)

    # find the minimum and maximum parameter values that correspond to the indices
    min_index = minimum(indices)
    max_index = maximum(indices)

    # find the parameter values that correspond to the indices
    min_parameter_value = min_index == 1 ? -Inf : parameter_values[min_index]
    max_parameter_value = max_index == length(parameter_values) ? Inf : parameter_values[max_index]

    return min_parameter_value, max_parameter_value
end

