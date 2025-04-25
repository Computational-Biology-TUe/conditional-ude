include("src/suppression_model.jl")

# TODO: Replace this with a proper experiment/example!
using SimpleChains: init_params
using JLD2

rng = StableRNG(27052023)

nl = 0.1
n_reps = 50
hidden_layer_dim = 3
cors = DataFrame()
regularization_levels = [0.0; exp10.([-2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, 0, 1, 2])]

# create neural net
neural_net = neural_network_model(hidden_layer_dim, 3; input_dims=4)
ude_lsup!(du, u, p, t) = ude_lsup!(du, u, p, t, [0.4, 0.9, 0.3], neural_net)
prob = ODEProblem(ude_lsup!, [10.0, 0.0, 0.0], (0.0, 30.0))

# generate data
for λ in regularization_levels
    println("Regularization level: ", λ)
    group_data, gt_sup_param = generate_data(
        [0.5, 2.5, 5.0, 7.5, 10.0, 12.5], [10, 10, 10, 10, 10, 10], range(0, stop=30, length=8); noise_multiplicative=nl)

    validation_data, gt_validation_param = generate_data(
        [0.5, 2.5, 5.0, 7.5, 10.0, 12.5], [5, 5, 5, 5, 5, 5], range(0, stop=30, length=8); noise_multiplicative=nl)
    
    validation_data_nonoise, gt_validation_param_nonoise = generate_data(
        [0.5, 2.5, 5.0, 7.5, 10.0, 12.5], [5, 5, 5, 5, 5, 5], range(0, stop=30, length=8); noise_multiplicative=0.0)

    correlations = fill(NaN, n_reps)
    losses = fill(NaN, n_reps)

    correlations_valid = fill(NaN, n_reps)
    losses_valid = fill(NaN, n_reps)

    correlations_valid_nonoise = fill(NaN, n_reps)
    losses_valid_nonoise = fill(NaN, n_reps)

    nn_params = Vector{Float64}[]

    for i in 1:n_reps
        p_init = [ComponentArray(
            theta = randn(size(group_data, 3)),
            neural = init_params(neural_net, rng=rng)
        ) for i in 1:10_000]

        p_init_valid = [rand(size(validation_data, 3)) for i in 1:10_000]

        try 
            res = fit_suppression_model(p_init, prob, group_data, range(0, stop=30, length=8), λ)
            correlations[i] = corspearman(gt_sup_param, res.u.theta)
            losses[i] = res.objective

            res_valid = validate_suppression_model(p_init_valid, prob, validation_data, range(0, stop=30, length=8), res.u.neural)
            correlations_valid[i] = corspearman(gt_validation_param, res_valid.u)
            losses_valid[i] = res_valid.objective
            
            res_valid_nonoise = validate_suppression_model(p_init_valid, prob, validation_data_nonoise, range(0, stop=30, length=8), res.u.neural)
            correlations_valid_nonoise[i] = corspearman(gt_validation_param_nonoise, res_valid_nonoise.u)
            losses_valid_nonoise[i] = res_valid_nonoise.objective
            push!(nn_params, Vector{Float64}(res.u.neural))  
        catch
            println("Optimization failed")
            push!(nn_params, [NaN])
        end
    end

    cors[!,Symbol("c"*string(nl))] = correlations
    cors[!,Symbol("l"*string(nl))] = losses
    cors[!,Symbol("vc"*string(nl))] = correlations_valid
    cors[!,Symbol("vl"*string(nl))] = losses_valid
    cors[!,Symbol("vcn"*string(nl))] = correlations_valid_nonoise
    cors[!,Symbol("vln"*string(nl))] = losses_valid_nonoise

    jldopen("suppression/results/lambda=$(λ).jld2", "w") do file
        file["neural_parameters"] = nn_params
        file["group_data"] = group_data
        file["validation_data"] = validation_data
        file["validation_data_nonoise"] = validation_data_nonoise
        file["correlations"] = correlations
        file["losses"] = losses
        file["correlations_valid"] = correlations_valid
        file["losses_valid"] = losses_valid
        file["correlations_valid_nonoise"] = correlations_valid_nonoise
        file["losses_valid_nonoise"] = losses_valid_nonoise
        file["gt_sup_param"] = gt_sup_param
        file["gt_validation_param"] = gt_validation_param
        file["gt_validation_param_nonoise"] = gt_validation_param_nonoise
        file["λ"] = λ
    end


    CSV.write("suppression/results/summary_lambda=$(λ).csv", cors)
end
