include("src/suppression_model.jl")

using SimpleChains: init_params
using JLD2

# make the run reproducible
rng = StableRNG(27052023)

nl = 0.1
select_best_n = 25
initial_space = 10_000
# set up the neural network
hidden_layer_dim = 5
cors = DataFrame()
regularization_levels = [0.0, 0.001, 0.01, 0.1, 1.0]

# create neural net
neural_net = neural_network_model(hidden_layer_dim, 3; input_dims=4)
ude_lsup!(du, u, p, t) = ude_lsup!(du, u, p, t, [0.4, 0.9, 0.3], neural_net)
prob = ODEProblem(ude_lsup!, [10.0, 0.0, 0.0], (0.0, 30.0))

group_data, gt_sup_param = generate_data(
    [0.5, 2.5, 5.0, 7.5, 10.0, 12.5], [15, 3, 3, 3, 3, 10], range(0, stop=30, length=8); noise_multiplicative=nl, rng=rng)

validation_data, gt_validation_param = generate_data(
    [0.5, 2.5, 5.0, 7.5, 10.0, 12.5], [5, 5, 5, 5, 5, 5], range(0, stop=30, length=8); noise_multiplicative=nl, rng=rng)

validation_data_nonoise, gt_validation_param_nonoise = generate_data(
    [0.5, 2.5, 5.0, 7.5, 10.0, 12.5], [5, 5, 5, 5, 5, 5], range(0, stop=30, length=8); noise_multiplicative=0.0, rng=rng)

# initialize the parameters
p_init = [ComponentArray(
    theta = randn(rng, size(group_data, 3)),
    neural = init_params(neural_net, rng=rng)
) for i in 1:initial_space]

p_init_valid = [rand(size(validation_data, 3)) for i in 1:initial_space]
# generate data
for λ in regularization_levels
    println("Regularization level: ", λ)

    correlations = fill(NaN, select_best_n)
    losses = fill(NaN, select_best_n)

    correlations_valid = fill(NaN, select_best_n)
    losses_valid = fill(NaN, select_best_n)

    correlations_valid_nonoise = fill(NaN, select_best_n)
    losses_valid_nonoise = fill(NaN, select_best_n)

    nn_params = Vector{Float64}[]

    results, traces = fit_suppression_model(p_init, prob, group_data, range(0, stop=30, length=8), λ; select_best_n=select_best_n)

    for (n,res) in enumerate(results)
        correlations[n] = corspearman(gt_sup_param, res.u.theta)
        losses[n] = res.objective

        res_valid, objective = validate_suppression_model(p_init_valid, prob, validation_data, range(0, stop=30, length=8), res.u.neural)
        correlations_valid[n] = corspearman(gt_validation_param, res_valid)
        losses_valid[n] = objective
    
        res_valid_nonoise, objective_nn = validate_suppression_model(p_init_valid, prob, validation_data_nonoise, range(0, stop=30, length=8), res.u.neural)
        correlations_valid_nonoise[n] = corspearman(gt_validation_param_nonoise, res_valid_nonoise)
        losses_valid_nonoise[n] = objective_nn
        push!(nn_params, Vector{Float64}(res.u.neural)) 
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
