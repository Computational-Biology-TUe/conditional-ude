using CairoMakie, DataFrames, CSV, StatsBase

include("src/suppression_model.jl")

neural_net = neural_network_model(hidden_layer_dim, 3; input_dims=4)
ude_lsup!(du, u, p, t) = ude_lsup!(du, u, p, t, [0.4, 0.9, 0.3], neural_net)
prob = ODEProblem(ude_lsup!, [10.0, 0.0, 0.0], (0.0, 30.0))

regularization_levels = [0.0; exp10.([-3, -2, -1, 0, 1, 2, 3])]


figure_correlations = let f = Figure()
    ax = Axis(f[1, 1], title="Correlations", xlabel="Regularization level", ylabel="Correlation")     
    for (i,λ) in enumerate(regularization_levels)

        nn_params, group_data, validation_data, validation_data_nonoise, correlations, losses, correlations_valid, losses_valid,  correlations_valid_nonoise, losses_valid_nonoise, gt_sup_param, gt_validation_param, gt_validation_param_nonoise = jldopen("suppression/results/lambda=$(λ).jld2", "r") do file
            file["neural_parameters"], file["group_data"], file["validation_data"], file["validation_data_nonoise"],
                file["correlations"], file["losses"], file["correlations_valid"], file["losses_valid"],
                file["correlations_valid_nonoise"], file["losses_valid_nonoise"], file["gt_sup_param"],
                file["gt_validation_param"], file["gt_validation_param_nonoise"]
        end

        # plot correlations
        scatter!(ax, repeat([i], length(correlations)) .+ 0.2 .* randn(length(correlations)), abs.(correlations), color = Makie.wong_colors()[1], markersize = 9, label = "Train")
        scatter!(ax, repeat([i+0.3], length(correlations_valid)) .+ 0.2 .* randn(length(correlations_valid)), abs.(correlations_valid), color = Makie.wong_colors()[2], markersize = 9, label = "Validation")
    end
    f
end