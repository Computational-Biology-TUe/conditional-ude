using CairoMakie, DataFrames, CSV, StatsBase, JLD2
using SimpleChains
using MultivariateStats



inch = 96
pt = 4/3
cm = inch / 2.54
linewidth = 21cm * 0.8

FONTS = (
; regular = "Fira Sans Light",
bold = "Fira Sans SemiBold",
italic = "Fira Sans Italic",
bold_italic = "Fira Sans SemiBold Italic",
)

include("src/suppression_model.jl")

neural_net = neural_network_model(5, 3; input_dims=4)
ude_lsup!(du, u, p, t) = ude_lsup!(du, u, p, t, [0.4, 0.9, 0.3], neural_net)
prob = ODEProblem(ude_lsup!, [10.0, 0.0, 0.0], (0.0, 30.0))

regularization_level = 0.01

test_data, gt_test_param = generate_data(
    [0.5, 2.5, 5.0, 7.5, 10.0, 12.5], [10, 10, 10, 10, 10, 10], range(0, stop=30, length=8); noise_multiplicative=0.1)

# get best model based on validation data
nn_params, group_data, validation_data, validation_data_nonoise, correlations, losses, correlations_valid, losses_valid,  correlations_valid_nonoise, losses_valid_nonoise, gt_sup_param, gt_validation_param, gt_validation_param_nonoise = jldopen("suppression/results/lambda=0.01.jld2", "r") do file
            file["neural_parameters"], file["group_data"], file["validation_data"], file["validation_data_nonoise"],
                file["correlations"], file["losses"], file["correlations_valid"], file["losses_valid"],
                file["correlations_valid_nonoise"], file["losses_valid_nonoise"], file["gt_sup_param"],
                file["gt_validation_param"], file["gt_validation_param_nonoise"]
end

# take the best model based on validation data
best_model_nn = nn_params[argmin(losses_valid)]

# evaluate on the test data
p_init_test = rand(1000)
results = []
objectives = []
for i in axes(test_data, 3)
    result, objective = validate_suppression_model_sigma(p_init_test, prob, test_data[:,:,i], range(0, stop=30, length=8), best_model_nn)
    push!(results, result)
    push!(objectives, objective)
end

betas = [result.ode for result in results]

figure_correlation = let f = Figure(size=(7cm, 7cm), fonts=FONTS, fontsize=8pt)
    ax = Axis(f[1, 1], title="Correlation (ρ = $(round(corspearman(betas, gt_test_param), digits=4)))", xlabel="βᵢ", ylabel="Ground truth parameter")     
    scatter_main = scatter!(ax, betas, gt_test_param, color = Makie.wong_colors()[1], markersize = 8, label = "Test")

    f
end

figure_model_fit = let f = Figure(size=(14cm, 14cm), fonts=FONTS, fontsize=8pt)
    axs = [Axis(f[i, j], xlabel = "Time", ylabel = "Concentration") for i in 1:2, j in 1:2]
    indices = sortperm(objectives)
    
    simulations = simul(
        ComponentArray(
            theta = betas,
            neural = best_model_nn
        ),
        prob,
        test_data,
        range(0, stop=30, length=100)
    )

    quantiles = ["Best", "25%", "50%", "75%"]
    states = ["A", "B", "C"]

    for (n, i) in enumerate(range(1, length(indices), length=4))
        ii = Int(round(i))
        idx = indices[ii]
        println(idx)
        position = ((n-1) % 2 + 1, (n-1) ÷ 2 + 1)
        for j in 1:3
            lines!(axs[position[1], position[2]], range(0.0, stop=30.0; length=100), simulations[j,:,idx], color = (Makie.wong_colors()[j], 1.0), linewidth=2, label="Model $(states[j])")
            scatter!(axs[position[1], position[2]], range(0.0, stop=30.0; length=8), test_data[j,:,idx], color = (Makie.wong_colors()[j], 1.0), markersize = 10, label="Data $(states[j])")
            axs[position[1], position[2]].title = "Subject $(idx) ($(quantiles[n]))"
        end
    end

    linkyaxes!(axs...)
    Legend(f[3,1:2], axs[1,1], orientation=:horizontal, merge=true)
    f
end

save("figures/suppression_correlation.png", figure_correlation, px_per_unit=300/inch)
save("figures/suppression_correlation.svg", figure_correlation, px_per_unit=300/inch)
save("figures/suppression_model_fit.png", figure_model_fit, px_per_unit=300/inch)
save("figures/suppression_model_fit.svg", figure_model_fit, px_per_unit=300/inch)

