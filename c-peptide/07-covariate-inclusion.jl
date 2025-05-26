# Model fit to the train data and evaluation on the test data
train_model = false
extension = "eps"
inch = 96
pt = 4/3
cm = inch / 2.54
linewidth = 13.07245cm
MANUSCRIPT_FIGURES = true
ECCB_FIGURES = true
FONTS = (
    ; regular = "Fira Sans Light",
    bold = "Fira Sans SemiBold",
    italic = "Fira Sans Italic",
    bold_italic = "Fira Sans SemiBold Italic",
)

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase

rng = StableRNG(232705)

include("../src/parameter-estimation.jl")
include("../src/neural-network.jl")
include("../src/utils.jl")
include("../src/likelihood-profiles.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# define the neural network
network = chain(4, 2, tanh; input_dims=3) # (3 inputs for conditional, age, and glucose)
t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)

# create the models
models_train = [
    CPeptideConditionalCovariateUDEModel(train_data.glucose[i,:], train_data.timepoints, Float64.(train_data.ages[i]), network, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

# train the models or load the trained model neural network parameters
if train_model

    # train on 70%, select on 30%
    indices_train, indices_validation = stratified_split(rng, train_data.types, 0.7)

    optsols_train = train(models_train[indices_train], train_data.timepoints, train_data.cpeptide[indices_train,:], rng)

    neural_network_parameters = [optsol.u.neural[:] for optsol in optsols_train]
    betas = [optsol.u.conditional[:] for optsol in optsols_train]

    objectives = evaluate_model(models_train[indices_validation],
    train_data.timepoints, train_data.cpeptide[indices_validation,:], neural_network_parameters,
    betas)

    best_model_index = argmin(sum(objectives, dims=2)[:])
    best_model = optsols_train[best_model_index]

    # save the models
    jldopen("source_data/cude_covariate_neural_parameters_2.jld2", "w") do file
        file["width"] = 4
        file["depth"] = 2
        file["parameters"] = neural_network_parameters
        file["betas"] = betas
        file["best_model_index"] = best_model_index
    end
else
    neural_network_parameters, betas, best_model_index = try
        jldopen("source_data/cude_covariate_neural_parameters_2.jld2") do file
            file["parameters"], file["betas"], file["best_model_index"]
        end
    catch
        error("Trained weights not found! Please train the model first by setting train_model to true")
    end
end

# obtain the betas for the train data
lb = minimum(betas[best_model_index]) - 0.1*abs(minimum(betas[best_model_index]))
ub = maximum(betas[best_model_index]) + 0.1*abs(maximum(betas[best_model_index]))

optsols = train_with_sigma(models_train, train_data.timepoints, train_data.cpeptide, neural_network_parameters[best_model_index], lbfgs_lower_bound=lb, lbfgs_upper_bound=ub)
betas_train = [optsol.u.ode[1] for optsol in optsols]
sigmas_train = [optsol.u.sigma for optsol in optsols]
objectives_train = ([optsol.objective for optsol in optsols] .- (length(train_data.timepoints)/2) .* log.(sigmas_train.^2)) .* (2 .* sigmas_train.^2)

# obtain the betas for the test data
t2dm = test_data.types .== "T2DM"
models_test = [
    CPeptideConditionalCovariateUDEModel(test_data.glucose[i,:], test_data.timepoints, Float64.(test_data.ages[i]), network, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
]

optsols = train_with_sigma(models_test, test_data.timepoints, test_data.cpeptide, neural_network_parameters[best_model_index], lbfgs_lower_bound=lb, lbfgs_upper_bound=ub)
betas_test = [optsol.u.ode[1] for optsol in optsols]
sigmas_test = [optsol.u.sigma for optsol in optsols]
objectives_test = ([optsol.objective for optsol in optsols] .- (length(test_data.timepoints)/2) .* log.(sigmas_test.^2)) .* (2 .* sigmas_test.^2)
COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
    )

figure_box_betas = let f = Figure()
    ax = Axis(f[1,1], xlabel="group", ylabel="beta")
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        betas_type = exp.(betas_test[type_indices])
        
        boxplot!(ax, fill(i, sum(type_indices)), betas_type, color=COLORS[type], markersize=5)
    end
    f
end

figure_clamp_correlation = let f = Figure(size=(8cm, 6cm), fonts=FONTS, fontsize=8pt)
    corr = corspearman([betas_test; betas_train], [test_data.first_phase; train_data.first_phase])
    ax = Axis(f[1,1], xlabel="βᵢ", ylabel="1ˢᵗ Phase Clamp", title="ρ = $(round(corr, digits=4))")
    
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        betas_type = exp.(betas_test[type_indices])
        clamp_type = test_data.first_phase[type_indices]
        
        scatter!(ax, betas_type, clamp_type, color=COLORS[type], markersize=10, label="Test $type")
    end

    scatter!(ax, exp.(betas_train), train_data.first_phase, color = (:black, 0.2), markersize=8, label="Train Data", marker=:star5)
    Legend(f[1,2], ax, orientation=:vertical)
    
    println(corr)
    f
end

save("figures/revision/supplementary/covariate_correlation.png", figure_clamp_correlation, px_per_unit=4)
save("figures/revision/supplementary/covariate_correlation.eps", figure_clamp_correlation, px_per_unit=4)
save("figures/revision/supplementary/covariate_correlation.svg", figure_clamp_correlation, px_per_unit=4)

function argmedian(x)
    return argmin(abs.(x .- median(x)))
end

# if MANUSCRIPT_FIGURES
    model_fit_figure = let fig
        fig = Figure(size = (12cm, 6cm), fontsize=8pt, fonts=FONTS)
        ga = [GridLayout(fig[1,1], ), GridLayout(fig[1,2], ), GridLayout(fig[1,3], )]
        gb = GridLayout(fig[1,4], nrow=1, ncol=1)
        # do the simulations
        sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
        sols = [Array(solve(model.problem, p=ComponentArray(conditional=[betas_test[i]], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
        
        axs = [Axis(ga[i][1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", title=type) for (i,type) in enumerate(unique(test_data.types))]

        for (i,type) in enumerate(unique(test_data.types))

            type_indices = test_data.types .== type

            c_peptide_data = test_data.cpeptide[type_indices,:]

            sol_idx = findfirst(objectives_test[type_indices] .== median(objectives_test[type_indices]))

            # find the median fit of the type
            sol_type = sols[type_indices][sol_idx]
            loss_values, loss_minimum, parameter_values = likelihood_profile(
                betas_test[type_indices][sol_idx], neural_network_parameters[best_model_index], models_test[type_indices][sol_idx], test_data.timepoints, c_peptide_data[sol_idx,:], betas_test[type_indices][sol_idx]-3, betas_test[type_indices][sol_idx]+5, sigmas_test[type_indices][sol_idx]; steps=10_000
            )

            # find the 95% confidence interval
            min_parameter, max_parameter = find_confidence_intervals(
                loss_values, loss_minimum, parameter_values; target=:raue95
            )
            println(min_parameter, betas_test[type_indices][sol_idx], max_parameter)

            # compute the solution with the lower and upper bounds
            sol_lower = Array(solve(models_test[type_indices][sol_idx].problem, p=ComponentArray(conditional=[min_parameter], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1))

            lines!(axs[i], sol_timepoints, sol_lower[:,1], color=(COLORS[type], 0.5), linewidth=1, label="95% CI", linestyle=:dot)

            if !isinf(max_parameter)
                sol_upper = Array(solve(models_test[type_indices][sol_idx].problem, p=ComponentArray(conditional=[max_parameter], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1))

                lines!(axs[i], sol_timepoints, sol_upper[:,1], color=(COLORS[type], 0.5), linewidth=1, label="95% CI", linestyle=:dot)
            end

            lines!(axs[i], sol_timepoints, sol_type[:,1], color=(COLORS[type], 1), linewidth=1.5, label="Model fit", linestyle=:solid)
            scatter!(axs[i], test_data.timepoints, c_peptide_data[sol_idx,:] , color=(COLORS[type], 1), markersize=5, label="Data")

        end

        linkyaxes!(axs...)

        ax = Axis(gb[1,1], xticks=([0,1,2], ["NGT", "IGT", "T2DM"]), xlabel="Type", ylabel="MSE")

        jitter_width = 0.1

        for (i, type) in enumerate(unique(train_data.types))
            jitter = rand(length(objectives_test)) .* jitter_width .- jitter_width/2
            type_indices = test_data.types .== type
            scatter!(ax, repeat([i-1], length(objectives_test[type_indices])) .+ jitter[type_indices] .- 0.1, objectives_test[type_indices], color=(COLORS[type], 0.8), markersize=3, label=type)
            violin!(ax, repeat([i-1], length(objectives_test[type_indices])) .+ 0.05, objectives_test[type_indices], color=(COLORS[type], 0.8), width=0.75, side=:right, strokewidth=1, datalimits=(0,Inf))
        end

    #  boxplot!(ax, repeat([0], sum(test_data.types .== "NGT")), objectives_test[test_data.types .== "NGT"], color=COLORS["NGT"], width=0.75)
    # boxplot!(ax, repeat([1], sum(test_data.types .== "IGT")),objectives_test[test_data.types .== "IGT"], color=COLORS["IGT"], width=0.75)
    # boxplot!(ax, repeat([2], sum(test_data.types .== "T2DM")),objectives_test[test_data.types .== "T2DM"], color=COLORS["T2DM"], width=0.75)

        Legend(fig[2,1:3], axs[1], orientation=:horizontal, merge=true)

        for (label, layout) in zip(["a", "b", "c", "d"], [ga; gb])
            Label(layout[1, 1, TopLeft()], label,
            fontsize = 12,
            font = :bold,
            padding = (0, 20, 12, 0),
            halign = :right)
        end

    fig
    end

save("figures/revision/supplementary/model_fit_test_covariate_median.png", model_fit_figure, px_per_unit=4)
save("figures/revision/supplementary/model_fit_test_covariate_median.eps", model_fit_figure, px_per_unit=4)
save("figures/revision/supplementary/model_fit_test_covariate_median.svg", model_fit_figure, px_per_unit=4)


#     model_fit_all_test = let fig
#         fig = Figure(size = (1000, 1500))
#         sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
#         sols = [Array(solve(model.problem, p=ComponentArray(ode=betas_test[i], neural=neural_network_parameters), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
        
#         n = length(models_test)
#         n_col = 5
#         locations = [
#             ((i - 1 + n_col) ÷ n_col, (n_col + i - 1) % n_col) for i in 1:n
#         ]
#         grids = [GridLayout(fig[loc[1], loc[2]]) for loc in locations]

#         axs = [Axis(gx[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title="Test Subject $(i) ($(test_data.types[i]))") for (i,gx) in enumerate(grids)]

#         for (i, (sol, ax)) in enumerate(zip(sols, axs))

#             c_peptide_data = test_data.cpeptide[i,:]
#             type = test_data.types[i]
#             lines!(ax, sol_timepoints, sol[:,1], color=(:black, 1), linewidth=2, label="Model fit", linestyle=:dot)
#             scatter!(ax, test_data.timepoints, c_peptide_data , color=(:black, 1), markersize=10, label="Data")

#         end

#         linkyaxes!(axs...)

#         Legend(fig[locations[end][1]+1, 0:4], axs[1], orientation=:horizontal)

#         fig
#     end

#     save("figures/supplementary/model_fit_test_all.$extension", model_fit_all_test, px_per_unit=4)

#     model_fit_train = let fig
#         fig = Figure(size = (linewidth, 6cm), fontsize=8pt)
#         ga = [GridLayout(fig[1,1], ), GridLayout(fig[1,2], ), GridLayout(fig[1,3], )]
#         gb = GridLayout(fig[1,4], nrow=1, ncol=1)

#         # do the simulations
#         sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
#         sols = [Array(solve(model.problem, p=ComponentArray(ode=betas_train[i], neural=neural_network_parameters), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_train)]
        
#         axs = [Axis(ga[i][1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", title=type) for (i,type) in enumerate(unique(train_data.types))]

#         for (i,type) in enumerate(unique(train_data.types))

#             type_indices = train_data.types .== type

#             c_peptide_data = train_data.cpeptide[type_indices,:]

#             sol_idx = argmedian(objectives_train[type_indices])

#             # find the median fit of the type
#             sol_type = sols[type_indices][sol_idx]

#             lines!(axs[i], sol_timepoints, sol_type[:,1], color=(COLORS[type], 1), linewidth=1.5, label="Model fit", linestyle=:dot)
#             scatter!(axs[i], train_data.timepoints, c_peptide_data[sol_idx,:] , color=(COLORS[type], 1), markersize=5, label="Data")

#         end

#         linkyaxes!(axs...)

#         ax = Axis(gb[1,1], xticks=([0,1,2], ["NGT", "IGT", "T2DM"]), xlabel="Type", ylabel="log₁₀ (Error)")
    
#         jitter_width = 0.1

#         for (i, type) in enumerate(unique(train_data.types))
#             jitter = rand(length(objectives_train)) .* jitter_width .- jitter_width/2
#             type_indices = train_data.types .== type
#             scatter!(ax, repeat([i-1], length(objectives_train[type_indices])) .+ jitter[type_indices] .- 0.1, objectives_train[type_indices], color=(COLORS[type], 0.8), markersize=3, label=type)
#             violin!(ax, repeat([i-1], length(objectives_train[type_indices])) .+ 0.05, objectives_train[type_indices], color=(COLORS[type], 0.8), width=0.75, side=:right, strokewidth=1, datalimits=(0,Inf))
#         end
    
#         # boxplot!(ax, repeat([0], sum(train_data.types .== "NGT")), log10.(objectives_train[train_data.types .== "NGT"]), color=COLORS["NGT"], width=0.75)
#         # boxplot!(ax, repeat([1], sum(train_data.types .== "IGT")),log10.(objectives_train[train_data.types .== "IGT"]), color=COLORS["IGT"], width=0.75)
#         # boxplot!(ax, repeat([2], sum(train_data.types .== "T2DM")),log10.(objectives_train[train_data.types .== "T2DM"]), color=COLORS["T2DM"], width=0.75)

#         Legend(fig[2,1:3], axs[1], orientation=:horizontal)

#         for (label, layout) in zip(["a", "b", "c", "d"], [ga; gb])
#             Label(layout[1, 1, TopLeft()], label,
#             fontsize = 12,
#             font = :bold,
#             padding = (0, 20, 12, 0),
#             halign = :right)
#         end

#     fig
#     end

#     save("figures/supplementary/model_fit_train_median.$extension", model_fit_train, px_per_unit=4)


#     # Correlation figure; 1st phase clamp, age, insulin sensitivity 
#     correlation_figure = let fig
#         fig = Figure(size = (linewidth, 6cm), fontsize=8pt)

#         #betas_train = optsols_train[argmin(objectives_train)].u.ode[:]
#         #betas_test = [optsol.u[1] for optsol in optsols_test]

#         correlation_first = corspearman([betas_train; betas_test], [train_data.first_phase; test_data.first_phase])
#         correlation_second = corspearman([betas_train; betas_test], [train_data.ages; test_data.ages])
#         correlation_isi = corspearman([betas_train; betas_test], [train_data.insulin_sensitivity; test_data.insulin_sensitivity])

#         markers=['●', '▴', '■']
#         MAKERS = Dict(
#             "NGT" => '●',
#             "IGT" => '▴',
#             "T2DM" => '■'
#         )
#         MARKERSIZES = Dict(
#             "NGT" => 5,
#             "IGT" => 9,
#             "T2DM" => 5
#         )

#         ga = GridLayout(fig[1,1])
#         gb = GridLayout(fig[1,2])
#         gc = GridLayout(fig[1,3])

#         ax_first = Axis(ga[1,1], xlabel="βᵢ", ylabel= "1ˢᵗ Phase Clamp", title="ρ = $(round(correlation_first, digits=4))")

#         scatter!(ax_first, exp.(betas_train), train_data.first_phase, color = (:black, 0.2), markersize=10, label="Train Data", marker='⋆')
#         for (i,type) in enumerate(unique(test_data.types))
#             type_indices = test_data.types .== type
#             scatter!(ax_first, exp.(betas_test[type_indices]), test_data.first_phase[type_indices], color=COLORS[type], label="Test $type", marker=MAKERS[type], markersize=MARKERSIZES[type])
#         end

#         ax_second = Axis(gb[1,1], xlabel="βᵢ", ylabel= "Age [y]", title="ρ = $(round(correlation_second, digits=4))")

#         scatter!(ax_second, exp.(betas_train), train_data.ages, color = (:black, 0.2), markersize=10, label="Train", marker='⋆')
#         for (i,type) in enumerate(unique(test_data.types))
#             type_indices = test_data.types .== type
#             scatter!(ax_second, exp.(betas_test[type_indices]), test_data.ages[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
#         end

#         ax_di = Axis(gc[1,1], xlabel="βᵢ", ylabel= "Ins. Sens. Index", title="ρ = $(round(correlation_isi, digits=4))")

#         scatter!(ax_di, exp.(betas_train), train_data.insulin_sensitivity, color = (:black, 0.2), markersize=10, label="Train", marker='⋆')
#         for (i,type) in enumerate(unique(test_data.types))
#             type_indices = test_data.types .== type
#             scatter!(ax_di, exp.(betas_test[type_indices]), test_data.insulin_sensitivity[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
#         end

#         Legend(fig[2,1:3], ax_first, orientation=:horizontal)

#         for (label, layout) in zip(["a", "b", "c"], [ga, gb, gc])
#             Label(layout[1, 1, TopLeft()], label,
#             fontsize = 12,
#             font = :bold,
#             padding = (0, 20, 8, 0),
#             halign = :right)
#         end
        
#         fig

#     end

#     save("figures/correlations_cude.$extension", correlation_figure, px_per_unit=4)

#     # supplementary correlation: 2nd phase clamp, body weight, bmi, disposition index
#     additional_correlation_figure = let fig = Figure(size = (linewidth, 6cm), fontsize=8pt)

#     correlation_first = corspearman([betas_train; betas_test], [train_data.second_phase; test_data.second_phase])
#     correlation_second = corspearman([betas_train; betas_test], [train_data.body_weights; test_data.body_weights])
#     correlation_total = corspearman([betas_train; betas_test], [train_data.bmis; test_data.bmis])
#     correlation_isi = corspearman([betas_train; betas_test], [train_data.disposition_indices; test_data.disposition_indices])

#     markers=['●', '▴', '■']
#     MAKERS = Dict(
#         "NGT" => '●',
#         "IGT" => '▴',
#         "T2DM" => '■'
#     )
#     MARKERSIZES = Dict(
#         "NGT" => 5,
#         "IGT" => 9,
#         "T2DM" => 5
#     )

#     ga = GridLayout(fig[1,1])
#     gb = GridLayout(fig[1,2])
#     gc = GridLayout(fig[1,3])
#     gd = GridLayout(fig[1,4])

#     ax_first = Axis(ga[1,1], xlabel="βᵢ", ylabel= "2ⁿᵈ Phase Clamp", title="ρ = $(round(correlation_first, digits=4))")

#     scatter!(ax_first, exp.(betas_train), train_data.second_phase, color = (:black, 0.2), markersize=10, label="Train Data", marker='⋆')
#     for (i,type) in enumerate(unique(test_data.types))
#         type_indices = test_data.types .== type
#         scatter!(ax_first, exp.(betas_test[type_indices]), test_data.second_phase[type_indices], color=COLORS[type], label="Test $type", marker=MAKERS[type], markersize=MARKERSIZES[type])
#     end

#     ax_second = Axis(gb[1,1], xlabel="βᵢ", ylabel= "Body weight [kg]", title="ρ = $(round(correlation_second, digits=4))")

#     scatter!(ax_second, exp.(betas_train), train_data.body_weights, color = (:black, 0.2), markersize=10, label="Train", marker='⋆')
#     for (i,type) in enumerate(unique(test_data.types))
#         type_indices = test_data.types .== type
#         scatter!(ax_second, exp.(betas_test[type_indices]), test_data.body_weights[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
#     end

#     ax_di = Axis(gc[1,1], xlabel="βᵢ", ylabel= "BMI [kg/m²]", title="ρ = $(round(correlation_total, digits=4))")

#     scatter!(ax_di, exp.(betas_train), train_data.bmis, color = (:black, 0.2), markersize=10, label="Train", marker='⋆')
#     for (i,type) in enumerate(unique(test_data.types))
#         type_indices = test_data.types .== type
#         scatter!(ax_di, exp.(betas_test[type_indices]), test_data.bmis[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
#     end


#     ax_isi = Axis(gd[1,1], xlabel="βᵢ", ylabel= "Clamp DI", title="ρ = $(round(correlation_isi, digits=4))")

#         scatter!(ax_isi, exp.(betas_train), train_data.disposition_indices, color = (:black, 0.2), markersize=15, label="Train", marker='⋆')
#         for (i,type) in enumerate(unique(test_data.types))
#             type_indices = test_data.types .== type
#             scatter!(ax_isi, exp.(betas_test[type_indices]), test_data.disposition_indices[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
#         end

#     Legend(fig[2,1:4], ax_first, orientation=:horizontal)

#     for (label, layout) in zip(["a", "b", "c", "d"], [ga, gb, gc, gd])
#         Label(layout[1, 1, TopLeft()], label,
#         fontsize = 12,
#         font = :bold,
#         padding = (0, 20, 8, 0),
#         halign = :right)
#     end
    
#     fig
#     end

#     save("figures/supplementary/correlations_other_cude.$extension", additional_correlation_figure, px_per_unit=4)

#     # sample data for symbolic regression
#     betas_combined = exp.([betas_train; betas_test])
#     glucose_combined = [train_data.glucose; test_data.glucose]

#     beta_range = LinRange(minimum(betas_combined), maximum(betas_combined)*1.1, 30)
#     glucose_range = LinRange(0.0, maximum(glucose_combined .- glucose_combined[:,1]) * 3, 100)

#     colnames = ["Beta", "Glucose", "Production"]
#     data = [ [β, glucose, chain([glucose, β], neural_network_parameters)[1] - chain([0.0, β], neural_network_parameters)[1]] for β in beta_range, glucose in glucose_range]
#     data = hcat(reshape(data, 100*30)...)

#     df = DataFrame(data', colnames)


#     figure_production = let f = Figure(size=(800,600))


#         ga = GridLayout(f[1,1])
#         gb = GridLayout(f[1,2])
#         #df = DataFrame(CSV.File("data/ohashi_production.csv"))
#         beta_values = df[1:30, :Beta]
        
#         ax = Axis(ga[1,1], xlabel="ΔG (mM)", ylabel="Production (nM min⁻¹)", title="Neural Network")
#         for (i, beta) in enumerate(beta_values)
#             df_beta = df[df[!,:Beta] .== beta, :]        
#             lines!(ax, df_beta.Glucose, df_beta.Production, color = i, colorrange=(1,30), colormap=:viridis)
#         end

#         Colorbar(f[1,2], limits=(beta_values[1], beta_values[end]), label="β")    
#         f

#     end

#     #CSV.write("data/ohashi_production.csv", df)
# end
# # ECCB submission
# COLORS = Dict(
#     "NGT" => RGBf(197/255, 205/255, 229/255),
#     "IGT" => RGBf(110/255, 129/255, 192/255),
#     "T2DM" => RGBf(41/255, 55/255, 148/255)
# )

# COLORS_2 = Dict(
#     "NGT" => RGBf(205/255, 234/255, 235/255),
#     "IGT" => RGBf(5/255, 149/255, 154/255),
#     "T2DM" => RGBf(3/255, 75/255, 77/255)
# )

# pagewidth = 21cm
# margin = 0.02 * pagewidth

# textwidth = pagewidth - 2 * margin
# aspect = 1

# data_figure = let f = Figure(
#     size = (0.25textwidth + 0.1textwidth, aspect*0.25textwidth), 
#     fontsize=7pt, fonts = FONTS,
#     backgroundcolor=:transparent)

#     # show the mean data
#     cpeptide = [train_data.cpeptide; test_data.cpeptide]
#     types = [train_data.types; test_data.types]
#     timepoints = train_data.timepoints

#     MARKERS = Dict(
#         "NGT" => '●',
#         "IGT" => '▴',
#         "T2DM" => '■'
#     )
#     MARKERSIZES = Dict(
#         "NGT" => 5pt,
#         "IGT" => 9pt,
#         "T2DM" => 5pt
#     )

#     ax = Axis(f[1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", 
#     backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=false, ygridvisible=false)
#     for (i, type) in enumerate(unique(types))
#         type_indices = types .== type
#         c_peptide_data = cpeptide[type_indices,:]
#         mean_c_peptide = mean(c_peptide_data, dims=1)[:]
#         std_c_peptide = std(c_peptide_data, dims=1)[:]

#         lines!(ax, timepoints, mean_c_peptide, color=(COLORS[type], 1), label="$type", linewidth=2)
#         scatter!(ax, timepoints, mean_c_peptide, color=(COLORS[type], 1), markersize=MARKERSIZES[type], marker=MARKERS[type], label="$type")
#         #band!(ax, timepoints, mean_c_peptide .- std_c_peptide, mean_c_peptide .+ std_c_peptide, color=(COLORS[type], 0.2), label="Std $type")
#     end
#     Legend(f[1,2], ax, orientation=:vertical, merge=true, backgroundcolor=:transparent, framevisible=false, labelfont=:bold, title="C-peptide", titlefont=:bold, titlefontsize=8pt, labelfontsize=8pt)
#     f
# end

# save("figures/eccb/data.$extension", data_figure, px_per_unit=600/inch)

# glucose_figure = let f = Figure(
#     size = (0.25textwidth + 0.1textwidth, aspect*0.25textwidth), 
#     fontsize=7pt, fonts = FONTS,
#     backgroundcolor=:transparent)

#     # show the mean data
#     glucose = [train_data.glucose; test_data.glucose]
#     types = [train_data.types; test_data.types]
#     timepoints = train_data.timepoints

#     MARKERS = Dict(
#         "NGT" => '●',
#         "IGT" => '▴',
#         "T2DM" => '■'
#     )
#     MARKERSIZES = Dict(
#         "NGT" => 5pt,
#         "IGT" => 9pt,
#         "T2DM" => 5pt
#     )

#     ax = Axis(f[1,1], xlabel="Time [min]", ylabel="Gₚₗ [mmol L⁻¹]", 
#     backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=false, ygridvisible=false)
#     for (i, type) in enumerate(unique(types))
#         type_indices = types .== type
#         glucose_data = glucose[type_indices,:]
#         mean_glucose = mean(glucose_data, dims=1)[:]
#         std_glucose = std(glucose_data, dims=1)[:]

#         lines!(ax, timepoints, mean_glucose, color=(COLORS[type], 1), label="$type", linewidth=2)
#         scatter!(ax, timepoints, mean_glucose, color=(COLORS[type], 1), markersize=MARKERSIZES[type], marker=MARKERS[type], label="$type")
#         errorbars!(ax, timepoints, mean_glucose, std_glucose, color=(COLORS[type], 1), whiskerwidth=6, label="$type")
#         #band!(ax, timepoints, mean_c_peptide .- std_c_peptide, mean_c_peptide .+ std_c_peptide, color=(COLORS[type], 0.2), label="Std $type")
#     end
#     Legend(f[1,2], ax, orientation=:vertical, merge=true, backgroundcolor=:transparent, framevisible=false, labelfont=:bold, title="C-peptide", titlefont=:bold, titlefontsize=8pt, labelfontsize=8pt)
#     f
# end

# save("figures/eccb/glucose.svg", glucose_figure, px_per_unit=600/inch)

# figure_production = let f = Figure(size = (0.25textwidth, 0.25aspect*textwidth), 
#     fontsize=7pt, fonts = FONTS,
#     backgroundcolor=:transparent)

#     # sample data for symbolic regression
#     betas_combined = exp.([betas_train; betas_test])
#     glucose_combined = [train_data.glucose; test_data.glucose]

#     beta_range = LinRange(minimum(betas_combined), maximum(betas_combined)*1.1, 3)
#     glucose_range = LinRange(0.0, maximum(glucose_combined .- glucose_combined[:,1]) * 3, 100)

#     colnames = ["Beta", "Glucose", "Production"]
#     data = [ [β, glucose, chain([glucose, β], neural_network_parameters)[1] - chain([0.0, β], neural_network_parameters)[1]] for β in beta_range, glucose in glucose_range]
#     data = hcat(reshape(data, 100*3)...)

#     df = DataFrame(data', colnames)

#     #df = DataFrame(CSV.File("data/ohashi_production.csv"))
#     beta_values = df[1:3, :Beta]
#     types = ["NGT", "IGT", "T2DM"]
    
#     ax = Axis(f[1,1], xlabel="ΔG (mM)", ylabel="Production (nM min⁻¹)", 
#     backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=false, ygridvisible=false)
#     for (i, beta) in enumerate(beta_values)
#         df_beta = df[df[!,:Beta] .== beta, :]        
#         lines!(ax, df_beta.Glucose, df_beta.Production, color = COLORS[types[i]], linewidth=2)
#     end

#     f

# end

# save("figures/eccb/production.svg", figure_production, px_per_unit=600/inch)

# model_fit_figure = let fig = Figure(size = (0.25textwidth + 0.1*textwidth, 0.25aspect*textwidth), 
#     fontsize=7pt, fonts = FONTS,
#     backgroundcolor=:transparent)
    
#     # do the simulations
#     sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
#     sols = [Array(solve(model.problem, p=ComponentArray(ode=[betas_test[i]], neural=neural_network_parameters), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
    
#     ax = Axis(fig[1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", 
#     backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=false, ygridvisible=false)
#     for (i,type) in enumerate(unique(test_data.types))

#         type_indices = test_data.types .== type

#         c_peptide_data = test_data.cpeptide[type_indices,:]

#         sol_idx = findfirst(objectives_test[type_indices] .== median(objectives_test[type_indices]))

#         # find the median fit of the type
#         sol_type = sols[type_indices][sol_idx]

#         MARKERS = Dict(
#             "NGT" => '●',
#             "IGT" => '▴',
#             "T2DM" => '■'
#         )
#         MARKERSIZES = Dict(
#             "NGT" => 5pt,
#             "IGT" => 9pt,
#             "T2DM" => 5pt
#         )

#         lines!(ax, sol_timepoints, sol_type[:,1], color=(COLORS[type], 1), linewidth=1.5, label="Model fit", linestyle=:solid)
#         scatter!(ax, test_data.timepoints, c_peptide_data[sol_idx,:] , color=(COLORS[type], 1), markersize=MARKERSIZES[type], marker=MARKERS[type], label="$type")

#     end
#     Legend(fig[1,2], ax, orientation=:vertical, merge=true, backgroundcolor=:transparent, framevisible=false, labelfont=:bold, title="C-peptide", titlefont=:bold, titlefontsize=8pt, labelfontsize=8pt)

# fig
# end

# save("figures/eccb/model_fit.$extension", model_fit_figure, px_per_unit=600/inch)

# correlation_figure = let fig = Figure(size = (0.3textwidth + 0.1*textwidth, 0.25aspect*textwidth), 
#     fontsize=7pt, fonts = FONTS,
#     backgroundcolor=:transparent)

#     #betas_train = optsols_train[argmin(objectives_train)].u.ode[:]
#     #betas_test = [optsol.u[1] for optsol in optsols_test]

#     correlation_first = corspearman([betas_train; betas_test], [train_data.first_phase; test_data.first_phase])
    
#     markers=['●', '▴', '■']
#     MAKERS = Dict(
#         "NGT" => '●',
#         "IGT" => '▴',
#         "T2DM" => '■'
#     )
#     MARKERSIZES = Dict(
#         "NGT" => 5,
#         "IGT" => 9,
#         "T2DM" => 5
#     )


#     ax_first = Axis(fig[1,1], xlabel="βᵢ", ylabel= "First Phase Clamp", title="ρ = $(round(correlation_first, digits=4))", backgroundcolor=:transparent, xgridvisible=false, ygridvisible=false, xlabelfont=:bold, ylabelfont=:bold)

#     scatter!(ax_first, exp.(betas_train), train_data.first_phase, color = (:black, 0.4), markersize=12, label="Train Data", marker='×')
#     for (i,type) in enumerate(unique(test_data.types))
#         type_indices = test_data.types .== type
#         scatter!(ax_first, exp.(betas_test[type_indices]), test_data.first_phase[type_indices], color=COLORS[type], label="Test $type", marker=MAKERS[type], markersize=MARKERSIZES[type])
#     end

#     Legend(fig[1,2], ax_first, orientation=:vertical, merge=true, backgroundcolor=:transparent, framevisible=false, labelfont=:bold, title="C-peptide", titlefont=:bold, titlefontsize=8pt, labelfontsize=8pt)
    
#     fig

# end

# save("figures/eccb/correlation.$extension", correlation_figure, px_per_unit=600/inch)