# Model fit to the train data and evaluation on the test data
train_model = false
extension = "png"

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase

rng = StableRNG(232705)

include("src/c-peptide-ude-models.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# define the neural network
chain = neural_network_model(2, 6)
t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)

# create the models
models_train = [
    CPeptideCUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

# train the models or load the trained model neural network parameters
if train_model

    # train on 70%, select on 30%
    indices_train, indices_validation = stratified_split(rng, train_data.types, 0.7)

    optsols_train = train(models_train[indices_train], train_data.timepoints, train_data.cpeptide[indices_train,:], rng)

    neural_network_parameters = [optsol.u.neural[:] for optsol in optsols_train]
    betas = [optsol.u.ode[:] for optsol in optsols_train]

    best_model_index = select_model(models_train[indices_validation],
    train_data.timepoints, train_data.cpeptide[indices_validation,:], neural_network_parameters,
    betas) 

    best_model = optsols_train[best_model_index]

    # save the neural network parameters
    neural_network_parameters = best_model.u.neural[:]

    # save the best model
    jldopen("source_data/cude_neural_parameters.jld2", "w") do file
        file["width"] = 6
        file["depth"] = 2
        file["parameters"] = neural_network_parameters
        file["betas"] = betas
    end
else

    neural_network_parameters, betas = try
        jldopen("source_data/cude_neural_parameters.jld2") do file
            file["parameters"], file["betas"]
        end
    catch
        error("Trained weights not found! Please train the model first by setting train_model to true")
    end
end
# obtain the betas for the train data
lb = minimum(betas[best_model_index]) - 0.1*abs(minimum(betas[best_model_index]))
ub = maximum(betas[best_model_index]) + 0.1*abs(maximum(betas[best_model_index]))

optsols = train(models_train, train_data.timepoints, train_data.cpeptide, neural_network_parameters, lbfgs_lower_bound=lb, lbfgs_upper_bound=ub)
betas_train = [optsol.u[1] for optsol in optsols]
objectives_train = [optsol.objective for optsol in optsols]

# obtain the betas for the test data
t2dm = test_data.types .== "T2DM"
models_test = [
    CPeptideCUDEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
]

optsols = train(models_test, test_data.timepoints, test_data.cpeptide, neural_network_parameters, lbfgs_lower_bound=lb, lbfgs_upper_bound=ub)
betas_test = [optsol.u[1] for optsol in optsols]
objectives_test = [optsol.objective for optsol in optsols]

model_fit_figure = let fig
    fig = Figure(size = (775, 300))
    ga = GridLayout(fig[1,1:3], )
    gb = GridLayout(fig[1,4], nrow=1, ncol=1)
    # do the simulations
    sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
    sols = [Array(solve(model.problem, p=ComponentArray(ode=[betas_test[i]], neural=neural_network_parameters), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
    
    axs = [Axis(ga[1,i], xlabel="Time [min]", ylabel="C-peptide [nM]", title=type) for (i,type) in enumerate(unique(test_data.types))]

    for (i,type) in enumerate(unique(test_data.types))

        type_indices = test_data.types .== type

        c_peptide_data = test_data.cpeptide[type_indices,:]

        sol_idx = findfirst(objectives_test[type_indices] .== median(objectives_test[type_indices]))

        # find the median fit of the type
        sol_type = sols[type_indices][sol_idx]

        lines!(axs[i], sol_timepoints, sol_type[:,1], color=(:black, 1), linewidth=2, label="Model fit", linestyle=:dot)
        scatter!(axs[i], test_data.timepoints, c_peptide_data[sol_idx,:] , color=(:black, 1), markersize=10, label="Data")

    end

    ax = Axis(gb[1,1], xticks=([0,1,2], ["NGT", "IGT", "T2DM"]), xlabel="Type", ylabel="log₁₀ (Error)")
    boxplot!(ax, repeat([0], sum(test_data.types .== "NGT")), log10.(objectives_test[test_data.types .== "NGT"]), color=COLORS["NGT"], width=0.75)
    boxplot!(ax, repeat([1], sum(test_data.types .== "IGT")),log10.(objectives_test[test_data.types .== "IGT"]), color=COLORS["IGT"], width=0.75)
    boxplot!(ax, repeat([2], sum(test_data.types .== "T2DM")),log10.(objectives_test[test_data.types .== "T2DM"]), color=COLORS["T2DM"], width=0.75)

    Legend(fig[2,1:3], axs[1], orientation=:horizontal)

    for (label, layout) in zip(["a", "b"], [ga, gb])
        Label(layout[1, 1, TopLeft()], label,
        fontsize = 18,
        font = :bold,
        padding = (0, 20, 8, 0),
        halign = :right)
    end

fig
end

save("figures/model_fit_test_median.$extension", model_fit_figure, px_per_unit=4)

model_fit_all_test = let fig
    fig = Figure(size = (1000, 1500))
    sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
    sols = [Array(solve(model.problem, p=ComponentArray(ode=betas_test[i], neural=neural_network_parameters), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
    
    n = length(models_test)
    n_col = 5
    locations = [
        ((i - 1 + n_col) ÷ n_col, (n_col + i - 1) % n_col) for i in 1:n
    ]
    grids = [GridLayout(fig[loc[1], loc[2]]) for loc in locations]

    axs = [Axis(gx[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title="Test Subject $(i) ($(test_data.types[i]))") for (i,gx) in enumerate(grids)]

    for (i, (sol, ax)) in enumerate(zip(sols, axs))

        c_peptide_data = test_data.cpeptide[i,:]
        type = test_data.types[i]
        lines!(ax, sol_timepoints, sol[:,1], color=(:black, 1), linewidth=2, label="Model fit", linestyle=:dot)
        scatter!(ax, test_data.timepoints, c_peptide_data , color=(:black, 1), markersize=10, label="Data")

    end

    Legend(fig[locations[end][1]+1, 0:4], axs[1], orientation=:horizontal)

    fig
end

save("figures/supplementary/model_fit_test_all.$extension", model_fit_all_test, px_per_unit=4)

function argmedian(x)
    return argmin(abs.(x .- median(x)))
end

model_fit_train = let fig
    fig = Figure(size = (775, 300))
    ga = GridLayout(fig[1,1:3], )
    gb = GridLayout(fig[1,4], nrow=1, ncol=1)

    # do the simulations
    sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
    sols = [Array(solve(model.problem, p=ComponentArray(ode=betas_train[i], neural=neural_network_parameters), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_train)]
    
    axs = [Axis(ga[1,i], xlabel="Time [min]", ylabel="C-peptide [nM]", title=type) for (i,type) in enumerate(unique(train_data.types))]

    for (i,type) in enumerate(unique(train_data.types))

        type_indices = train_data.types .== type

        c_peptide_data = train_data.cpeptide[type_indices,:]

        sol_idx = argmedian(objectives_train[type_indices])

        # find the median fit of the type
        sol_type = sols[type_indices][sol_idx]

        lines!(axs[i], sol_timepoints, sol_type[:,1], color=(:black, 1), linewidth=2, label="Model fit", linestyle=:dot)
        scatter!(axs[i], train_data.timepoints, c_peptide_data[sol_idx,:] , color=(:black, 1), markersize=10, label="Data")

    end

    ax = Axis(gb[1,1], xticks=([0,1,2], ["NGT", "IGT", "T2DM"]), xlabel="Type", ylabel="log₁₀ (Error)")
    boxplot!(ax, repeat([0], sum(train_data.types .== "NGT")), log10.(objectives_train[train_data.types .== "NGT"]), color=COLORS["NGT"], width=0.75)
    boxplot!(ax, repeat([1], sum(train_data.types .== "IGT")),log10.(objectives_train[train_data.types .== "IGT"]), color=COLORS["IGT"], width=0.75)
    boxplot!(ax, repeat([2], sum(train_data.types .== "T2DM")),log10.(objectives_train[train_data.types .== "T2DM"]), color=COLORS["T2DM"], width=0.75)

    Legend(fig[2,1:3], axs[1], orientation=:horizontal)

    for (label, layout) in zip(["a", "b"], [ga, gb])
        Label(layout[1, 1, TopLeft()], label,
        fontsize = 18,
        font = :bold,
        padding = (0, 20, 8, 0),
        halign = :right)
    end

fig
end

save("figures/supplementary/model_fit_train_median.$extension", model_fit_train, px_per_unit=4)


# Correlation figure; 1st phase clamp, age, insulin sensitivity 
correlation_figure = let fig
    fig = Figure(size=(700,300))

    #betas_train = optsols_train[argmin(objectives_train)].u.ode[:]
    #betas_test = [optsol.u[1] for optsol in optsols_test]

    correlation_first = corspearman([betas_train; betas_test], [train_data.first_phase; test_data.first_phase])
    correlation_second = corspearman([betas_train; betas_test], [train_data.ages; test_data.ages])
    correlation_isi = corspearman([betas_train; betas_test], [train_data.insulin_sensitivity; test_data.insulin_sensitivity])

    markers=['●', '▴', '■']
    MAKERS = Dict(
        "NGT" => '●',
        "IGT" => '▴',
        "T2DM" => '■'
    )
    MARKERSIZES = Dict(
        "NGT" => 10,
        "IGT" => 18,
        "T2DM" => 10
    )

    ga = GridLayout(fig[1,1])
    gb = GridLayout(fig[1,2])
    gc = GridLayout(fig[1,3])

    ax_first = Axis(ga[1,1], xlabel="βᵢ", ylabel= "1ˢᵗ Phase Clamp [μIU mL⁻¹ min]", title="ρ = $(round(correlation_first, digits=4))")

    scatter!(ax_first, exp.(betas_train), train_data.first_phase, color = (:black, 0.2), markersize=15, label="Train Data", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        scatter!(ax_first, exp.(betas_test[type_indices]), test_data.first_phase[type_indices], color=COLORS[type], label="Test $type", marker=MAKERS[type], markersize=MARKERSIZES[type])
    end

    ax_second = Axis(gb[1,1], xlabel="βᵢ", ylabel= "Age [y]", title="ρ = $(round(correlation_second, digits=4))")

    scatter!(ax_second, exp.(betas_train), train_data.ages, color = (:black, 0.2), markersize=15, label="Train", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        scatter!(ax_second, exp.(betas_test[type_indices]), test_data.ages[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
    end

    ax_di = Axis(gc[1,1], xlabel="βᵢ", ylabel= "Insulin Sensitivity Index", title="ρ = $(round(correlation_isi, digits=4))")

    scatter!(ax_di, exp.(betas_train), train_data.insulin_sensitivity, color = (:black, 0.2), markersize=15, label="Train", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        scatter!(ax_di, exp.(betas_test[type_indices]), test_data.insulin_sensitivity[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
    end

    Legend(fig[2,1:3], ax_first, orientation=:horizontal)

    for (label, layout) in zip(["a", "b", "c"], [ga, gb, gc])
        Label(layout[1, 1, TopLeft()], label,
        fontsize = 18,
        font = :bold,
        padding = (0, 20, 8, 0),
        halign = :right)
    end
    
    fig

end

save("figures/correlations_cude.$extension", correlation_figure, px_per_unit=4)

# supplementary correlation: 2nd phase clamp, body weight, bmi, disposition index
additional_correlation_figure = let fig = Figure(size=(1000,300))

   correlation_first = corspearman([betas_train; betas_test], [train_data.second_phase; test_data.second_phase])
   correlation_second = corspearman([betas_train; betas_test], [train_data.body_weights; test_data.body_weights])
   correlation_total = corspearman([betas_train; betas_test], [train_data.bmis; test_data.bmis])
   correlation_isi = corspearman([betas_train; betas_test], [train_data.disposition_indices; test_data.disposition_indices])

   markers=['●', '▴', '■']
   MAKERS = Dict(
       "NGT" => '●',
       "IGT" => '▴',
       "T2DM" => '■'
   )
   MARKERSIZES = Dict(
       "NGT" => 10,
       "IGT" => 18,
       "T2DM" => 10
   )

   ga = GridLayout(fig[1,1])
   gb = GridLayout(fig[1,2])
   gc = GridLayout(fig[1,3])
   gd = GridLayout(fig[1,4])

   ax_first = Axis(ga[1,1], xlabel="βᵢ", ylabel= "2ⁿᵈ Phase Clamp [μIU mL⁻¹ min]", title="ρ = $(round(correlation_first, digits=4))")

   scatter!(ax_first, exp.(betas_train), train_data.second_phase, color = (:black, 0.2), markersize=15, label="Train Data", marker='⋆')
   for (i,type) in enumerate(unique(test_data.types))
       type_indices = test_data.types .== type
       scatter!(ax_first, exp.(betas_test[type_indices]), test_data.second_phase[type_indices], color=COLORS[type], label="Test $type", marker=MAKERS[type], markersize=MARKERSIZES[type])
   end

   ax_second = Axis(gb[1,1], xlabel="βᵢ", ylabel= "Body weight [kg]", title="ρ = $(round(correlation_second, digits=4))")

   scatter!(ax_second, exp.(betas_train), train_data.body_weights, color = (:black, 0.2), markersize=15, label="Train", marker='⋆')
   for (i,type) in enumerate(unique(test_data.types))
       type_indices = test_data.types .== type
       scatter!(ax_second, exp.(betas_test[type_indices]), test_data.body_weights[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
   end

   ax_di = Axis(gc[1,1], xlabel="βᵢ", ylabel= "BMI [kg/m²]", title="ρ = $(round(correlation_total, digits=4))")

   scatter!(ax_di, exp.(betas_train), train_data.bmis, color = (:black, 0.2), markersize=15, label="Train", marker='⋆')
   for (i,type) in enumerate(unique(test_data.types))
       type_indices = test_data.types .== type
       scatter!(ax_di, exp.(betas_test[type_indices]), test_data.bmis[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
   end


   ax_isi = Axis(gd[1,1], xlabel="βᵢ", ylabel= "Clamp Disposition Index", title="ρ = $(round(correlation_isi, digits=4))")

    scatter!(ax_isi, exp.(betas_train), train_data.disposition_indices, color = (:black, 0.2), markersize=15, label="Train", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        scatter!(ax_isi, exp.(betas_test[type_indices]), test_data.disposition_indices[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
    end

   Legend(fig[2,1:4], ax_first, orientation=:horizontal)

   for (label, layout) in zip(["a", "b", "c", "d"], [ga, gb, gc, gd])
       Label(layout[1, 1, TopLeft()], label,
       fontsize = 18,
       font = :bold,
       padding = (0, 20, 8, 0),
       halign = :right)
   end
   
   fig
end

save("figures/supplementary/correlations_other_cude.$extension", additional_correlation_figure, px_per_unit=4)

# sample data for symbolic regression
betas_combined = exp.([betas_train; betas_test])
glucose_combined = [train_data.glucose; test_data.glucose]

beta_range = LinRange(minimum(betas_combined), maximum(betas_combined), 30)
glucose_range = LinRange(0.0, maximum(glucose_combined .- glucose_combined[:,1]), 30)

colnames = ["Beta", "Glucose", "Production"]
data = [ [β, glucose, chain([glucose, β], neural_network_parameters)[1] - chain([0.0, β], neural_network_parameters)[1]] for β in beta_range, glucose in glucose_range]
data = hcat(reshape(data, 20*20)...)

df = DataFrame(data', colnames)
CSV.write("data/ohashi_production.csv", df)