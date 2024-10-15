# Model fit to the train data and evaluation on the test data
extension = "png"

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase

rng = StableRNG(232705)

include("src/c-peptide-ude-models.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# define the production function 
function production(ΔG, k)
    prod = ΔG >= 0 ? 1.78ΔG/(ΔG + k) : 0.0
    return prod
end

figure_production = let f = Figure(size=(600,300))

    df = DataFrame(CSV.File("data/ohashi_production.csv"))
    beta_values = df[1:20, :Beta]
    
    ax = Axis(f[1,1], xlabel="ΔG (mM)", ylabel="Production (nM min⁻¹)", title="Neural Network")
    for (i, beta) in enumerate(beta_values)
        df_beta = df[df[!,:Beta] .== beta, :]        
        lines!(ax, df_beta.Glucose, df_beta.Production, color = i, colorrange=(1,20), colormap=:viridis, label = "Beta = $beta_1")
    end
    k_values = 167 .* beta_values.^3 .+ 21.8
    println(k_values)
    ax2 = Axis(f[1,2], xlabel="ΔG (mM)", ylabel="Production (nM min⁻¹)", title="Symbolic")
    for (i, k) in enumerate(k_values)
        df_beta = df[df[!,:Beta] .== beta_values[i], :]
        
        production_values = production.(df_beta.Glucose, k)        
        lines!(ax2, df_beta.Glucose, production_values, color = i, colorrange=(1,20), colormap=:viridis, label = "Beta = $beta_1")
    end
    Colorbar(f[1,3], limits=(beta_values[1], beta_values[end]))
    f


end

t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)

# create the models
models_train = [
    CPeptideODEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], production, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

t2dm = test_data.types .== "T2DM"
models_test = [
    CPeptideODEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], production, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
]

all_models = [models_train; models_test]
cpeptide_data = [train_data.cpeptide; test_data.cpeptide]
timepoints = train_data.timepoints
optsols = OptimizationSolution[]
optfunc = OptimizationFunction(loss, AutoForwardDiff())
for (i,model) in enumerate(all_models)

    optprob = OptimizationProblem(optfunc, [40.0], (model, timepoints, cpeptide_data[i,:]),
    lb = 0.0, ub = 1000.0)
    optsol = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000)
    push!(optsols, optsol)
end

betas = [optsol.u[1] for optsol in optsols]
objectives = [optsol.objective for optsol in optsols]

# TODO: combine figures in one
model_fit_figure = let fig
    fig = Figure(size = (775, 300))
    ga = GridLayout(fig[1,1:3], )
    gb = GridLayout(fig[1,4], nrow=1, ncol=1)
    # do the simulations
    sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
    sols = [Array(solve(model.problem, p=betas[i], saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(all_models)]
    
    axs = [Axis(ga[1,i], xlabel="Time [min]", ylabel="C-peptide [nM]", title=type) for (i,type) in enumerate(unique(test_data.types))]

    for (i,type) in enumerate(unique(test_data.types))

        type_indices = [train_data.types; test_data.types] .== type

        cpeptide_data_type = cpeptide_data[type_indices,:]

        sol_idx = findfirst(objectives[type_indices] .== median(objectives[type_indices]))

        # find the median fit of the type
        sol_type = sols[type_indices][sol_idx]

        lines!(axs[i], sol_timepoints, sol_type[:,1], color=(:black, 1), linewidth=2, label="Model fit", linestyle=:dot)
        scatter!(axs[i], test_data.timepoints, cpeptide_data_type[sol_idx,:] , color=(:black, 1), markersize=10, label="Data")

    end

    ax = Axis(gb[1,1], xticks=([0,1,2], ["NGT", "IGT", "T2DM"]), xlabel="Type", ylabel="log₁₀ (Error)")
    boxplot!(ax, repeat([0], sum(test_data.types .== "NGT")), log10.(objectives[[train_data.types; test_data.types] .== "NGT"]), color=COLORS["NGT"], width=0.75)
    boxplot!(ax, repeat([1], sum(test_data.types .== "IGT")),log10.(objectives[[train_data.types; test_data.types] .== "IGT"]), color=COLORS["IGT"], width=0.75)
    boxplot!(ax, repeat([2], sum(test_data.types .== "T2DM")),log10.(objectives[[train_data.types; test_data.types] .== "T2DM"]), color=COLORS["T2DM"], width=0.75)

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

#save("figures/model_fit_test_median.$extension", model_fit_figure, px_per_unit=4)

# Correlation figure; 1st phase clamp, age, insulin sensitivity 
correlation_figure = let fig
    fig = Figure(size=(700,300))

    #betas_train = optsols_train[argmin(objectives_train)].u.ode[:]
    #betas_test = [optsol.u[1] for optsol in optsols_test]

    correlation_first = corspearman(betas, [train_data.first_phase; test_data.first_phase])
    correlation_second = corspearman(betas, [train_data.ages; test_data.ages])
    correlation_isi = corspearman(betas, [train_data.insulin_sensitivity; test_data.insulin_sensitivity])

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

    #scatter!(ax_first, exp.(betas), train_data.first_phase, color = (:black, 0.2), markersize=15, label="Train Data", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = [train_data.types; test_data.types] .== type
        scatter!(ax_first, log10.(betas[type_indices]), [train_data.first_phase; test_data.first_phase][type_indices], color=COLORS[type], label="$type", marker=MAKERS[type], markersize=MARKERSIZES[type])
    end

    ax_second = Axis(gb[1,1], xlabel="βᵢ", ylabel= "Age [y]", title="ρ = $(round(correlation_second, digits=4))")

    #scatter!(ax_second, exp.(betas_train), train_data.ages, color = (:black, 0.2), markersize=15, label="Train", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = [train_data.types; test_data.types] .== type
        scatter!(ax_second, log10.(betas[type_indices]), [train_data.ages; test_data.ages][type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
    end

    ax_di = Axis(gc[1,1], xlabel="βᵢ", ylabel= "Insulin Sensitivity Index", title="ρ = $(round(correlation_isi, digits=4))")

    #scatter!(ax_di, exp.(betas_train), train_data.insulin_sensitivity, color = (:black, 0.2), markersize=15, label="Train", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = [train_data.types; test_data.types] .== type
        scatter!(ax_di, log10.(betas[type_indices]), [train_data.insulin_sensitivity; test_data.insulin_sensitivity][type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
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

#save("figures/correlations_cude.$extension", correlation_figure, px_per_unit=4)

