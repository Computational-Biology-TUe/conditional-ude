# Model fit to the train data and evaluation on the test data
extension = "eps"
inch = 96
pt = 4/3
cm = inch / 2.54
linewidth = 13.07245cm



using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase

COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
)

rng = StableRNG(232705)

include("../src/parameter-estimation.jl")
include("../src/utils.jl")
include("../src/likelihood-profiles.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# define the production function 
function production(ΔG, k)
    prod = ΔG >= 0 ? 1.78ΔG/(ΔG + k[1]) : 0.0
    return prod
end

figure_production = let f = Figure(size=(linewidth,6cm), fontsize=10pt)


    ga = GridLayout(f[1,1])
    gb = GridLayout(f[1,2])
    df = DataFrame(CSV.File("data/ohashi_production.csv"))
    beta_values = df[1:20, :Beta]
    
    ax = Axis(ga[1,1], xlabel="ΔG (mM)", ylabel="Production (nM min⁻¹)", title="Neural Network")
    for (i, beta) in enumerate(beta_values)
        df_beta = df[df[!,:Beta] .== beta, :]        
        lines!(ax, df_beta.Glucose, df_beta.Production, color = i, colorrange=(1,20), colormap=:viridis)
    end
    k_values = 167 .* beta_values.^3 .+ 21.8
    println(k_values)
    ax2 = Axis(gb[1,1], xlabel="ΔG (mM)", ylabel="Production (nM min⁻¹)", title="Symbolic")
    for (i, k) in enumerate(k_values)
        df_beta = df[df[!,:Beta] .== beta_values[i], :]
        
        production_values = production.(df_beta.Glucose, k)        
        lines!(ax2, df_beta.Glucose, production_values, color = i, colorrange=(1,20), colormap=:viridis)
    end
    Colorbar(f[1,3], limits=(beta_values[1], beta_values[end]), label="β")
    
    for (label, layout) in zip(["a", "b"], [ga, gb])
        Label(layout[1, 1, TopLeft()], label,
        fontsize = 12pt,
        font = :bold,
        padding = (0, 20, 8, 0),
        halign = :right)
    end

    linkyaxes!(ax, ax2)

    
    f

end

save("figures/supplementary/dose_response_neural_symbolic.$extension", figure_production, px_per_unit=300/inch)


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
optfunc = OptimizationFunction(loss_sigma, AutoForwardDiff())
for (i,model) in enumerate(all_models)

    optprob = OptimizationProblem(optfunc, ComponentArray(ode=[40.0], sigma=1.0), (model, timepoints, cpeptide_data[i,:]),
    lb = 0.0, ub = 1000.0)
    optsol = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000)
    push!(optsols, optsol)
end

betas = [optsol.u.ode[1] for optsol in optsols]
sigmas = [optsol.u.sigma for optsol in optsols]
objectives = ([optsol.objective for optsol in optsols].- (length(test_data.timepoints)/2) .* log.(sigmas.^2)) .* (2 .* sigmas.^2)


model_fit_figure = let fig
    fig = Figure(size = (linewidth, 10cm), fontsize=8pt)
    gas = [GridLayout(fig[1,1]), GridLayout(fig[1,2]), GridLayout(fig[1,3])]
    gb = GridLayout(fig[1,4])
    # do the simulations
    sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
    sols = [Array(solve(model.problem, p=betas[i], saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(all_models)]
    
    axs = [Axis(gas[i][1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title=type) for (i,type) in enumerate(unique(test_data.types))]
    lbs = [-40, -30, -80]
    ubs = [5000, 50, 5000]
    for (i,type) in enumerate(unique(test_data.types))

        type_indices = [train_data.types; test_data.types] .== type

        cpeptide_data_type = cpeptide_data[type_indices,:]

        sol_idx = findfirst(objectives[type_indices] .== median(objectives[type_indices]))

        # find the median fit of the type
        sol_type = sols[type_indices][sol_idx]

        lines!(axs[i], sol_timepoints, sol_type[:,1], color=(COLORS[type], 1), linewidth=1.5, label="Model fit", linestyle=:dot)
        scatter!(axs[i], test_data.timepoints, cpeptide_data_type[sol_idx,:] , color=(COLORS[type], 1), markersize=5, label="Data")

        # run PLA
        loss_values, loss_minimum, parameter_values = likelihood_profile(
            betas[type_indices][sol_idx], loss, (all_models[type_indices][sol_idx], train_data.timepoints, cpeptide_data_type[sol_idx,:]), betas[type_indices][sol_idx]+lbs[i], betas[type_indices][sol_idx]+ubs[i], sigmas[type_indices][sol_idx]; steps=10_000
        )
        min_parameter_value, max_parameter_value = find_confidence_intervals(loss_values, loss_minimum, parameter_values; target=:cantelli95)

        if !isinf(min_parameter_value)
            # compute the solution with the lower and upper bounds
            sol_lower = Array(solve(all_models[type_indices][sol_idx].problem, p=[min_parameter_value], saveat=sol_timepoints, save_idxs=1))
            lines!(axs[i], sol_timepoints, sol_lower[:,1], color=(COLORS[type], 0.5), linewidth=1, label="95% CI", linestyle=:dot)
        end 
        if !isinf(max_parameter_value)
            sol_upper = Array(solve(all_models[type_indices][sol_idx].problem, p=[max_parameter_value], saveat=sol_timepoints, save_idxs=1))

            lines!(axs[i], sol_timepoints, sol_upper[:,1], color=(COLORS[type], 0.5), linewidth=1, label="95% CI", linestyle=:dot)
        end

    end

    linkyaxes!(axs...)

    ax = Axis(gb[1,1], xticks=([0,1,2], ["NGT", "IGT", "T2DM"]), xlabel="Type", ylabel="MSE", xticklabelrotation=pi/4)

    jitter_width = 0.1

    for (i, type) in enumerate(unique(train_data.types))
        jitter = rand(length(objectives)) .* jitter_width .- jitter_width/2
        type_indices = [train_data.types .== type; test_data.types .== type]
        scatter!(ax, repeat([i-1], length(objectives[type_indices])) .+ jitter[type_indices] .- 0.1, objectives[type_indices], color=(COLORS[type], 0.8), markersize=3, label=type)
        violin!(ax, repeat([i-1], length(objectives[type_indices])) .+ 0.05, objectives[type_indices], color=(COLORS[type], 0.8), width=0.75, side=:right, strokewidth=1, datalimits=(0,Inf))
    end

    Legend(fig[2,1:3], axs[1], orientation=:horizontal, merge=true)

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
        "NGT" => 5,
        "IGT" => 9,
        "T2DM" => 5
    )

    gc = GridLayout(fig[3,1:4])
    gcs = [GridLayout(gc[1,1]), GridLayout(gc[1,2]), GridLayout(gc[1,3])]

    ax_first = Axis(gcs[1][1,1], xlabel="log₁₀ [kₘ]", ylabel= "1ˢᵗ Phase Clamp", title="ρ = $(round(correlation_first, digits=4))")

    #scatter!(ax_first, exp.(betas), train_data.first_phase, color = (:black, 0.2), markersize=15, label="Train Data", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = [train_data.types; test_data.types] .== type
        scatter!(ax_first, log10.(betas[type_indices]), [train_data.first_phase; test_data.first_phase][type_indices], color=COLORS[type], label="$type", marker=MAKERS[type], markersize=MARKERSIZES[type])
    end

    ax_second = Axis(gcs[2][1,1], xlabel="log₁₀ [kₘ]", ylabel= "Age [y]", title="ρ = $(round(correlation_second, digits=4))")

    #scatter!(ax_second, exp.(betas_train), train_data.ages, color = (:black, 0.2), markersize=15, label="Train", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = [train_data.types; test_data.types] .== type
        scatter!(ax_second, log10.(betas[type_indices]), [train_data.ages; test_data.ages][type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
    end

    ax_di = Axis(gcs[3][1,1], xlabel="log₁₀ [kₘ]", ylabel= "Insulin Sensitivity Index", title="ρ = $(round(correlation_isi, digits=4))")

    #scatter!(ax_di, exp.(betas_train), train_data.insulin_sensitivity, color = (:black, 0.2), markersize=15, label="Train", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = [train_data.types; test_data.types] .== type
        scatter!(ax_di, log10.(betas[type_indices]), [train_data.insulin_sensitivity; test_data.insulin_sensitivity][type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
    end

    Legend(fig[4,1:4], ax_first, orientation=:horizontal)

    for (label, layout) in zip(["a", "b", "c", "d", "e", "f", "g"], [gas; [gb]; gcs])
        Label(layout[1, 1, TopLeft()], label,
        fontsize = 10pt,
        font = :bold,
        padding = (0, 20, 8, 0),
        halign = :right)
    end
rowgap!(fig.layout, 1, 0)
rowgap!(fig.layout, 2, 5)
rowgap!(fig.layout, 3, 5)

fig
end



save("figures/symbolic_regression_internal.$extension", model_fit_figure, px_per_unit=300/inch)
types= [train_data.types; test_data.types]
# Figure Sj - likelihood profiles
figure_likelihood_profiles = let f = Figure(size=(7cm, 7cm), fontsize=8pt, fonts=FONTS)

    ax = Axis(f[1,1], xlabel="Δβ", ylabel="ΔLikelihood", xlabelfont=:bold, ylabelfont=:bold, xgridvisible=true, ygridvisible=true, topspinevisible=false, rightspinevisible=false)
    for (i, model) in enumerate(all_models)
        timepoints = train_data.timepoints
        lower_bound = betas[i] - 100.0
        upper_bound = betas[i] + 500.0

        loss_values, loss_minimum, parameter_values = likelihood_profile(betas[i], loss, (model, timepoints, cpeptide_data[i,:]), lower_bound, upper_bound, sigmas[i]; steps=1000)

        lines!(ax, range(-100.0, stop=500.0, length=1000), loss_values .- loss_minimum, color=(COLORS[types[i]], 0.2), label="Likelihood profile")
    end

    ylims!(ax, 0.0, 10.0)
    hlines!(ax, 7.16,-3.0,3.0, color=(Makie.wong_colors()[1], 1), label="95% Cantelli Threshold", linestyle=:dash, linewidth=2.5)

    f
end