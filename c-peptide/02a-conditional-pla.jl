# Profile likelihood analysis for C-peptide

extension = "eps"
inch = 96
pt = 4/3
cm = inch / 2.54
linewidth = 13.07245cm
MANUSCRIPT_FIGURES = false
ECCB_FIGURES = true
FONTS = (
    ; regular = "Fira Sans Light",
    bold = "Fira Sans SemiBold",
    italic = "Fira Sans Italic",
    bold_italic = "Fira Sans SemiBold Italic",
)

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase
using Distributions: Chisq


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

neural_network_parameters, betas, best_model_index = try
    jldopen("source_data/cude_neural_parameters.jld2") do file
        file["parameters"], file["betas"], file["best_model_index"]
    end
catch
    error("Trained weights not found! Please train the model first by setting train_model to true")
end

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

function likelihood_profile(β, neural_network_parameters, model, timepoints, cpeptide_data, lower_bound, upper_bound; steps=1000)
    
    loss_minimum = loss(β, (model, timepoints, cpeptide_data, neural_network_parameters))

    loss_values = Float64[]

    for β in range(lower_bound, stop=upper_bound, length=steps)
        loss_value = loss(β, (model, timepoints, cpeptide_data, neural_network_parameters))
        push!(loss_values, loss_value)
    end

    return loss_values, loss_minimum
end

likelihood_profile_plot = let f = Figure()
    quan95 = 7.16; quan90 = 5.24;
    ax = Axis(f[1, 1], title="Likelihood profiles for all models", xlabel="Δβ", ylabel="ΔLikelihood")
    for (i, model) in enumerate(models_train)
        timepoints = train_data.timepoints
        cpeptide_data = train_data.cpeptide[i,:]
        lower_bound = betas_train[i] - 5.0
        upper_bound = betas_train[i] + 3.0

        loss_values, loss_minimum = likelihood_profile(betas_train[i], neural_network_parameters, model, timepoints, cpeptide_data, lower_bound, upper_bound)

        lines!(ax, range(-5.0, stop=3.0, length=1000), loss_values .- loss_minimum, color=(Makie.wong_colors()[2], 0.2), label="Likelihood profile")
        #ylims!(ax, 0.0, 10.0)
    end

    hlines!(ax, quan95,-3.0,3.0, color=(Makie.wong_colors()[1], 1), label="95% Cantelli Threshold", linestyle=:dash, linewidth=2.5)
    hlines!(ax, quan90,-3.0,3.0, color=(Makie.wong_colors()[3], 1), label="90% Cantelli Threshold", linestyle=:dash, linewidth=2.5)
    ylims!(ax, 0.0, 10.0)

    axislegend(ax, merge=true)
    f
end

save("figures/likelihood_profile.png", likelihood_profile_plot)