# Model fit to the train data and evaluation on the test data
# Partial pooling using ADVI to estimate the neural network and conditional
# parameters simultaneously


train_model = false
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

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra
using Bijectors: bijector

rng = StableRNG(232705)

include("../src/neural-network.jl")
include("../src/utils.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# define the neural network
network = chain(4, 2, tanh)
t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)

# create the models
models_train = [
    CPeptideConditionalUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], network, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

# # train on 70%, select on 30%
indices_train, indices_validation = stratified_split(rng, train_data.types, 0.7)


function predict(β, neural_network_parameters, problem, timepoints)
    p_model = ComponentArray(conditional = β, neural = neural_network_parameters)
    solution = Array(solve(problem, Tsit5(), p = p_model, saveat = timepoints, save_idxs = 1))

    if length(solution) < length(timepoints)
        # if the solution is shorter than the timepoints, we need to pad it
        solution = vcat(solution, fill(missing, length(timepoints) - length(solution)))
    end

    return solution
end

@model function partial_pooled(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where T

    # distribution for the population mean and standard deviation
    μ_beta ~ Normal(0.0, 1.0)
    σ_beta ~ InverseGamma(2, 3)

    # distribution for the individual model parameters
    β ~ MvNormal(ones(length(models)) * μ_beta, σ_beta * I)

    # distribution for the neural network parameters
    nn ~ MvNormal(zeros(length(neural_network_parameters)), 1.0 * I)

    # distribution for the model error
    σ ~ InverseGamma(2, 3)
    
    predictions = Array{T}(undef, length(models), length(timepoints))
    for i in eachindex(models)
        predictions[i,:] = predict(β[i], nn, models[i].problem, timepoints)
    end

    for i in axes(predictions, 2)
        data[:,i] ~ MvNormal(predictions[:,i], σ*I)
    end

    return nothing
end

# neural_network_parameters, betas, best_model_index = try
#     jldopen("source_data/cude_neural_parameters.jld2") do file
#         file["parameters"], file["betas"], file["best_model_index"]
#     end
# catch
#     error("Trained weights not found! Please train the model first by setting train_model to true")
# end
for repeat in 1:25
    println("Repeat: ", repeat)
    turing_model = partial_pooled(train_data.cpeptide[indices_train,:], train_data.timepoints, models_train[indices_train], init_params(models_train[1].chain, rng=rng));

    advi = ADVI(5, 10)
    advi_model = vi(turing_model, advi)


    _, sym2range = bijector(turing_model, Val(true));

    z = rand(advi_model, 10_000)
    sampled_nn_params = z[union(sym2range[:nn]...),:] # sampled parameters
    nn_params = mean(sampled_nn_params, dims=2)[:]
    sampled_betas = z[union(sym2range[:β]...),:] # sampled parameters
    betas = mean(sampled_betas, dims=2)[:]

    # save the models
    jldopen("source_data/advi/cude_result_$(repeat).jld2", "w") do file
        file["width"] = 4
        file["depth"] = 2
        file["parameters"] = nn_params
        file["betas"] = betas
    end
end

# random_effect_distribution = z[union(sym2range[:μ_beta]...),:][:]

# individual_effects = z[union(sym2range[:β]...),:]

# types = train_data.types[indices_train]

# random_effect_figure = let f = Figure()
#     ax = Axis(f[1,1])

#     for type in unique(types)
#         individual_mean_effects = mean(individual_effects[types .== type, :], dims=1)[:]
#         density!(ax, exp.(individual_mean_effects), color = (COLORS[type], 0.5), label= type)
#     end

#     axislegend(ax)

#     f
# end

#     # sampled parameters


# predictions = [
#     predict(betas[i], nn_params, models_train[idx].problem, train_data.timepoints) for (i,idx) in enumerate(indices_train)
# ]

# figure_model_fit = let f = Figure()
#     subject = 41
#     ax = Axis(f[1, 1], title = "Model fit", xlabel = "Time", ylabel = "C-peptide")
#     # sample parameters
#     samples = rand(advi_model, 1000)

#     for params in eachcol(samples)
#         nn_params = params[union(sym2range[:nn]...)]
#         betas = params[union(sym2range[:β]...)]

#         prediction = predict(betas[subject], nn_params, models_train[indices_train[subject]].problem, train_data.timepoints)
#         lines!(ax, train_data.timepoints, prediction, color = Makie.wong_colors()[1], alpha = 0.01)
#     end

#     scatter!(ax, train_data.timepoints, train_data.cpeptide[indices_train[subject],:], color = "black", markersize = 10)
#     f
# end

# figure_avgs = let f = Figure()
#     ax = Axis(f[1, 1], title = "Population mean", xlabel = "First Phase", ylabel = "β")
#     scatter!(ax, train_data.first_phase[indices_train], exp.(betas), color = "black", markersize = 10)
#     f
# end

# correlation = corspearman(train_data.first_phase[indices_train], exp.(betas))