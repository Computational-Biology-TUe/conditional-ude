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
    COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
    )

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra
using Bijectors: bijector
using ComponentArrays: ComponentArray

rng = StableRNG(232705)

include("../src/neural-network.jl")
include("../src/utils.jl")
include("../src/c-peptide-models.jl")

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

if train_model
    for repeat in 1:25
        println("Repeat: ", repeat)
        turing_model = partial_pooled(train_data.cpeptide[indices_train,:], train_data.timepoints, models_train[indices_train], init_params(models_train[1].chain, rng=rng));

        advi = ADVI(5, 3000)
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
else
    trained_models = []
    for repeat in 1:25
        nn_params, betas = try
            jldopen("source_data/advi/cude_result_$(repeat).jld2") do file
                file["parameters"], file["betas"]
            end
        catch
            error("Trained weights not found! Please train the model first by setting train_model to true")
        end
        push!(trained_models, (nn_params, betas))
    end
end

@model function partial_pooled_fixed_nn(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where T

    # distribution for the population mean and standard deviation
    μ_beta ~ Normal(0.0, 1.0)
    σ_beta ~ InverseGamma(2, 3)

    # distribution for the individual model parameters
    β ~ MvNormal(ones(length(models)) * μ_beta, σ_beta * I)

    # distribution for the model error
    σ ~ InverseGamma(2, 3)
    
    predictions = Array{T}(undef, length(models), length(timepoints))
    for i in eachindex(models)
        predictions[i,:] = predict(β[i], neural_network_parameters, models[i].problem, timepoints)
    end

    for i in axes(predictions, 2)
        data[:,i] ~ MvNormal(predictions[:,i], σ*I)
    end

    return nothing
end

# train models with fixed nn parameters


cpeptide = [train_data.cpeptide; test_data.cpeptide]

models_test = [
    CPeptideConditionalUDEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], network, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
]

models = [models_train; models_test]

for repeat_ in 1:25
    turing_model = partial_pooled_fixed_nn(cpeptide, train_data.timepoints, models, trained_models[repeat_][1]);

    advi = ADVI(5, 1000)
    advi_model = vi(turing_model, advi)

    _, sym2range = bijector(turing_model, Val(true));

    z = rand(advi_model, 10_000)
    sampled_betas = z[union(sym2range[:β]...),:] # sampled parameters
    betas = mean(sampled_betas, dims=2)[:]
    betas_test = betas[end-34:end]
    types = [train_data.types; test_data.types]
    random_effect_figure = let f = Figure()
        ax = Axis(f[1,1], xlabel="βᵢ value", ylabel="Density")

        for type in unique(types)
            individual_mean_effects =betas[types .== type]
            density!(ax, exp.(individual_mean_effects), color = (COLORS[type], 0.5), label= type)
        end

        axislegend(ax)

        f
    end

    save("figures/advi/model_$(repeat_)_random_effects.eps", random_effect_figure, px_per_unit=4)
    save("figures/advi/model_$(repeat_)_random_effects.svg", random_effect_figure, px_per_unit=4)
    save("figures/advi/model_$(repeat_)_random_effects.png", random_effect_figure, px_per_unit=4)


    figure_correlations = let f = Figure(size=(400,400))
        clamp_indices = [train_data.first_phase; test_data.first_phase]
        corval = corspearman(betas, clamp_indices)
        ax = Axis(f[1,1], xlabel="βᵢ value", ylabel="First phase clamp", title="ρ = $(round(corval, digits=4))")



        for type in unique(types)
            individual_mean_effects =betas[types .== type]
            scatter!(ax, exp.(individual_mean_effects), clamp_indices[types .== type], color=(COLORS[type], 1.0), label=type)
    
        end
        Legend(f[2,1], ax, merge=true, orientation=:horizontal)
        f
    end

    save("figures/advi/model_$(repeat_)_correlations.eps", figure_correlations, px_per_unit=4)
    save("figures/advi/model_$(repeat_)_correlations.svg", figure_correlations, px_per_unit=4)
    save("figures/advi/model_$(repeat_)_correlations.png", figure_correlations, px_per_unit=4)

    model_fit_all_test = let fig
        fig = Figure(size = (1000, 1500))
        sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
        sols = [Array(solve(model.problem, p=ComponentArray(conditional=[betas_test[i]], neural=trained_models[repeat_][1]), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
        
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

            lines!(ax, sol_timepoints, sol[:,1], color=(:black, 1), linewidth=2, label="Model fit", linestyle=:solid)
            scatter!(ax, test_data.timepoints, c_peptide_data , color=(:black, 1), markersize=10, label="Data")

        end

        linkyaxes!(axs...)

        Legend(fig[locations[end][1]+1, 0:4], axs[1], orientation=:horizontal, merge=true)

        fig
    end

    save("figures/advi/model_$(repeat_)_model_fit_test.eps", model_fit_all_test, px_per_unit=4)
    save("figures/advi/model_$(repeat_)_model_fit_test.svg", model_fit_all_test, px_per_unit=4)
    save("figures/advi/model_$(repeat_)_model_fit_test.png", model_fit_all_test, px_per_unit=4)

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