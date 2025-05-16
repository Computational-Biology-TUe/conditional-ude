using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, Random, Statistics, FileIO

MAKE_FIGURES = true

rng = StableRNG(232705)

include("../src/parameter-estimation.jl")
include("../src/neural-network.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# Fit the c-peptide data with a regular UDE model on the average data of the train subgroup
mean_c_peptide_train = mean(train_data.cpeptide, dims=1)
std_c_peptide_train = std(train_data.cpeptide, dims=1)[:]

mean_glucose_train = mean(train_data.glucose, dims=1)

neural_network = chain(
    4, 2, tanh; input_dims=1
)

model_train = CPeptideUDEModel(mean_glucose_train[:], train_data.timepoints, mean(train_data.ages), neural_network, mean_c_peptide_train[:], false)
optsols_train = train(model_train, train_data.timepoints, mean_c_peptide_train[:], rng)

best_model = optsols_train[argmin([optsol.objective for optsol in optsols_train])]
neural_network_parameters = best_model.u[:]

if MAKE_FIGURES

    # define colors for the figures
    COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
    )

    COLORLIST = [
        RGBf(252/255, 253/255, 191/255),
        RGBf(254/255, 191/255, 132/255),
        RGBf(250/255, 127/255, 94/255),
    ]

    FONTS = (
    ; regular = "Fira Sans Light",
    bold = "Fira Sans SemiBold",
    italic = "Fira Sans Italic",
    bold_italic = "Fira Sans SemiBold Italic",
)

    inch = 96
    pt = 4/3
    cm = inch / 2.54
    linewidth = 21cm * 0.8

    # simulate the model for the train data
    models_train = [
        CPeptideUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], neural_network, train_data.cpeptide[i,:], train_data.types[i] == "T2DM") for i in axes(train_data.glucose, 1)
    ]

    data_timepoints = train_data.timepoints
    solutions_train = [Array(solve(model.problem, Tsit5(), p=neural_network_parameters, saveat=data_timepoints, save_idxs=1)) for model in models_train]

    errors_train = [sum(abs2, sol - train_data.cpeptide[i,:])/length(sol) for (i, sol) in enumerate(solutions_train)]

    # simulate the model for the test data
    models_test = [
        CPeptideUDEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], neural_network, test_data.cpeptide[i,:], test_data.types[i] == "T2DM") for i in axes(test_data.glucose, 1)
    ]

    data_timepoints = test_data.timepoints
    solutions_test = [Array(solve(model.problem, Tsit5(), p=neural_network_parameters, saveat=data_timepoints, save_idxs=1)) for model in models_test]

    errors_test = [sum(abs2, sol - test_data.cpeptide[i,:])/length(sol) for (i, sol) in enumerate(solutions_test)]

    
    # figure for error
    figure_error = let f = Figure(size=(linewidth/3, 5cm), fontsize=8pt, fonts = FONTS)
        ax = Axis(f[1,1], xlabel="", ylabel="MSE", xticks=([1,3],["Train set", "Test set"]))
        ax.xgridvisible = false
        ax.ygridvisible = true
        ax.xlabelfont = :bold
        ax.ylabelfont = :bold
        ax.xticklabelfont = :bold
        ax.backgroundcolor = :transparent

        jitter_width = 0.1

        for (i, type) in enumerate(unique(train_data.types))
            jitter = rand(length(errors_train)) .* jitter_width .- jitter_width/2
            type_indices = train_data.types .== type
            scatter!(ax, repeat([0+i]/2, length(errors_train[type_indices])) .+ jitter[type_indices] .- 0.1, errors_train[type_indices], color=(COLORS[type], 0.8), markersize=3, label=type)
            violin!(ax, repeat([0+i/2], length(errors_train[type_indices])), errors_train[type_indices], color=(COLORS[type], 0.8), width=0.5, side=:right, strokewidth=1, datalimits=(0,Inf))

            jitter_2 = rand(length(errors_test)) .* jitter_width .- jitter_width/2
            type_indices = test_data.types .== type
            scatter!(ax, repeat([2+i/2], length(errors_test[type_indices])) .+ jitter_2[type_indices] .- 0.1, errors_test[type_indices], color=(COLORS[type], 0.8), markersize=3, label=type)
            violin!(ax, repeat([2+i/2], length(errors_test[type_indices])), errors_test[type_indices], color=(COLORS[type], 0.8), width=0.5, side=:right, strokewidth=1, datalimits=(0,Inf))
        end
    f
    end

    save("figures/revision/figure_1/model_fit_error.png", figure_error, px_per_unit=300/inch)
    save("figures/revision/figure_1/model_fit_error.svg", figure_error, px_per_unit=300/inch)

    # figures for model fits
    plot_timepoints = train_data.timepoints[1]:0.1:train_data.timepoints[end]
    figure_model_fit = let f = Figure(size=(linewidth*0.9, 7cm), fontsize=8pt, fonts = FONTS)
        axs = []
        for (i, type) in enumerate(unique(test_data.types))

            type_indices = test_data.types .== type
    
            mean_c_peptide = mean(test_data.cpeptide[type_indices,:], dims=1)[:]
            std_c_peptide = std(test_data.cpeptide[type_indices,:], dims=1)[:]
            mean_glucose = mean(test_data.glucose[type_indices,:], dims=1)[:]
            mean_age = mean(test_data.ages[type_indices])
            ax = Axis(f[1,i], title=type, xlabel="Time [min]", ylabel="C-peptide [nmol/L]", backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=true, ygridvisible=true, topspinevisible=false, rightspinevisible=false)

            model = CPeptideUDEModel(mean_glucose, test_data.timepoints, mean_age, neural_network, mean_c_peptide,type == type)
            sol = Array(solve(model.problem, Tsit5(), p=neural_network_parameters, saveat=plot_timepoints, save_idxs=1))
            lines!(ax, plot_timepoints, sol, color=(COLORS[type], 1), linewidth=2, label="Model", linestyle=:dash)

            scatter!(ax, test_data.timepoints, mean_c_peptide, color=(COLORS[type], 1), markersize=8, label="Data (mean ± std)")
            errorbars!(ax, test_data.timepoints, mean_c_peptide, std_c_peptide, color=(COLORS[type], 1), whiskerwidth=7, label="Data (mean ± std)", linewidth=1.5)
            push!(axs, ax)
        end
        linkyaxes!(axs...)

        # add legend
        legend = Legend(f[2, 1:3], axs[1], merge=true, orientation = :horizontal, fontsize=8pt, font = FONTS)

       f

    end
    save("figures/revision/figure_1/model_fit_ude_test.png", figure_model_fit, px_per_unit=300/inch)
    save("figures/revision/figure_1/model_fit_ude_test.svg", figure_model_fit, px_per_unit=300/inch)

    # dose response figure
    figure_dose_response = let f = Figure(size=(6cm, 6cm), fontsize=8pt, fonts = FONTS)
        ax = Axis(f[1,1], xlabel="ΔGlucose [mg/dL]", ylabel="c-peptide prod. [nmol/(L ⋅ min)]", backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=true, ygridvisible=true, topspinevisible=false, rightspinevisible=false)

        glucose_samples = 0.0:0.01:maximum(test_data.glucose)-minimum(test_data.glucose)

        production(g) = neural_network([g], neural_network_parameters)[1]

        lines!(ax, glucose_samples, production.(glucose_samples), color=(Makie.wong_colors()[1], 1.0), linewidth=2, label="Model", linestyle=:solid)
        f
    end

    save("figures/revision/figure_1/dose_response.png", figure_dose_response, px_per_unit=300/inch)
    save("figures/revision/figure_1/dose_response.svg", figure_dose_response, px_per_unit=300/inch)

    # Supplementary experiment and figure - Fit on NGT only
    figure_fit_ngt = let f = Figure(size=(linewidth*0.9, 7cm), fontsize=8pt, fonts = FONTS)

        c_peptide = [train_data.cpeptide; test_data.cpeptide]
        glucose = [train_data.glucose; test_data.glucose]
        types = train_data.types
        ages = [train_data.ages; test_data.ages]

        train_cpeptide = mean(train_data.cpeptide[types .== "NGT",:], dims=1)[:]
        train_glucose = mean(train_data.glucose[types .== "NGT",:], dims=1)[:]
        train_ages = mean(train_data.ages[types .== "NGT"])

        neural_network = chain(
            4, 2, tanh; input_dims=1
        )

        model_train = CPeptideUDEModel(train_glucose[:], train_data.timepoints, train_ages, neural_network, train_cpeptide, false)
        optsols_train = train(model_train, train_data.timepoints, train_cpeptide, rng)

        best_model = optsols_train[argmin([optsol.objective for optsol in optsols_train])]
        neural_network_parameters = best_model.u[:]

        axs = []
        for (i, type) in enumerate(unique(test_data.types))

            type_indices = test_data.types .== type
    
            mean_c_peptide = mean(test_data.cpeptide[type_indices,:], dims=1)[:]
            std_c_peptide = std(test_data.cpeptide[type_indices,:], dims=1)[:]
            mean_glucose = mean(test_data.glucose[type_indices,:], dims=1)[:]
            mean_age = mean(test_data.ages[type_indices])
            ax = Axis(f[1,i], title=type, xlabel="Time [min]", ylabel="C-peptide [nmol/L]", backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=true, ygridvisible=true, topspinevisible=false, rightspinevisible=false)

            model = CPeptideUDEModel(mean_glucose, test_data.timepoints, mean_age, neural_network, mean_c_peptide,type == type)
            sol = Array(solve(model.problem, Tsit5(), p=neural_network_parameters, saveat=plot_timepoints, save_idxs=1))
            lines!(ax, plot_timepoints, sol, color=(COLORS[type], 1), linewidth=2, label="Model", linestyle=:dash)

            scatter!(ax, test_data.timepoints, mean_c_peptide, color=(COLORS[type], 1), markersize=8, label="Data (mean ± std)")
            errorbars!(ax, test_data.timepoints, mean_c_peptide, std_c_peptide, color=(COLORS[type], 1), whiskerwidth=7, label="Data (mean ± std)", linewidth=1.5)
            push!(axs, ax)
        end
        linkyaxes!(axs...)

        # add legend
        legend = Legend(f[2, 1:3], axs[1], merge=true, orientation = :horizontal, fontsize=8pt, font = FONTS)
        f
    end

    save("figures/revision/figure_sy/model_fit_ude_test_ngt.png", figure_fit_ngt, px_per_unit=300/inch)
end