# Model fit to the train data and evaluation on the test data
RETRAIN_MODEL = false
MAKE_FIGURES = true

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

indices_train, indices_validation = stratified_split(rng, train_data.types, 0.7)

# define the neural network
network = chain(4, 2, tanh)
t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)

# create the models
models_train = [
    CPeptideConditionalUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], network, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

# train the models or load the trained model neural network parameters
if RETRAIN_MODEL
    optsols_train = train(models_train[indices_train], train_data.timepoints, train_data.cpeptide[indices_train,:], rng)
    neural_network_parameters = [optsol.u.neural[:] for optsol in optsols_train]
    betas = [optsol.u.conditional[:] for optsol in optsols_train]

    objectives = evaluate_model(models_train[indices_validation],
    train_data.timepoints, train_data.cpeptide[indices_validation,:], neural_network_parameters,
    betas)

    best_model_index = argmin(sum(objectives, dims=2)[:])
    best_model = optsols_train[best_model_index]

    # save the models
    jldopen("source_data/cude_neural_parameters.jld2", "w") do file
        file["width"] = 4
        file["depth"] = 2
        file["parameters"] = neural_network_parameters
        file["betas"] = betas
        file["best_model_index"] = best_model_index
    end
else
    neural_network_parameters, betas, best_model_index = try
        jldopen("source_data/cude_neural_parameters.jld2") do file
            file["parameters"], file["betas"], file["best_model_index"]
        end
    catch
        error("Trained weights not found! Please train the model first by setting train_model to true")
    end
end

objectives = evaluate_model(models_train[indices_validation],
train_data.timepoints, train_data.cpeptide[indices_validation,:], neural_network_parameters,
betas)

second_best_index = sortperm(sum(objectives, dims=2)[:])[2]

    COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
    )


    inch = 96
    pt = 4/3
    cm = inch / 2.54
    linewidth = 21cm * 0.8


    FONTS = (
    ; regular = "assets/fonts/Newsreader_9pt-Regular.ttf",
    bold = "assets/fonts/Newsreader_9pt-Bold.ttf",
    italic = "assets/fonts/Newsreader_9pt-Italic.ttf",
    bold_italic = "assets/fonts/Newsreader_9pt-BoldItalic.ttf",
)

# obtain the betas for the train data
    lb = minimum(betas[best_model_index]) - 0.1*abs(minimum(betas[best_model_index]))
    ub = maximum(betas[best_model_index]) + 0.1*abs(maximum(betas[best_model_index]))

    optsols = train_with_sigma(models_train, train_data.timepoints, train_data.cpeptide, neural_network_parameters[best_model_index], lbfgs_lower_bound=lb, lbfgs_upper_bound=ub, initial_beta=-1.0)
    betas_train = [optsol.u.ode[1] for optsol in optsols]
    sigmas_train = [optsol.u.sigma for optsol in optsols]
    objectives_train = ([optsol.objective for optsol in optsols] .- (length(train_data.timepoints)/2) .* log.(sigmas_train.^2)) .* (2 .* sigmas_train.^2)


    # obtain the betas for the test data
    t2dm = test_data.types .== "T2DM"
    models_test = [
        CPeptideConditionalUDEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], network, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
    ]

    optsols = train_with_sigma(models_test, test_data.timepoints, test_data.cpeptide, neural_network_parameters[best_model_index], lbfgs_lower_bound=lb, lbfgs_upper_bound=ub, initial_beta=-1.0)
    betas_test = [optsol.u.ode[1] for optsol in optsols]
    sigmas_test = [optsol.u.sigma for optsol in optsols]
    objectives_test = ([optsol.objective for optsol in optsols] .- (length(test_data.timepoints)/2) .* log.(sigmas_test.^2)) .* (2 .* sigmas_test.^2)

    for type in unique(train_data.types)
    type_indices = [train_data.types; test_data.types] .== type
    println("Type: ", type)
    mse_values_type = [objectives_train; objectives_test][type_indices]
    println("MSE: ", mean(mse_values_type))
    end

if MAKE_FIGURES

    COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
    )


    inch = 96
    pt = 4/3
    cm = inch / 2.54
    linewidth = 21cm * 0.8

    FONTS = (
    ; regular = "assets/fonts/Newsreader_9pt-Regular.ttf",
    bold = "assets/fonts/Newsreader_9pt-Bold.ttf",
    italic = "assets/fonts/Newsreader_9pt-Italic.ttf",
    bold_italic = "assets/fonts/Newsreader_9pt-BoldItalic.ttf",
)

    # Figure 3 - model fits and errors for the test data

    # obtain the betas for the train data
    lb = minimum(betas[best_model_index]) - 0.1*abs(minimum(betas[best_model_index]))
    ub = maximum(betas[best_model_index]) + 0.1*abs(maximum(betas[best_model_index]))

    optsols = train_with_sigma(models_train, train_data.timepoints, train_data.cpeptide, neural_network_parameters[best_model_index], lbfgs_lower_bound=lb, lbfgs_upper_bound=ub, initial_beta=-1.0)
    betas_train = [optsol.u.ode[1] for optsol in optsols]
    sigmas_train = [optsol.u.sigma for optsol in optsols]
    objectives_train = ([optsol.objective for optsol in optsols] .- (length(train_data.timepoints)/2) .* log.(sigmas_train.^2)) .* (2 .* sigmas_train.^2)


    # obtain the betas for the test data
    t2dm = test_data.types .== "T2DM"
    models_test = [
        CPeptideConditionalUDEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], network, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
    ]

    optsols = train_with_sigma(models_test, test_data.timepoints, test_data.cpeptide, neural_network_parameters[best_model_index], lbfgs_lower_bound=lb, lbfgs_upper_bound=ub, initial_beta=-1.0)
    betas_test = [optsol.u.ode[1] for optsol in optsols]
    sigmas_test = [optsol.u.sigma for optsol in optsols]
    objectives_test = ([optsol.objective for optsol in optsols] .- (length(test_data.timepoints)/2) .* log.(sigmas_test.^2)) .* (2 .* sigmas_test.^2)

    for type in unique(train_data.types)
    type_indices = [train_data.types; test_data.types] .== type
    println("Type: ", type)
    mse_values_type = [objectives_train; objectives_test][type_indices]
    println("MSE: ", mean(mse_values_type))
    end


    model_fit_figure = let fig = Figure(size = (linewidth*0.9, 6cm), fontsize=8pt, fonts=FONTS)
        # do the simulations
        sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
        sols = [Array(solve(model.problem, p=ComponentArray(conditional=[betas_test[i]], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
        
        axs = [Axis(fig[1,i], title=type, xlabel="Time [min]", ylabel= i == 1 ? "C-peptide [nmol/L]" : "", backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=true, ygridvisible=true, topspinevisible=false, rightspinevisible=false) for (i,type) in enumerate(unique(test_data.types))]

        for (i,type) in enumerate(unique(test_data.types))

            type_indices = test_data.types .== type

            c_peptide_data = test_data.cpeptide[type_indices,:]

            sol_idx = findfirst(objectives_test[type_indices] .== median(objectives_test[type_indices]))

            # find the median fit of the type
            sol_type = sols[type_indices][sol_idx]

            # obtain confidence intervals with likelihood profiles
            loss_values, loss_minimum, parameter_values = likelihood_profile(
                betas_test[type_indices][sol_idx], neural_network_parameters[best_model_index], models_test[type_indices][sol_idx], test_data.timepoints, c_peptide_data[sol_idx,:], betas_test[type_indices][sol_idx]-10, betas_test[type_indices][sol_idx]+15, sigmas_test[type_indices][sol_idx]; steps=10_000
            )

            # find the 95% confidence interval
            min_parameter, max_parameter = find_confidence_intervals(
                loss_values, loss_minimum, parameter_values
            )

            # compute the solution with the lower and upper bounds
            sol_lower = Array(solve(models_test[type_indices][sol_idx].problem, p=ComponentArray(conditional=[min_parameter], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1))

            lines!(axs[i], sol_timepoints, sol_lower[:,1], color=(COLORS[type], 0.5), linewidth=1, label="95% CI", linestyle=:dot)

            if !isinf(max_parameter)
                sol_upper = Array(solve(models_test[type_indices][sol_idx].problem, p=ComponentArray(conditional=[max_parameter], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1))

                lines!(axs[i], sol_timepoints, sol_upper[:,1], color=(COLORS[type], 0.5), linewidth=1, label="95% CI", linestyle=:dot)
            end

            lines!(axs[i], sol_timepoints, sol_type[:,1], color=(COLORS[type], 1), linewidth=2, label="Model fit", linestyle=:solid)
            scatter!(axs[i], test_data.timepoints, c_peptide_data[sol_idx,:] , color=(COLORS[type], 1), markersize=7, label="Data")

        end

        linkyaxes!(axs...)

        ax = Axis(fig[1,4], xticks=([0,1,2], ["NGT", "IGT", "T2DM"]), xlabel="Type", ylabel="MSE", xlabelfont=:bold, ylabelfont=:bold, xgridvisible=true, ygridvisible=true)

        jitter_width = 0.1

        for (i, type) in enumerate(unique(train_data.types))
            jitter = rand(length(objectives_test)) .* jitter_width .- jitter_width/2
            type_indices = test_data.types .== type
            scatter!(ax, repeat([i-1], length(objectives_test[type_indices])) .+ jitter[type_indices] .- 0.1, objectives_test[type_indices], color=(COLORS[type], 0.8), markersize=3, label=type)
            violin!(ax, repeat([i-1], length(objectives_test[type_indices])) .+ 0.05, objectives_test[type_indices], color=(COLORS[type], 0.8), width=0.75, side=:right, strokewidth=1, datalimits=(0,Inf))
        end

        Legend(fig[2,1:3], axs[1], merge=true, orientation=:horizontal)

    fig
    end

    save("figures/revision/figure_3/model_fit_test_median.png", model_fit_figure, px_per_unit=300/inch)
    save("figures/revision/figure_3/model_fit_test_median.svg", model_fit_figure, px_per_unit=300/inch)


    # Figure 4 - correlation between betas and other parameters

    correlation_figure = let fig = Figure(size = (linewidth, 7cm), fontsize=8pt, fonts=FONTS)
        # compute the correlations
        correlation_first = corspearman([betas_train; betas_test], [train_data.first_phase; test_data.first_phase])
        correlation_second = corspearman([betas_train; betas_test], [train_data.ages; test_data.ages])
        correlation_isi = corspearman([betas_train; betas_test], [train_data.insulin_sensitivity; test_data.insulin_sensitivity])

        markers=[:circle, :utriangle, :rect]
        MAKERS = Dict(
            "NGT" => :circle,
            "IGT" => :utriangle,
            "T2DM" => :rect
        )
        MARKERSIZES = Dict(
            "NGT" => 9,
            "IGT" => 9,
            "T2DM" => 9
        )

        ax_first = Axis(fig[1,1], xlabel="βᵢ", ylabel= "1ˢᵗ Phase Clamp", title="ρ = $(round(correlation_first, digits=4))")

        scatter!(ax_first, exp.(betas_train), train_data.first_phase, color = (:black, 0.2), markersize=6, label="Train Data", marker=:star5)
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax_first, exp.(betas_test[type_indices]), test_data.first_phase[type_indices], color=COLORS[type], label="Test $type", marker=MAKERS[type], markersize=MARKERSIZES[type])
        end

        ax_second = Axis(fig[1,2], xlabel="βᵢ", ylabel= "Age [y]", title="ρ = $(round(correlation_second, digits=4))")

        scatter!(ax_second, exp.(betas_train), train_data.ages, color = (:black, 0.2), markersize=6, label="Train", marker=:star5)
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax_second, exp.(betas_test[type_indices]), test_data.ages[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
        end

        ax_di = Axis(fig[1,3], xlabel="βᵢ", ylabel= "Ins. Sens. Index", title="ρ = $(round(correlation_isi, digits=4))")

        scatter!(ax_di, exp.(betas_train), train_data.insulin_sensitivity, color = (:black, 0.2), markersize=6, label="Train", marker=:star5)
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax_di, exp.(betas_test[type_indices]), test_data.insulin_sensitivity[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
        end

        Legend(fig[2,1:3], ax_first, orientation=:horizontal)
        
        fig

    end

    save("figures/revision/figure_4/correlation.png", correlation_figure, px_per_unit=300/inch)
    save("figures/revision/figure_4/correlation.svg", correlation_figure, px_per_unit=300/inch)


    additional_correlation_figure = let fig = Figure(size = (linewidth, 6cm), fontsize=8pt)

        correlation_first = corspearman([betas_train; betas_test], [train_data.second_phase; test_data.second_phase])
        correlation_second = corspearman([betas_train; betas_test], [train_data.body_weights; test_data.body_weights])
        correlation_total = corspearman([betas_train; betas_test], [train_data.bmis; test_data.bmis])
        correlation_isi = corspearman([betas_train; betas_test], [train_data.disposition_indices; test_data.disposition_indices])

        markers=[:circle, :utriangle, :rect]
        MAKERS = Dict(
            "NGT" => :circle,
            "IGT" => :utriangle,
            "T2DM" => :rect
        )
        MARKERSIZES = Dict(
            "NGT" => 9,
            "IGT" => 9,
            "T2DM" => 9
        )

        ga = GridLayout(fig[1,1])
        gb = GridLayout(fig[1,2])
        gc = GridLayout(fig[1,3])
        gd = GridLayout(fig[1,4])

        ax_first = Axis(ga[1,1], xlabel="βᵢ", ylabel= "2ⁿᵈ Phase Clamp", title="ρ = $(round(correlation_first, digits=4))")

        scatter!(ax_first, exp.(betas_train), train_data.second_phase, color = (:black, 0.2), markersize=6, label="Train Data", marker=:star5)
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax_first, exp.(betas_test[type_indices]), test_data.second_phase[type_indices], color=COLORS[type], label="Test $type", marker=MAKERS[type], markersize=MARKERSIZES[type])
        end

        ax_second = Axis(gb[1,1], xlabel="βᵢ", ylabel= "Body weight [kg]", title="ρ = $(round(correlation_second, digits=4))")

        scatter!(ax_second, exp.(betas_train), train_data.body_weights, color = (:black, 0.2), markersize=6, label="Train", marker=:star5)
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax_second, exp.(betas_test[type_indices]), test_data.body_weights[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
        end

        ax_di = Axis(gc[1,1], xlabel="βᵢ", ylabel= "BMI [kg/m²]", title="ρ = $(round(correlation_total, digits=4))")

        scatter!(ax_di, exp.(betas_train), train_data.bmis, color = (:black, 0.2), markersize=6, label="Train", marker=:star5)
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax_di, exp.(betas_test[type_indices]), test_data.bmis[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
        end


        ax_isi = Axis(gd[1,1], xlabel="βᵢ", ylabel= "Clamp DI", title="ρ = $(round(correlation_isi, digits=4))")

            scatter!(ax_isi, exp.(betas_train), train_data.disposition_indices, color = (:black, 0.2), markersize=6, label="Train", marker=:star5)
            for (i,type) in enumerate(unique(test_data.types))
                type_indices = test_data.types .== type
                scatter!(ax_isi, exp.(betas_test[type_indices]), test_data.disposition_indices[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
            end

        Legend(fig[2,1:4], ax_first, orientation=:horizontal)

        for (label, layout) in zip(["a", "b", "c", "d"], [ga, gb, gc, gd])
            Label(layout[1, 1, TopLeft()], label,
            fontsize = 12,
            font = :bold,
            padding = (0, 20, 8, 0),
            halign = :right)
        end
        
        fig
    end

    save("figures/revision/supplementary/correlation_sup.png", additional_correlation_figure, px_per_unit=300/inch)
    save("figures/revision/supplementary/correlation_sup.eps", additional_correlation_figure, px_per_unit=300/inch)

    # Figure Sj - likelihood profiles
    figure_likelihood_profiles = let f = Figure(size=(14cm, 7cm), fontsize=8pt, fonts=FONTS)

        ax = Axis(f[1,1], xlabel="Δβ", ylabel="ΔLikelihood", xlabelfont=:bold, ylabelfont=:bold, xgridvisible=true, ygridvisible=true, topspinevisible=false, rightspinevisible=false)

        models = [models_train; models_test]
        betas = [betas_train; betas_test]
        cpeptide = [train_data.cpeptide; test_data.cpeptide]
        sigmas = [sigmas_train; sigmas_test]
        n_identifiable = 0; n_practically_identifiable = 0; n_unidentifiable = 0
        plots = []
        for (i, model) in enumerate(models)
            timepoints = train_data.timepoints
            cpeptide_data = cpeptide[i,:]
            lower_bound = betas[i] - 10.0
            upper_bound = betas[i] + 10.0

            loss_values, loss_minimum, parameter_values = likelihood_profile(betas[i], neural_network_parameters[best_model_index], model, timepoints, cpeptide_data, lower_bound, upper_bound, sigmas[i]; steps=1000)

            # check identifiability
            Δloss = loss_values .- loss_minimum
            lower_bound = findfirst(Δloss .<= 7.16)
            upper_bound = findlast(Δloss .<= 7.16)

            if lower_bound == 1 && upper_bound == length(Δloss)
                # unidentifiable
                linecolor = (Makie.wong_colors()[4], 1.0)
                label = "Unidentifiable"
                n_unidentifiable += 1
            elseif lower_bound == 1 || upper_bound == length(Δloss)
                # practically unidentifiable
                linecolor = (Makie.wong_colors()[1], 1.0)
                label = "Practically unidentifiable"
                n_practically_identifiable += 1
            else
                # identifiable
                linecolor = (Makie.wong_colors()[2], 0.1)
                label = "Identifiable"
                n_identifiable += 1
            end

            push!(plots, (loss_values .- loss_minimum, label, linecolor))
        end

        for plot in plots
            parameter_values = range(-10, 10, length(plot[1]))
            if plot[2] == "Identifiable"
                lines!(ax, parameter_values, plot[1], color=plot[3], label="Identifiable (n = $n_identifiable)")
            elseif plot[2] == "Practically unidentifiable"
                lines!(ax, parameter_values, plot[1], color=plot[3], label="Practically unidentifiable (n = $n_practically_identifiable)")
            else
                lines!(ax, parameter_values, plot[1], color=plot[3], label="Unidentifiable (n = $n_unidentifiable)")
            end
        end

        ylims!(ax, 0.0, 10.0)
        hlines!(ax, 7.16,-3.0,3.0, color=(Makie.wong_colors()[3], 1), label="95% Cantelli Threshold", linestyle=:dash, linewidth=2.5)
        Legend(f[1,2], ax, orientation=:vertical, merge=true)

        f
    end

    save("figures/revision/supplementary/likelihood_curves.png", figure_likelihood_profiles, px_per_unit=300/inch)
    save("figures/revision/supplementary/likelihood_curves.svg", figure_likelihood_profiles, px_per_unit=300/inch)
    save("figures/revision/supplementary/likelihood_curves.eps", figure_likelihood_profiles, px_per_unit=300/inch)


   figure_other_betas = let f = Figure(size = (1000, 1000), fontsize=8pt)
        other_betas = betas
        g = GridLayout(f[1, 1], nrow=5, ncol=5)
        axs = [Axis(g[1 + (i-1) ÷ 5, 1 + (i-1) % 5], xlabel="βᵢ", ylabel="First Phase Clamp", title="Model $(i)") for i in eachindex(betas)]
        for (i,b) in enumerate(other_betas)
            correlation = corspearman(exp.(b), train_data.first_phase[indices_train])
            println("Model $(i): ", correlation)
            background_color = correlation > 0.0 ? (COLORS["T2DM"], 0.1) : (COLORS["IGT"], 0.1)
            axs[i].backgroundcolor = background_color
            scatter!(axs[i], exp.(b), train_data.first_phase[indices_train], color = (:black, 0.9), markersize=6, marker=:circle)
        end
    f
    end

    save("figures/revision/supplementary/other_betas.png", figure_other_betas, px_per_unit=300/inch)
    save("figures/revision/supplementary/other_betas.svg", figure_other_betas, px_per_unit=300/inch)
    save("figures/revision/supplementary/other_betas.eps", figure_other_betas, px_per_unit=300/inch)


    model_fit_train = let fig
        fig = Figure(size = (linewidth, 6cm), fontsize=8pt)
        ga = [GridLayout(fig[1,1], ), GridLayout(fig[1,2], ), GridLayout(fig[1,3], )]
        gb = GridLayout(fig[1,4], nrow=1, ncol=1)

        # do the simulations
        sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
        sols = [Array(solve(model.problem, p=ComponentArray(conditional=[betas_train[i]], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_train)]
        
        axs = [Axis(ga[i][1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", title=type) for (i,type) in enumerate(unique(train_data.types))]

        for (i,type) in enumerate(unique(train_data.types))

            type_indices = train_data.types .== type

            c_peptide_data = train_data.cpeptide[type_indices,:]

            sol_idx = argmedian(objectives_train[type_indices])

            # find the median fit of the type
            sol_type = sols[type_indices][sol_idx]
# obtain confidence intervals with likelihood profiles
            loss_values, loss_minimum, parameter_values = likelihood_profile(
                betas_train[type_indices][sol_idx], neural_network_parameters[best_model_index], models_train[type_indices][sol_idx], test_data.timepoints, c_peptide_data[sol_idx,:], betas_train[type_indices][sol_idx]-10, betas_train[type_indices][sol_idx]+15, sigmas_train[type_indices][sol_idx]; steps=10_000
            )

            # find the 95% confidence interval
            min_parameter, max_parameter = find_confidence_intervals(
                loss_values, loss_minimum, parameter_values
            )

            # compute the solution with the lower and upper bounds
            sol_lower = Array(solve(models_train[type_indices][sol_idx].problem, p=ComponentArray(conditional=[min_parameter], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1))

            lines!(axs[i], sol_timepoints, sol_lower[:,1], color=(COLORS[type], 0.5), linewidth=1, label="95% CI", linestyle=:dot)

            if !isinf(max_parameter)
                sol_upper = Array(solve(models_train[type_indices][sol_idx].problem, p=ComponentArray(conditional=[max_parameter], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1))

                lines!(axs[i], sol_timepoints, sol_upper[:,1], color=(COLORS[type], 0.5), linewidth=1, label="95% CI", linestyle=:dot)
            end

            lines!(axs[i], sol_timepoints, sol_type[:,1], color=(COLORS[type], 1), linewidth=2, label="Model fit", linestyle=:solid)
            scatter!(axs[i], test_data.timepoints, c_peptide_data[sol_idx,:] , color=(COLORS[type], 1), markersize=7, label="Data")

        end

        linkyaxes!(axs...)

        ax = Axis(gb[1,1], xticks=([0,1,2], ["NGT", "IGT", "T2DM"]), xlabel="Type", ylabel="log₁₀ (Error)")
    
        jitter_width = 0.1

        for (i, type) in enumerate(unique(train_data.types))
            jitter = rand(length(objectives_train)) .* jitter_width .- jitter_width/2
            type_indices = train_data.types .== type
            scatter!(ax, repeat([i-1], length(objectives_train[type_indices])) .+ jitter[type_indices] .- 0.1, objectives_train[type_indices], color=(COLORS[type], 0.8), markersize=3, label=type)
            violin!(ax, repeat([i-1], length(objectives_train[type_indices])) .+ 0.05, objectives_train[type_indices], color=(COLORS[type], 0.8), width=0.75, side=:right, strokewidth=1, datalimits=(0,Inf))
        end
    
        # boxplot!(ax, repeat([0], sum(train_data.types .== "NGT")), log10.(objectives_train[train_data.types .== "NGT"]), color=COLORS["NGT"], width=0.75)
        # boxplot!(ax, repeat([1], sum(train_data.types .== "IGT")),log10.(objectives_train[train_data.types .== "IGT"]), color=COLORS["IGT"], width=0.75)
        # boxplot!(ax, repeat([2], sum(train_data.types .== "T2DM")),log10.(objectives_train[train_data.types .== "T2DM"]), color=COLORS["T2DM"], width=0.75)

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

    save("figures/revision/supplementary/model_fit_train_median.eps", model_fit_train, px_per_unit=4)
    save("figures/revision/supplementary/model_fit_train_median.svg", model_fit_train, px_per_unit=4)
    save("figures/revision/supplementary/model_fit_train_median.png", model_fit_train, px_per_unit=4)

    

end


model_fit_all_test = let fig
    fig = Figure(size = (1000, 1500))
    sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
    sols = [Array(solve(model.problem, p=ComponentArray(conditional=[betas_test[i]], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
    
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


        loss_values, loss_minimum, parameter_values = likelihood_profile(
                betas_test[i], neural_network_parameters[best_model_index], models_test[i], test_data.timepoints, c_peptide_data, betas_test[i]-10, betas_test[i]+15, sigmas_test[i]; steps=10_000
            )

        # find the 95% confidence interval
        min_parameter, max_parameter = find_confidence_intervals(
            loss_values, loss_minimum, parameter_values
        )

        # compute the solution with the lower and upper bounds
        if !isinf(min_parameter)
            sol_lower = Array(solve(models_test[i].problem, p=ComponentArray(conditional=[min_parameter], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1))

            lines!(ax, sol_timepoints, sol_lower[:,1], color=(:black, 0.8), linewidth=1, label="95% CI", linestyle=:dot)
        end

        if !isinf(max_parameter)
            sol_upper = Array(solve(models_test[i].problem, p=ComponentArray(conditional=[max_parameter], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1))

            lines!(ax, sol_timepoints, sol_upper[:,1], color=(:black, 0.8), linewidth=1, label="95% CI", linestyle=:dot)
        end


        lines!(ax, sol_timepoints, sol[:,1], color=(:black, 1), linewidth=2, label="Model fit", linestyle=:solid)
        scatter!(ax, test_data.timepoints, c_peptide_data , color=(:black, 1), markersize=10, label="Data")

    end

    linkyaxes!(axs...)

    Legend(fig[locations[end][1]+1, 0:4], axs[1], orientation=:horizontal, merge=true)

    fig
end

save("figures/revision/supplementary/model_fit_test_all.eps", model_fit_all_test, px_per_unit=4)
save("figures/revision/supplementary/model_fit_test_all.svg", model_fit_all_test, px_per_unit=4)
save("figures/revision/supplementary/model_fit_test_all.png", model_fit_all_test, px_per_unit=4)


# Figure Sx - beta distribution and sampled parameters
figure_beta_distribution = let f = Figure(size=(8cm, 7cm), fontsize=8pt, fonts=FONTS)

    ax = Axis(f[1,1], xlabel="βᵢ", ylabel="Density", topspinevisible=false, rightspinevisible=false, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=false, ygridvisible=true)
    density!(ax, exp.([betas_train; betas_test]), color = (:black, 0.4),label="all types")

    for (i,type) in enumerate(unique(test_data.types))
        type_indices_test = test_data.types .== type
        type_indices_train = train_data.types .== type
        density!(ax, [exp.(betas_test[type_indices_test]); exp.(betas_train[type_indices_train])], color=(COLORS[type], 0.3), label=type)
    end

    Legend(f[2,1], ax, orientation=:horizontal, fontsize=8pt, font=FONTS)
    f
end

figure_sampled_simulations = let f = Figure(size=(16cm, 7cm), fontsize=8pt, fonts=FONTS)

    sol_timepoints = train_data.timepoints[1]:0.1:train_data.timepoints[end]

    axs = [Axis(f[1,i], xlabel="Time [min]", ylabel= i == 1 ? "C-peptide [nmol/L]" : "", title=type, backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=true, ygridvisible=true, topspinevisible=false, rightspinevisible=false) for (i,type) in enumerate(unique(test_data.types))]
    # for each type, sample betas and simulate the model
    for (i,type) in enumerate(unique(test_data.types))

        type_indices_test = test_data.types .== type
        type_indices_train = train_data.types .== type

        # mean c-peptide data
        avg_cpeptide = mean([train_data.cpeptide[type_indices_train, :]; test_data.cpeptide[type_indices_test, :]], dims=1)[:]
        avg_glucose = mean([train_data.glucose[type_indices_train, :]; test_data.glucose[type_indices_test, :]], dims=1)[:]
        avg_ages = mean([train_data.ages[type_indices_train]; test_data.ages[type_indices_test]])
        std_cpeptide = std([train_data.cpeptide[type_indices_train, :]; test_data.cpeptide[type_indices_test, :]], dims=1)[:]

        model = 
            CPeptideConditionalUDEModel(avg_glucose, train_data.timepoints, avg_ages, network, avg_cpeptide, type == "T2DM")

        betas_from = [betas_train[type_indices_train]; betas_test[type_indices_test]]
        # sample betas
        sampled_betas = rand(rng, betas_from, 500)

        # simulate the model
        sols = [Array(solve(model.problem, p=ComponentArray(conditional=[sampled_betas[i]], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1)) for i in 1:500]

        for sol in sols
            lines!(axs[i], sol_timepoints, sol[:,1], color=(:black, 0.005), label="Simulation (sample)")
        end

        mean_sol = mean(hcat(sols...), dims=2)[:]
        
        # plot the mean c-peptide data
        errorbars!(axs[i], train_data.timepoints, avg_cpeptide, std_cpeptide, color="black", label="Data (Mean ± std)", whiskerwidth=5)
        scatter!(axs[i], train_data.timepoints, avg_cpeptide, color="black", markersize=7, label="Data (Mean ± std)")

        # plot the simulations
        lines!(axs[i], sol_timepoints, mean_sol, color=(COLORS[type], 0.9), label="Simulation (mean)", linewidth=2)





    end
    linkyaxes!(axs...)
    # add legend
    legend = Legend(f[2,1:3], axs[1], merge=true, orientation = :horizontal, fontsize=8pt, font = FONTS)

    f
end

save("figures/revision/supplementary/beta_distribution.png", figure_beta_distribution, px_per_unit=300/inch)
save("figures/revision/supplementary/sampled_simulations.png", figure_sampled_simulations, px_per_unit=300/inch)
save("figures/revision/supplementary/beta_distribution.svg", figure_beta_distribution, px_per_unit=300/inch)
save("figures/revision/supplementary/sampled_simulations.svg", figure_sampled_simulations, px_per_unit=300/inch)

second_best_index = 8
figure_second_best_correlation = let f = Figure(size=(6cm, 6cm), fontsize=8pt, fonts=FONTS)

    markers=[:circle, :utriangle, :rect]
    MARKERS = Dict(
        "NGT" => :circle,
        "IGT" => :utriangle,
        "T2DM" => :rect
    )
    MARKERSIZES = Dict(
        "NGT" => 9,
        "IGT" => 9,
        "T2DM" => 9
    )
    # obtain the betas for the train data
    lb = minimum(betas[second_best_index]) - 0.1*abs(minimum(betas[second_best_index]))
    ub = maximum(betas[second_best_index]) + 0.1*abs(maximum(betas[second_best_index]))

    optsols = train(models_train, train_data.timepoints, train_data.cpeptide, neural_network_parameters[second_best_index], lbfgs_lower_bound=lb, lbfgs_upper_bound=ub, initial_beta=-1.0)
    betas_train_2 = [optsol.u[1] for optsol in optsols]
    objectives_train = [optsol.objective for optsol in optsols]

    # obtain the betas for the test data
    t2dm = test_data.types .== "T2DM"
    models_test = [
        CPeptideConditionalUDEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], network, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
    ]

    optsols = train(models_test, test_data.timepoints, test_data.cpeptide, neural_network_parameters[second_best_index], lbfgs_lower_bound=lb, lbfgs_upper_bound=ub, initial_beta=-1.0)
    betas_test_2 = [optsol.u[1] for optsol in optsols]
    objectives_test = [optsol.objective for optsol in optsols]

    correlation_first = corspearman([betas_train_2; betas_test_2], [betas_train; betas_test])

    ax = Axis(f[1,1], xlabel="βᵢ Model 2", ylabel= "βᵢ Model 1", title="ρ = $(round(correlation_first, digits=4))")

    scatter!(ax, exp.(betas_train_2), exp.(betas_train), color = (:black, 0.2), markersize=6, label="Train Data", marker=:star5)
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        scatter!(ax, exp.(betas_test_2[type_indices]),  exp.(betas_test[type_indices]), color=COLORS[type], label="Test $type", marker=MARKERS[type], markersize=MARKERSIZES[type])
    end

    f
end


save("figures/revision/figure_s8/second_best_correlation_comparison.png", figure_second_best_correlation, px_per_unit=300/inch)

background_color = "#111111"

# Figure comparison with non-conditional model
figure_comparison = let f = Figure(size=(17cm, 7cm), fontsize=10pt, fonts=FONTS, backgroundcolor=:transparent, textcolor=background_color)

    COLORS_COMPARISON = Dict(
        "NGT" => "#4D6F64",
        "IGT" => "#A97B6D",
        "T2DM" => "#5C7FA3"
    )

    MAKERS = Dict(
    "NGT" => :circle,
    "IGT" => :utriangle,
    "T2DM" => :rect
)

    # fit the non-conditional model
    # Fit the c-peptide data with a regular UDE model on the average data of the train subgroup
    mean_c_peptide_train = mean(train_data.cpeptide, dims=1)
    std_c_peptide_train = std(train_data.cpeptide, dims=1)[:]

    mean_glucose_train = mean(train_data.glucose, dims=1)

    neural_network_nc = chain(
        4, 2, tanh; input_dims=1
    )

    model_train_nc = CPeptideUDEModel(mean_glucose_train[:], train_data.timepoints, mean(train_data.ages), neural_network_nc, mean_c_peptide_train[:], false)
    optsols_train_nc = train(model_train_nc, train_data.timepoints, mean_c_peptide_train[:], rng)

    best_model_nc = optsols_train_nc[argmin([optsol.objective for optsol in optsols_train_nc])]
    neural_network_parameters_nc = best_model_nc.u[:]

    plot_timepoints = train_data.timepoints[1]:0.1:train_data.timepoints[end]
    ax_cude = Axis(f[1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", title="cUDE", backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=true, ygridvisible=true, topspinevisible=false, rightspinevisible=false, spinewidth=0.05cm, bottomspinecolor=background_color, leftspinecolor=background_color, xtickcolor=background_color, ytickcolor=background_color, xticklabelcolor=background_color, yticklabelcolor=background_color, xgridcolor=(background_color,0.3), ygridcolor=(background_color,0.3))

    ax_ude = Axis(f[1,2], xlabel="Time [min]", backgroundcolor=:transparent, title="UDE",xlabelfont=:bold, ylabelfont=:bold, xgridvisible=true, ygridvisible=true, topspinevisible=false, rightspinevisible=false, spinewidth=0.05cm, bottomspinecolor=background_color, leftspinecolor=background_color, xtickcolor=background_color, ytickcolor=background_color, xticklabelcolor=background_color, yticklabelcolor=background_color, xgridcolor=(background_color,0.3), ygridcolor=(background_color,0.3))
    for (i, type) in enumerate(unique(test_data.types))

        type_indices = test_data.types .== type

        mean_c_peptide = mean(test_data.cpeptide[type_indices,:], dims=1)
        std_c_peptide = std(test_data.cpeptide[type_indices,:], dims=1)[:]
        ste_c_peptide = std_c_peptide ./ sqrt(sum(type_indices))
        mean_glucose = mean(test_data.glucose[type_indices,:], dims=1)[:]
        mean_age = mean(test_data.ages[type_indices])


        # conditional model

        ## define model
        model = CPeptideConditionalUDEModel(mean_glucose, test_data.timepoints, mean_age, network, mean_c_peptide[:], type == "T2DM")
        
        ## estimate beta
        optsol = train([model], test_data.timepoints, mean_c_peptide, neural_network_parameters[best_model_index], lbfgs_lower_bound=lb, lbfgs_upper_bound=ub, initial_beta=-1.0)
        beta = optsol[1].u[1]

        # simulate the model
        sol = Array(solve(model.problem, Tsit5(), p=ComponentArray(conditional=[beta], neural=neural_network_parameters[best_model_index]), saveat=plot_timepoints, save_idxs=1))

        lines!(ax_cude, plot_timepoints, sol, color=(COLORS_COMPARISON[type], 1), linewidth=0.1cm, label="$type (Model)", linestyle=:solid)

        # non-conditional model
        ## define model
        model = CPeptideUDEModel(mean_glucose, test_data.timepoints, mean_age, neural_network_nc, mean_c_peptide[:],type == "T2DM")
        sol = Array(solve(model.problem, Tsit5(), p=neural_network_parameters_nc, saveat=plot_timepoints, save_idxs=1))
        lines!(ax_ude, plot_timepoints, sol, color=(COLORS_COMPARISON[type], 1), linewidth=0.1cm, label="$type (Model)", linestyle=:solid)

        scatter!(ax_ude, test_data.timepoints, mean_c_peptide[:], color=(COLORS_COMPARISON[type], 1), markersize=0.4cm, label="$type (mean ± std. err.)", marker=MAKERS[type])
        errorbars!(ax_ude, test_data.timepoints, mean_c_peptide[:], ste_c_peptide, color=(COLORS_COMPARISON[type], 1), whiskerwidth=7, label="$type (mean ± std. err.)", linewidth=0.08cm)
        scatter!(ax_cude, test_data.timepoints, mean_c_peptide[:], color=(COLORS_COMPARISON[type], 1), markersize=0.4cm, label="$type (mean ± std. err.)", marker=MAKERS[type])
        errorbars!(ax_cude, test_data.timepoints, mean_c_peptide[:], ste_c_peptide, color=(COLORS_COMPARISON[type], 1), whiskerwidth=7, label="$type (mean ± std. err.)", linewidth=0.08cm)
        #push!(axs, ax)
    end
    linkyaxes!(ax_cude, ax_ude)

    # add legend
    legend = Legend(f[1, 3], ax_cude, merge=true, orientation = :vertical, fontsize=8pt, font = FONTS, backgroundcolor=:transparent, framevisible=false)

    f

end

save("figures/eccb/poster_plot.svg", figure_comparison, px_per_unit=300/inch)

figure_comparison_errors = let f = Figure(size=(10cm, 7cm), fontsize=10pt, fonts=FONTS, backgroundcolor=:transparent, textcolor=background_color)

    COLORS_COMPARISON = Dict(
        "NGT" => "#4D6F64",
        "IGT" => "#A68A64",
        "T2DM" => "#5C7FA3"
    )

    MAKERS = Dict(
    "NGT" => :circle,
    "IGT" => :utriangle,
    "T2DM" => :rect
)

    # fit the non-conditional model
    # Fit the c-peptide data with a regular UDE model on the average data of the train subgroup
    mean_c_peptide_train = mean(train_data.cpeptide, dims=1)
    std_c_peptide_train = std(train_data.cpeptide, dims=1)[:]

    mean_glucose_train = mean(train_data.glucose, dims=1)

    neural_network_nc = chain(
        4, 2, tanh; input_dims=1
    )

    model_train_nc = CPeptideUDEModel(mean_glucose_train[:], train_data.timepoints, mean(train_data.ages), neural_network_nc, mean_c_peptide_train[:], false)
    optsols_train_nc = train(model_train_nc, train_data.timepoints, mean_c_peptide_train[:], rng)

    best_model_nc = optsols_train_nc[argmin([optsol.objective for optsol in optsols_train_nc])]
    neural_network_parameters_nc = best_model_nc.u[:]

    #plot_timepoints = train_data.timepoints[1]:0.1:train_data.timepoints[end]
    ax_cude = Axis(f[1,1], xlabel="Group", ylabel="MAE (nmol/L)", backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=true, ygridvisible=true, topspinevisible=false, rightspinevisible=false, spinewidth=0.05cm, bottomspinecolor=background_color, leftspinecolor=background_color, xtickcolor=background_color, ytickcolor=background_color, xticklabelcolor=background_color, yticklabelcolor=background_color, xgridcolor=(background_color,0.3), ygridcolor=(background_color,0.3), xticks=(0:2, ["NGT", "IGT", "T2DM"]))


    cude_errors = []
    ude_errors = []
    for (i, type) in enumerate(test_data.types)

        type_indices = test_data.types .== type

        c_peptide = test_data.cpeptide[i,:]
        glucose = test_data.glucose[i,:]
        age = test_data.ages[i]

        # non-conditional model
        ## define model
        model = CPeptideUDEModel(glucose, test_data.timepoints, age, neural_network_nc, c_peptide,type == "T2DM")
        sol = Array(solve(model.problem, Tsit5(), p=neural_network_parameters_nc, saveat=test_data.timepoints, save_idxs=1))

        mse_cpeptide_nc = sum(abs2, sol[:,1] .- c_peptide)
        push!(ude_errors, mse_cpeptide_nc)
    end

    # make boxplots
    type_map = Dict("NGT" => 0, "IGT" => 1, "T2DM" => 2)
    locs = [type_map[type] for type in test_data.types]
    colors = [COLORS_COMPARISON[type] for type in test_data.types]
    violin!(ax_cude, locs .+ 0.2, sqrt.(objectives_test), color=(COLORS_COMPARISON["NGT"], 0.6), width=0.8, side=:right, strokewidth=1, datalimits=(0,Inf))
    scatter!(ax_cude, locs .+ 0.1, sqrt.(objectives_test), color=COLORS_COMPARISON["NGT"], markersize=10, label="cUDE", marker=:circle)
    violin!(ax_cude, locs .-0.2, sqrt.(ude_errors), color=(COLORS_COMPARISON["T2DM"], 0.6), width=0.8, side=:right, strokewidth=1, datalimits=(0,Inf))
    scatter!(ax_cude, locs .-0.3, sqrt.(ude_errors), color=COLORS_COMPARISON["T2DM"], markersize=10, label="UDE", marker=:utriangle)

    # add legend
    legend = Legend(f[1, 2], ax_cude, merge=true, orientation = :vertical, fontsize=12pt, font = FONTS, backgroundcolor=:transparent, framevisible=false, markersize=14)

    f

end

save("figures/eccb/poster_plot_errors.svg", figure_comparison_errors, px_per_unit=300/inch, backgroundcolor=:transparent)

correlation_figure = let fig = Figure(size = (11cm, 7cm), fontsize=12pt, fonts=FONTS, backgroundcolor=:transparent)
    # compute the correlations
    correlation_first = corspearman([betas_train; betas_test], [train_data.first_phase; test_data.first_phase])
    correlation_second = corspearman([betas_train; betas_test], [train_data.ages; test_data.ages])
    correlation_isi = corspearman([betas_train; betas_test], [train_data.insulin_sensitivity; test_data.insulin_sensitivity])

    markers=[:circle, :utriangle, :rect]

        COLORS_COMPARISON = Dict(
        "NGT" => "#4D6F64",
        "IGT" => "#A97B6D",
        "T2DM" => "#5C7FA3"
    )

    MAKERS = Dict(
        "NGT" => :circle,
        "IGT" => :utriangle,
        "T2DM" => :rect
    )
    MARKERSIZES = Dict(
        "NGT" => 15,
        "IGT" => 15,
        "T2DM" => 15
    )

    ax_first = Axis(fig[1,1], xlabel="βᵢ", ylabel= "1ˢᵗ Phase Clamp", title="ρ = $(round(correlation_first, digits=4))", backgroundcolor=:transparent)

    scatter!(ax_first, exp.(betas_train), train_data.first_phase, color = (:black, 0.2), markersize=10, label="Train Data", marker=:star5)
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        scatter!(ax_first, exp.(betas_test[type_indices]), test_data.first_phase[type_indices], color=COLORS_COMPARISON[type], label="Test $type", marker=MAKERS[type], markersize=MARKERSIZES[type])
    end

    Legend(fig[1,2], ax_first, orientation=:vertical, backgroundcolor=:transparent, framevisible=false)

    
    fig

end

save("figures/eccb/correlation_first_phase.svg", correlation_figure, px_per_unit=300/inch, backgroundcolor=:transparent)


# save("figures/other_betas.png", figure_other_betas, px_per_unit=4)

# #if MANUSCRIPT_FIGURES



#     model_fit_figure = let fig
#         fig = Figure(size = (linewidth, 6cm), fontsize=8pt)
#         ga = [GridLayout(fig[1,1], ), GridLayout(fig[1,2], ), GridLayout(fig[1,3], )]
#         gb = GridLayout(fig[1,4], nrow=1, ncol=1)
#         # do the simulations
#         sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
#         sols = [Array(solve(model.problem, p=ComponentArray(conditional=[betas_test[i]], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
        
#         axs = [Axis(ga[i][1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", title=type) for (i,type) in enumerate(unique(test_data.types))]

#         for (i,type) in enumerate(unique(test_data.types))

#             type_indices = test_data.types .== type

#             c_peptide_data = test_data.cpeptide[type_indices,:]

#             sol_idx = findfirst(objectives_test[type_indices] .== median(objectives_test[type_indices]))

#             # find the median fit of the type
#             sol_type = sols[type_indices][sol_idx]

#             lines!(axs[i], sol_timepoints, sol_type[:,1], color=(COLORS[type], 1), linewidth=1.5, label="Model fit", linestyle=:dot)
#             scatter!(axs[i], test_data.timepoints, c_peptide_data[sol_idx,:] , color=(COLORS[type], 1), markersize=5, label="Data")

#         end

#         linkyaxes!(axs...)

#         ax = Axis(gb[1,1], xticks=([0,1,2], ["NGT", "IGT", "T2DM"]), xlabel="Type", ylabel="MSE")

#         jitter_width = 0.1

#         for (i, type) in enumerate(unique(train_data.types))
#             jitter = rand(length(objectives_test)) .* jitter_width .- jitter_width/2
#             type_indices = test_data.types .== type
#             scatter!(ax, repeat([i-1], length(objectives_test[type_indices])) .+ jitter[type_indices] .- 0.1, objectives_test[type_indices], color=(COLORS[type], 0.8), markersize=3, label=type)
#             violin!(ax, repeat([i-1], length(objectives_test[type_indices])) .+ 0.05, objectives_test[type_indices], color=(COLORS[type], 0.8), width=0.75, side=:right, strokewidth=1, datalimits=(0,Inf))
#         end

#     #  boxplot!(ax, repeat([0], sum(test_data.types .== "NGT")), objectives_test[test_data.types .== "NGT"], color=COLORS["NGT"], width=0.75)
#     # boxplot!(ax, repeat([1], sum(test_data.types .== "IGT")),objectives_test[test_data.types .== "IGT"], color=COLORS["IGT"], width=0.75)
#     # boxplot!(ax, repeat([2], sum(test_data.types .== "T2DM")),objectives_test[test_data.types .== "T2DM"], color=COLORS["T2DM"], width=0.75)

#         Legend(fig[2,1:3], axs[1], orientation=:horizontal)



#     fig
#     end

#     save("figures/model_fit_test_median.$extension", model_fit_figure, px_per_unit=4)

#     model_fit_all_test = let fig
#         fig = Figure(size = (1000, 1500))
#         sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
#         sols = [Array(solve(model.problem, p=ComponentArray(conditional=betas_test[i], neural=neural_network_parameters[best_model_index]), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
        
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
#     

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