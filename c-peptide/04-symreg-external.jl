# Model fit to the train data and evaluation on the test data
extension = "svg"
inch = 96
pt = 4/3
cm = inch / 2.54
linewidth = 13.07245cm


COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
)

MANUSCRIPT_FIGURES = true
ECCB_FIGURES = true
        FONTS = (
    ; regular = "assets/fonts/Newsreader_9pt-Regular.ttf",
    bold = "assets/fonts/Newsreader_9pt-Bold.ttf",
    italic = "assets/fonts/Newsreader_9pt-Italic.ttf",
    bold_italic = "assets/fonts/Newsreader_9pt-BoldItalic.ttf",
)

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase

rng = StableRNG(232705)

include("../src/parameter-estimation.jl")
include("../src/utils.jl")
include("../src/likelihood-profiles.jl")

# Load the data
glucose_data, cpeptide_data, timepoints = jldopen("data/fujita.jld2", "r") do file
    file["glucose"], file["cpeptide"], Float64.(file["timepoints"])
end

# define the production function 
function production(ΔG, k)
    prod = ΔG >= 0 ? 1.78ΔG/(ΔG + k[1]) : 0.0
    return prod
end

# create the models
models = [
    CPeptideODEModel(glucose_data[i,:], timepoints, 29.0, production, cpeptide_data[i,:], false) for i in axes(glucose_data, 1)
]

optsols = OptimizationSolution[]
optfunc = OptimizationFunction(loss_sigma, AutoForwardDiff())
for (i,model) in enumerate(models)

    optprob = OptimizationProblem(optfunc, ComponentArray(ode=[40.0], sigma=1.0), (model, timepoints, cpeptide_data[i,:]),
    lb = 0.0, ub = 1000.0)
    optsol = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000)
    push!(optsols, optsol)
end

betas = [optsol.u.ode[1] for optsol in optsols]
sigmas = [optsol.u.sigma for optsol in optsols]
objectives = ([optsol.objective for optsol in optsols].- (length(timepoints)/2) .* log.(sigmas.^2)) .* (2 .* sigmas.^2)

function argmedian(x)
    return argmin(abs.(x .- median(x)))
end

function argquantile(x, q)
    return argmin(abs.(x .- quantile(x, q)))
end

if MANUSCRIPT_FIGURES

    model_fit_figure = let f = Figure(size=(linewidth,6cm), fontsize=10pt, fonts=FONTS)

        ga = GridLayout(f[1,1:2])
        gb = GridLayout(f[1,3:4])
        gc = GridLayout(f[1,5:6])

        sol_timepoints = timepoints[1]:0.1:timepoints[end]
        sols = [Array(solve(model.problem, p=[beta], saveat=sol_timepoints, save_idxs=1)) for (model,beta) in zip(models, betas)]

        ax1 = Axis(gb[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title="50%") 

        median_index = argmedian(objectives)
        lquantile_index = argquantile(objectives, 0.25)
        uquantile_index = argquantile(objectives, 0.75)

        lines!(ax1, sol_timepoints, sols[median_index], color=COLORS["NGT"], linestyle=:solid, linewidth=2, label="Model")
        scatter!(ax1, timepoints, cpeptide_data[median_index,:], color=:black, markersize=5, label="Data")

        # run PLA
        loss_values, loss_minimum, parameter_values = likelihood_profile(
            betas[median_index], loss, (models[median_index], timepoints, cpeptide_data[median_index,:]), betas[median_index]-25, betas[median_index]+1000, sigmas[median_index]; steps=10_000
        )
        min_parameter_value, max_parameter_value = find_confidence_intervals(loss_values, loss_minimum, parameter_values; target=:cantelli95)
        println(min_parameter_value, max_parameter_value)
        if !isinf(min_parameter_value)
            # compute the solution with the lower and upper bounds
            sol_lower = Array(solve(models[median_index].problem, p=[min_parameter_value], saveat=sol_timepoints, save_idxs=1))
            lines!(ax1, sol_timepoints, sol_lower[:,1], color=(COLORS["NGT"], 0.5), linewidth=1.5, label="95% CI", linestyle=:dot)
        end 
        if !isinf(max_parameter_value)
            sol_upper = Array(solve(models[median_index].problem, p=[max_parameter_value], saveat=sol_timepoints, save_idxs=1))

            lines!(ax1, sol_timepoints, sol_upper[:,1], color=(COLORS["NGT"], 0.5), linewidth=1.5, label="95% CI", linestyle=:dot)
        end

        ax2 = Axis(ga[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title="25%")
        lines!(ax2, sol_timepoints, sols[lquantile_index], color=COLORS["NGT"], linestyle=:solid, linewidth=2, label="Model")
        scatter!(ax2, timepoints, cpeptide_data[lquantile_index,:], color=:black, markersize=5, label="Data")

        # run PLA
        loss_values, loss_minimum, parameter_values = likelihood_profile(
            betas[lquantile_index], loss, (models[lquantile_index], timepoints, cpeptide_data[lquantile_index,:]), betas[lquantile_index]-25, betas[lquantile_index]+1000, sigmas[lquantile_index]; steps=10_000
        )
        min_parameter_value, max_parameter_value = find_confidence_intervals(loss_values, loss_minimum, parameter_values; target=:cantelli95)
        println(min_parameter_value, max_parameter_value)
        if !isinf(min_parameter_value)
            # compute the solution with the lower and upper bounds
            sol_lower = Array(solve(models[lquantile_index].problem, p=[min_parameter_value], saveat=sol_timepoints, save_idxs=1))
            lines!(ax2, sol_timepoints, sol_lower[:,1], color=(COLORS["NGT"], 0.5), linewidth=1.5, label="95% CI", linestyle=:dot)
        end 
        if !isinf(max_parameter_value)
            sol_upper = Array(solve(models[lquantile_index].problem, p=[max_parameter_value], saveat=sol_timepoints, save_idxs=1))

            lines!(ax2, sol_timepoints, sol_upper[:,1], color=(COLORS["NGT"], 0.5), linewidth=1.5, label="95% CI", linestyle=:dot)
        end

        ax3 = Axis(gc[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title="75%")
        lines!(ax3, sol_timepoints, sols[uquantile_index], color=COLORS["NGT"], linestyle=:solid, linewidth=2, label="Model")
        scatter!(ax3, timepoints, cpeptide_data[uquantile_index,:], color=:black, markersize=5, label="Data")

        # run PLA
        loss_values, loss_minimum, parameter_values = likelihood_profile(
            betas[uquantile_index], loss, (models[uquantile_index], timepoints, cpeptide_data[uquantile_index,:]), betas[uquantile_index]-25, betas[uquantile_index]+1000, sigmas[uquantile_index]; steps=10_000
        )
        min_parameter_value, max_parameter_value = find_confidence_intervals(loss_values, loss_minimum, parameter_values; target=:cantelli95)
        println(min_parameter_value, max_parameter_value)
        if !isinf(min_parameter_value)
            # compute the solution with the lower and upper bounds
            sol_lower = Array(solve(models[uquantile_index].problem, p=[min_parameter_value], saveat=sol_timepoints, save_idxs=1))
            lines!(ax3, sol_timepoints, sol_lower[:,1], color=(COLORS["NGT"], 0.5), linewidth=1.5, label="95% CI", linestyle=:dot)
        end 
        if !isinf(max_parameter_value)
            sol_upper = Array(solve(models[uquantile_index].problem, p=[max_parameter_value], saveat=sol_timepoints, save_idxs=1))

            lines!(ax3, sol_timepoints, sol_upper[:,1], color=(COLORS["NGT"], 0.5), linewidth=1.5, label="95% CI", linestyle=:dot)
        end


        linkyaxes!(ax1, ax2, ax3)
        Legend(f[2,1:6], ax1, orientation=:horizontal, merge=true)

        gd = GridLayout(f[1,7])
        ax = Axis(gd[1,1], ylabel="Error", xticks=([],[]))
        jitter_width = 0.1
        #boxplot!(ax, repeat([1], length(objectives)), objectives, color=COLORS["NGT"], strokewidth=2, width=0.5)
        jitter = rand(length(objectives)) .* jitter_width .- jitter_width/2
        #type_indices = [train_data.types .== type; test_data.types .== type]
        scatter!(ax, repeat([0], length(objectives)) .+ jitter .- 0.05, objectives, color=(COLORS["NGT"], 0.8), markersize=3)
        violin!(ax, repeat([0], length(objectives)) .+ 0.05, objectives, color=(COLORS["NGT"], 0.8), width=0.5, side=:right, strokewidth=1, datalimits=(0,Inf))

        for (label, layout) in zip(["a", "b", "c", "d"], [ga, gb, gc, gd])
            Label(layout[1, 1, TopLeft()], label,
            fontsize = 12pt,
            font = :bold,
            padding = (0, 20, 8, 0),
            halign = :right)
        end

        f
    end

    save("figures/model_fit_external.$extension", model_fit_figure, px_per_unit=300/inch)
end

#if ECCB_FIGURES


    # ECCB submission
        COLORS = Dict(
        "NGT" => "#4D6F64",
        "IGT" => "#A97B6D",
        "T2DM" => "#5C7FA3"
    )

    COLORS_2 = Dict(
        "NGT" => RGBf(205/255, 234/255, 235/255),
        "IGT" => RGBf(5/255, 149/255, 154/255),
        "T2DM" => RGBf(3/255, 75/255, 77/255)
    )

    pagewidth = 21cm
    margin = 0.02 * pagewidth

    textwidth = pagewidth - 2 * margin
    aspect = 1

    model_fit_figure = let f = Figure(size=(16cm,10cm), fontsize=20pt, fonts=FONTS, backgroundcolor=:transparent)

        sol_timepoints = timepoints[1]:0.1:timepoints[end]
        sols = [Array(solve(model.problem, p=[beta], saveat=sol_timepoints, save_idxs=1)) for (model,beta) in zip(models, betas)]

    ax1 = Axis(f[1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=true, ygridvisible=true, topspinevisible=false, rightspinevisible=false, spinewidth=0.05cm, bottomspinecolor=background_color, leftspinecolor=background_color, xtickcolor=background_color, ytickcolor=background_color, xticklabelcolor=background_color, yticklabelcolor=background_color, xgridcolor=(background_color,0.3), ygridcolor=(background_color,0.3))

        median_index = argmedian(objectives)
        #lquantile_index = argquantile(objectives, 0.25)
        uquantile_index = argquantile(objectives, 0.3)

        # lines!(ax1, sol_timepoints, sols[median_index], color=COLORS["T2DM"], linestyle=:solid, linewidth=0.15cm, label="Model")
        # scatter!(ax1, timepoints, cpeptide_data[median_index,:], color=COLORS["T2DM"], markersize=0.5cm, label="Data")

        # lines!(ax1, sol_timepoints, sols[lquantile_index], color=COLORS["IGT"], linestyle=:solid, linewidth=0.15cm, label="Model")
        # scatter!(ax1, timepoints, cpeptide_data[lquantile_index,:], color=COLORS["IGT"], markersize=0.5cm, label="Data")

        lines!(ax1, sol_timepoints, sols[uquantile_index], color=COLORS["NGT"], linestyle=:solid, linewidth=0.15cm, label="Model")
        scatter!(ax1, timepoints, cpeptide_data[uquantile_index,:], color=COLORS["NGT"], markersize=0.5cm, label="Data")

        vspan!(ax1, 0, 120, color=(:black, 0.1), label="Train time")
        Legend(f[1,2], ax1, orientation=:vertical, merge=true, backgroundcolor=:transparent, framevisible=false, labelfont=:bold, title="C-peptide", titlefont=:bold, titlefontsize=8pt, labelfontsize=8pt)


        f
    end

    save("figures/eccb/external_poster.$extension", model_fit_figure, px_per_unit=600/inch)
#end




