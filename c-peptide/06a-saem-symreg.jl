include("../src/saem-symreg.jl")

# load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
	file["train"], file["test"]
end;

train_individuals = [(
	subjectid = train_data.subject_numbers[i],
	cpeptide = train_data.cpeptide[i,:],
	glucose = train_data.glucose[i,:],
	age = train_data.ages[i],
	condition = train_data.types[i],
	timepoints = train_data.timepoints
) for i in eachindex(train_data.ages)];

test_individuals = [(
	subjectid = test_data.subject_numbers[i],
	cpeptide = test_data.cpeptide[i,:],
	glucose = test_data.glucose[i,:],
	age = test_data.ages[i],
	condition = test_data.types[i],
	timepoints = test_data.timepoints
) for i in eachindex(test_data.ages)];

all_individuals = vcat(train_individuals, test_individuals);

# run SAEM on the best sample
saem_result = SAEM(
    all_individuals,
    75.0;
    prior_Ω = 0.1,
    σ = 0.1,
    iterations = 100,
    n_burnin_iterations = 30,
    proposal_std = 0.4,
    proposal_std_bounds = (1e-3, 1.0),
    α = 0.7,
    n_mcmc_steps = 1,
    initial_mcmc_steps = 20,
    target_acceptance_rate = 0.25,
    initial_temperature = 1.0,
    temperature_decay = 0.2,
    Ω_learning_rate = 1.0,
)

# obtain the MAP estimates of the individuals
individual_modes = []
mle_estimates = []
all_samples = []
prog = Progress(length(train_individuals) + length(test_individuals); desc="Computing individual effects...")
for (i, individual) in enumerate(all_individuals)
    prior_Ω = saem_result.Ω
    km_pop = saem_result.km_pop

    individual_samples = []
    p_individual = 0.0
    for _ in 1:3000
        p_individual, accepted = mcmc_step(p_individual, km_pop, individual, saem_result.σ, prior_Ω, 0.3)
        push!(individual_samples, p_individual)
    end

    # Compute the MAP estimate
    map_opt = OptimizationFunction((x,_) -> map_objective(km_pop, individual, saem_result.σ, x[1], prior_Ω), AutoForwardDiff())
    map_prob = OptimizationProblem(map_opt, [p_individual])
    map_sol = Optimization.solve(map_prob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=100)
    mode = map_sol.u[1]
    push!(individual_modes, mode)

    # Compute the MLE estimate
    mle_opt = OptimizationFunction((x,_) -> -1*individual_log_likelihood(map_individual(km_pop, x[1]), individual, saem_result.σ), AutoForwardDiff())
    mle_prob = OptimizationProblem(mle_opt, [p_individual])
    mle_sol = Optimization.solve(mle_prob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=100)
    mle_estimate = mle_sol.u[1]
    push!(mle_estimates, mle_estimate)

    push!(all_samples, individual_samples)
    next!(prog)
end

COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
    )

MARKERS = Dict(
    "T2DM" => :circle,
    "NGT" => :diamond,
    "IGT" => :rect
)

figure_mle_map = let f = Figure()
    ax = Axis(f[1,1], xlabel="MLE", ylabel="MAP", title="SAEM: MAP vs MLE")
    for (i, type) in enumerate(unique([train_data.types; test_data.types]))
        type_indices = [train_data.types; test_data.types] .== type
        scatter!(ax, exp.(mle_estimates[type_indices]), exp.(individual_modes[type_indices]), color=(COLORS[type], 0.1), strokewidth=2, strokecolor=(COLORS[type], 1.0), markersize=12, marker=MARKERS[type], label="$(type)")
    end
    lines!(ax, 0:0.1:20, 0:0.1:20, color=:black, linestyle=:dash, label="y=x", linewidth=2)

    ax_inset = Axis(f[1, 1],
    width=Relative(0.35),
    height=Relative(0.35),
    halign=0.1,
    valign=0.9,
    title="Zoomed View")

    xlims!(ax_inset, 0, 2)
    ylims!(ax_inset, 0, 2)
    for (i, type) in enumerate(unique([train_data.types; test_data.types]))
        type_indices = [train_data.types; test_data.types] .== type
        scatter!(ax_inset, exp.(mle_estimates[type_indices]), exp.(individual_modes[type_indices]), color=(COLORS[type], 0.1), strokewidth=1, strokecolor=(COLORS[type], 0.8), markersize=6, marker=MARKERS[type], label="$(type)")
    end
    lines!(ax_inset, 0:0.1:3, 0:0.1:3, color=:black, linestyle=:dash, label="y=x", linewidth=2)

    lines!(ax, [0, 2, 2, 0, 0], [0, 0, 2, 2, 0], color=:black, linestyle=:dot, label="Bounds", linewidth=1.5)


    f
end

figure = let f = Figure()
    ax1 = Axis(f[1,1], xlabel="Param", ylabel="Covariate", title="SAEM: ρ = $(round(StatsBase.corspearman(Float64.(mle_estimates), [train_data.first_phase; test_data.first_phase]), digits=4))")
    for (i, type) in enumerate(unique([train_data.types; test_data.types]))
        type_indices_train = train_data.types .== type
        type_indices_test = test_data.types .== type

        train_modes = mle_estimates[1:length(train_individuals)]
        test_modes = mle_estimates[length(train_individuals)+1:end]
        train_parameters = map_individual.(saem_result.km_pop, train_modes)
        scatter!(ax1,log10.(train_parameters[type_indices_train]), train_data.first_phase[type_indices_train], color=(COLORS[type], 0.3), markersize=12, label="Train Data", marker=:utriangle,strokecolor=(COLORS[type], 1.0), strokewidth=2)

        test_parameters = map_individual.(saem_result.km_pop, test_modes)
        scatter!(ax1,log10.(test_parameters[type_indices_test]), test_data.first_phase[type_indices_test], color=(COLORS[type], 0.3), strokewidth=2, strokecolor=(COLORS[type], 1.0), markersize=12, marker=MARKERS[type], label="Test $(type)")
    end
    Legend(f[2,1], ax1, merge=true, orientation=:horizontal)
    f
end

test_modes = individual_modes[length(train_individuals)+1:end]
test_mle_estimates = mle_estimates[length(train_individuals)+1:end]
test_samples = all_samples[length(train_individuals)+1:end]
figure_simulation = let f = Figure(size=(1200, 1200))
    axs = []
    prog = Progress(length(test_individuals); desc="Simulating individuals")
    for (i, individual) in enumerate(test_individuals)

        position = (i-1) % 5 + 1
        row = (i-1) ÷ 5 + 1

        ax = Axis(f[row,position], xlabel="Time", ylabel="C-peptide")
        #lines!(ax, individual.timepoints[1]:0.1:individual.timepoints[end], Array(y_pred), color=(COLORS[individual.condition], 0.9), label=individual.subjectid)
        for sample in test_samples[i][1000:5:end]
            kM_indiv = map_individual(saem_result.km_pop, sample)
            y_pred_sample = simulate(kM_indiv, individual; timepoints = individual.timepoints[1]:0.1:individual.timepoints[end])
            lines!(ax, individual.timepoints[1]:0.1:individual.timepoints[end], Array(y_pred_sample), color=(:grey, 0.02), label="Sample")
        end
        mle_estimate = map_individual(saem_result.km_pop, test_mle_estimates[i])
        map_estimate = map_individual(saem_result.km_pop, test_modes[i])
        y_pred_mle = simulate(mle_estimate, individual; timepoints = individual.timepoints[1]:0.1:individual.timepoints[end])
        lines!(ax, individual.timepoints[1]:0.1:individual.timepoints[end], Array(y_pred_mle), color=(COLORS[individual.condition], 1.0), linestyle=:dash, label="MLE", linewidth=2)
        y_pred_map = simulate(map_estimate, individual; timepoints = individual.timepoints[1]:0.1:individual.timepoints[end])
        lines!(ax, individual.timepoints[1]:0.1:individual.timepoints[end], Array(y_pred_map), color=(COLORS[individual.condition], 1.0), linestyle=:dot, label="MAP", linewidth=2)

        scatter!(ax, individual.timepoints, individual.cpeptide, color=(COLORS[individual.condition], 0.7), strokewidth=1.5, strokecolor=(COLORS[individual.condition], 1.0), markersize=15, label="Data", marker=:diamond)
        push!(axs, ax)
        next!(prog)
    end

    Legend(f[8,1:5],axs[1], merge=true, orientation=:horizontal)

    f
end