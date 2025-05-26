# Estimation with SAEM
include("../src/neural-network.jl")
include("../src/saem.jl")

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


# load the neural network
rng = StableRNG(232705)
network = chain(4, 2, tanh)

# pretrain with MLE on a random small subset of individuals
println("Running MLE on a small subset of individuals...")

train_individuals_subset = [rand(rng, train_individuals[train_data.types .== "NGT"], 5); rand(rng, train_individuals[train_data.types .== "IGT"], 5); rand(rng, train_individuals[train_data.types .== "T2DM"], 5)]

objective_mle(p, (individuals, σ, network)) = total_nll(p.individuals, p.neural, individuals, σ, network)

# initialize the neural network parameters and select the best sample
initial_neural_params = [init_params(network, rng=rng) for _ in 1:2500]
initial_samples = [objective_mle(ComponentArray(neural = pars, individuals = zeros(length(train_individuals_subset))), (train_individuals_subset, 1.0, network)) for pars in initial_neural_params]

# top 5 samples
println("Selecting the best 15 samples...")
top_15_samples = partialsortperm(initial_samples,1:15)

optfunc_mle = OptimizationFunction(objective_mle, AutoForwardDiff())
results = []

prog_mle = Progress(length(top_15_samples); desc="Running MLE on top 15 samples")
# run MLE on the top 15 samples
for sample in top_15_samples
    try
        optprob_mle = OptimizationProblem(optfunc_mle, ComponentArray(neural = initial_neural_params[sample], individuals = zeros(length(train_individuals_subset))), (train_individuals_subset, 1.0, network))

        optsol_mle_1 = Optimization.solve(optprob_mle, Optimisers.Adam(1e-3), maxiters=500)

        optprob_mle_2 = OptimizationProblem(optfunc_mle, optsol_mle_1.u, (train_individuals_subset, 1.0, network))

        optsol_mle_2 = Optimization.solve(optprob_mle_2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=500)

        push!(results, optsol_mle_2)
    catch
        println("Optimization failed for sample ", sample)
    end
    next!(prog_mle)
end

println("MLE optimization completed.")
# select the sample with the lowest objective function
best_sample_index = argmin([optsol.objective for optsol in results])
optsol_mle = results[best_sample_index]

# run SAEM on the best sample
saem_result = SAEM(
    train_individuals,
    optsol_mle.u.neural,
    network;
    σ = 0.5,
    prior_η = mean(optsol_mle.u.individuals),
    prior_Ω = 20*var(optsol_mle.u.individuals),
    iterations = 180,
    n_burnin_iterations = 80,
    proposal_std = 0.8,
    proposal_std_bounds = (1e-3, 10.0),
    α = 0.7,
    n_mcmc_steps = 25,
    initial_mcmc_steps = 25,
    target_acceptance_rate = 0.35,
    initial_temperature = 2.0,
    temperature_decay = 0.2,
    Ω_learning_rate = 0.04,
)

# obtain the MAP estimates of the individuals
individual_modes = []
mle_estimates = []
all_samples = []
mse_values = []
prog = Progress(length(train_individuals) + length(test_individuals); desc="Computing individual effects...")
for (i, individual) in enumerate([train_individuals; test_individuals])
    prior = saem_result.η
    prior_Ω = saem_result.Ω

    neural_params = saem_result.p_neural

    individual_samples = []
    p_individual = prior
    for _ in 1:3000
        p_individual, accepted = mcmc_step(p_individual, neural_params, individual, saem_result.σ, prior_Ω, 0.3, network; prior_individual = prior)
        push!(individual_samples, p_individual)
    end

    # Compute the MAP estimate
    map_opt = OptimizationFunction((x,_) -> map_objective(x[1], neural_params, individual, saem_result.σ, prior_Ω, network; prior_individual=prior), AutoForwardDiff())
    map_prob = OptimizationProblem(map_opt, [p_individual], (neural_params, individual, saem_result.σ, prior_Ω, network))
    map_sol = Optimization.solve(map_prob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=100)
    mode = map_sol.u[1]
    push!(individual_modes, mode)

    # Compute the MLE estimate
    mle_opt = OptimizationFunction((x,_) -> -1*individual_log_likelihood(x[1], neural_params, individual, network, saem_result.σ), AutoForwardDiff())
    mle_prob = OptimizationProblem(mle_opt, [p_individual], (neural_params, individual, saem_result.σ, network))
    mle_sol = Optimization.solve(mle_prob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=100)
    mle_estimate = mle_sol.u[1]
    push!(mle_estimates, mle_estimate)

    # Compute the mean squared error
    mse = -2 * individual_log_likelihood(mode, neural_params, individual, network, 1.0)
    #push!(individual_modes, mode)
    push!(mse_values, mse)
    push!(all_samples, individual_samples)
    next!(prog)
end

for type in unique(train_data.types)
    type_indices = [train_data.types; test_data.types] .== type
    println("Type: ", type)
    println("MSE: ", mean(mse_values[type_indices]))
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
# saem_result.p_individuals
# saem_result.p_population
figure = let f = Figure()
    ax1 = Axis(f[1,1], xlabel="exp(ηᵢ)", ylabel="1ˢᵗ Phase Clamp", title="SAEM: ρ = $(round(StatsBase.corspearman(Float64.(individual_modes), [train_data.first_phase; test_data.first_phase]), digits=4))")
    for (i, type) in enumerate(unique([train_data.types; test_data.types]))
        type_indices_train = train_data.types .== type
        type_indices_test = test_data.types .== type

        train_modes = individual_modes[1:length(train_individuals)]
        test_modes = individual_modes[length(train_individuals)+1:end]

        scatter!(ax1,exp.(train_modes[type_indices_train]), train_data.first_phase[type_indices_train], color=(:black, 0.2), markersize=12, label="Train Data", marker=:utriangle)

        scatter!(ax1,exp.(test_modes[type_indices_test]), test_data.first_phase[type_indices_test], color=(COLORS[type], 0.3), strokewidth=2, strokecolor=(COLORS[type], 1.0), markersize=12, marker=MARKERS[type], label="Test $(type)")
    end
    Legend(f[2,1], ax1, merge=true, orientation=:horizontal)
    f
end

save("SAEM_correation.png", figure)
length(train_individuals[1:15:end])
individual_modes
test_modes = individual_modes[length(train_individuals)+1:end]
test_samples = all_samples[length(train_individuals)+1:end]
figure_simulation = let f = Figure(size=(1200, 1200))
    axs = []
    prog = Progress(length(test_individuals); desc="Simulating individuals")
    for (i, individual) in enumerate(test_individuals)

        position = (i-1) % 5 + 1
        row = (i-1) ÷ 5 + 1

        ax = Axis(f[row,position], xlabel="Time", ylabel="C-peptide")
        #lines!(ax, individual.timepoints[1]:0.1:individual.timepoints[end], Array(y_pred), color=(COLORS[individual.condition], 0.9), label=individual.subjectid)
        for sample in test_samples[i][1000:10:end]
            y_pred_sample = simulate(saem_result.p_neural, sample, individual, network; timepoints = individual.timepoints[1]:0.1:individual.timepoints[end])
            lines!(ax, individual.timepoints[1]:0.1:individual.timepoints[end], Array(y_pred_sample), color=(COLORS[individual.condition], 0.01), label="Sample")
        end

        # compute the likelihood values of the samples
        likelihood_values = [individual_log_likelihood(sample, saem_result.p_neural, individual, network, saem_result.σ) for sample in test_samples[i][1000:end]]
        map_objective_values = [map_objective(sample, saem_result.p_neural, individual, saem_result.σ, saem_result.Ω, network; prior_individual=saem_result.η) for sample in test_samples[i][1000:end]]

        # find the MLE and MAP samples
        mle_opt = OptimizationFunction((x,_) -> -1*individual_log_likelihood(x[1], saem_result.p_neural, individual, network, saem_result.σ), AutoForwardDiff())
        mle_prob = OptimizationProblem(mle_opt, [test_samples[i][argmax(likelihood_values)]], (saem_result.p_neural, individual, saem_result.σ, network))
        mle_sol = Optimization.solve(mle_prob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=100)

        map_opt = OptimizationFunction((x,_) -> map_objective(x[1], saem_result.p_neural, individual, saem_result.σ, saem_result.Ω, network; prior_individual=saem_result.η), AutoForwardDiff())
        map_prob = OptimizationProblem(map_opt, [test_samples[i][argmin(map_objective_values)]], (saem_result.p_neural, individual, saem_result.σ, saem_result.Ω, network))
        map_sol = Optimization.solve(map_prob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=100)

        y_pred_mle = simulate(saem_result.p_neural, mle_sol.u[1], individual, network; timepoints = individual.timepoints[1]:0.1:individual.timepoints[end])
        lines!(ax, individual.timepoints[1]:0.1:individual.timepoints[end], Array(y_pred_mle), color=:black, linestyle=:dash, label="MLE", linewidth=2)
        y_pred_map = simulate(saem_result.p_neural, map_sol.u[1], individual, network; timepoints = individual.timepoints[1]:0.1:individual.timepoints[end])
        lines!(ax, individual.timepoints[1]:0.1:individual.timepoints[end], Array(y_pred_map), color=:black, linestyle=:dot, label="MAP", linewidth=2)

        scatter!(ax, individual.timepoints, individual.cpeptide, color=(COLORS[individual.condition], 0.4), strokewidth=1.5, strokecolor=(:black, 1.0), markersize=12, label="Data", marker=:diamond)
        push!(axs, ax)
        next!(prog)
    end

    Legend(f[8,1:5],axs[1], merge=true, orientation=:horizontal)

    f
end

save("SAEM_simulation.png", figure_simulation)

# save beta results
glucose_ranges = 0.0:0.1:maximum([train_data.glucose; test_data.glucose]) .- minimum([train_data.glucose; test_data.glucose])
beta_ranges = exp.(minimum(individual_modes):0.01:maximum(individual_modes))

neural_simulations = [network([glucose; beta], saem_result.p_neural)[1] for glucose in glucose_ranges, beta in beta_ranges]

# convert to long format
neural_simulations_long = [[glucose, beta, neural_simulations[i,j]] for (i, glucose) in enumerate(glucose_ranges), (j, beta) in enumerate(beta_ranges)]

# convert to 1D array of arrays
neural_simulations_long = reshape(neural_simulations_long, length(neural_simulations_long))

hcat(neural_simulations_long...)

using DataFrames
df = DataFrame(hcat(neural_simulations_long...)', [:glucose, :beta, :cpeptide])

using CSV

CSV.write("neural_simulations.csv", df)
minimum(individual_modes), maximum(individual_modes)
figure_dose_response = let f = Figure()
    ax = Axis(f[1,1], xlabel="Glucose", ylabel="C-peptide Production", title="SAEM: Dose-Response")
    glucose_range = 0.0:0.1:(maximum([train_data.glucose; test_data.glucose])-minimum([train_data.glucose; test_data.glucose]))
    for beta in range(minimum(individual_modes), maximum(individual_modes), length=25)
        y_pred = [network([g; exp(beta)], saem_result.p_neural)[1] - network([0; exp(beta)], saem_result.p_neural)[1] for g in glucose_range]
        lines!(ax, glucose_range, y_pred, color=beta, colorrange=(minimum(individual_modes), maximum(individual_modes)))
    end
    f
end

