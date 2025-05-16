# Estimation with SAEM
include("../src/neural-network.jl")
using StableRNGs, CairoMakie, Distributions, JLD2, OrdinaryDiffEq, Optimization, OptimizationOptimisers, OptimizationOptimJL, ProgressMeter, LinearAlgebra, DataInterpolations, LineSearches
using ComponentArrays: ComponentArray
using SciMLBase: successful_retcode

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

function van_cauter_parameters(individual)
	age = individual.age
	t2dm = individual.condition == "T2DM"

    # set "van Cauter" parameters
    short_half_life = t2dm ? 4.52 : 4.95
    fraction = t2dm ? 0.78 : 0.76
    long_half_life = 0.14 * age + 29.2

    k1 = fraction * (log(2)/long_half_life) + (1-fraction) * (log(2)/short_half_life)
    k0 = (log(2)/short_half_life)*(log(2)/long_half_life)/k1
    k2 = (log(2)/short_half_life) + (log(2)/long_half_life) - k0 - k1

    return k0, k1, k2
end

function c_peptide_cude!(du, u, p, t, k0, k1, k2, glucose, c0, network)
    # production
    production = network([glucose(t); exp(p.conditional)], p.neural)[1] - network([0.0; exp(p.conditional)], p.neural)[1]

    du[1] = -k2*u[1] + k1*u[2] - k0 * (u[1] - c0) + production
    du[2] = k2*u[1] - k1*u[2]
end

function simulate(p_neural, p_individual, individual, network; timepoints = individual.timepoints)

    # van Cauter parameters
    k0, k1, k2 = van_cauter_parameters(individual)

    # glucose interpolation
    glucose = LinearInterpolation(individual.glucose, individual.timepoints)

    # initial conditions
    c0 = individual.cpeptide[1]
    u0 = [c0, (k2/k1)*c0]
    tspan = (timepoints[1], timepoints[end])

    # ODE problem
    prob = ODEProblem(
        (du, u, p, t) -> c_peptide_cude!(du, u, p, t, k0, k1, k2, glucose, c0, network),
        u0,
        tspan,
        ComponentArray(conditional = p_individual, neural = p_neural)
    )

    return solve(prob, saveat = timepoints, save_idxs = 1)
end

function individual_log_likelihood(p_individual, p_neural, individual, network, σ)

    y_pred = simulate(p_neural, p_individual, individual, network)

    if !successful_retcode(y_pred)
        # If the solver fails, return infinity
        return Inf
    end

    return -sum((individual.cpeptide - Array(y_pred)).^2) / (2 * σ^2)
end

function map_objective(p_individual, p_neural, individual, σ, Ω, network; prior_individual=0.0)
    ll = individual_log_likelihood(p_individual, p_neural, individual, network, σ)
    prior = logpdf(Normal(prior_individual, Ω), p_individual)
    return -(ll + prior) # negative for minimization
end

function compute_individual_maps(p_individuals, p_neural, individuals, σ, Ω, network; prior_individual=0.0)
    new_p_individuals = similar(p_individuals)

    optfunc = OptimizationFunction((x, (p_neural, individual, σ, Ω, network)) -> map_objective(x[1], p_neural, individual, σ, Ω, network; prior_individual=prior_individual), AutoForwardDiff())

    for (i, individual) in enumerate(individuals)
        result = Optimization.solve(OptimizationProblem(optfunc, [p_individuals[i]], (p_neural, individual, σ, Ω, network)), LBFGS())
        new_p_individuals[i] = result.u[1]
    end
    return new_p_individuals
end

function mcmc_step(p_individual, p_neural, individual, σ, Ω, proposal_std, network; prior_individual = 0.0)
    # Propose new parameters
    p_individual_new = p_individual + randn() * proposal_std

    # prior ratio
    prior_ratio = logpdf(Normal(prior_individual, Ω), p_individual_new) - logpdf(Normal(prior_individual, Ω), p_individual)

    # Compute the log-likelihood of the proposed parameters
    log_likelihood_new = individual_log_likelihood(p_individual_new, p_neural, individual, network, σ)

    # Compute the log-likelihood of the current parameters
    log_likelihood_current = individual_log_likelihood(p_individual, p_neural, individual, network, σ)

    # Compute the acceptance ratio
    likelihood_ratio = log_likelihood_new - log_likelihood_current

    # Accept or reject the proposed parameters
    if log(rand()) < (prior_ratio + likelihood_ratio)
        return p_individual_new, true
    else
        return p_individual, false
    end
end

function total_nll(p_individuals, p_neural, individuals, σ, network)
    nll = 0.0
    for (p_individual,individual) in zip(p_individuals, individuals)
        nll += -individual_log_likelihood(p_individual, p_neural, individual, network, σ)
    end
    return nll
end

function update_population_parameters(p_individuals, p_neural, individuals, σ, Ω, network)
    objective(p, (p_individuals, individuals, σ, Ω, network)) = total_nll(p_individuals, p, individuals, σ, network)
    optfunc = OptimizationFunction(objective, AutoForwardDiff())

    # training step 1 (Adam)
    optprob_train = OptimizationProblem(optfunc, p_neural, (p_individuals, individuals, σ, Ω, network))
    optsol_train = Optimization.solve(optprob_train, Optimisers.Adam(1e-2), maxiters=50)
    
    # training step 2 (LBFGS)
    optprob_train_2 = OptimizationProblem(optfunc, optsol_train.u, (p_individuals, individuals, σ, Ω, network))
    optsol_train_2 = Optimization.solve(optprob_train_2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=10)

    return optsol_train_2.u
end

function SAEM(
    individuals,
    initial_neural_params,
    prior_Ω,
    σ,
    network;
    initial_individuals = 0.0,
    iterations = 500,
    proposal_std = 0.1,
    n_burnin = 100,
    α = 0.7,
    initial_mcmc_steps = n_mcmc_steps,
    n_mcmc_steps = 1,
    target_acceptance_rate = 0.25
)

    println("Initializing the SAEM algorithm...")

    # initialize the random effect parameters
    p_individuals = fill(initial_individuals, length(individuals))
    p_neural = initial_neural_params
    Ω = copy(prior_Ω)

    # initialize diagnostic monitoring
    total_nll_values = Float64[]
    acceptance_rates = Float64[]

    # initialize the progress bar
    prog = Progress(iterations)

    for iteration in 1:iterations

        loglikelihood = 0.0
        gamma = iteration <= n_burnin ? 1.0 : 1.0 / (iteration - n_burnin)^α
        acceptance_count = 0
        n_mcmc_steps_iter = iteration <= n_burnin ? initial_mcmc_steps : n_mcmc_steps

        # loop over individuals
        for (i, individual) in enumerate(individuals)

            # MCMC steps
            at_least_one_accepted = false
            for step in 1:n_mcmc_steps_iter
                p_indiv_new, accepted = mcmc_step(p_individuals[i], p_neural, individual, σ, Ω, proposal_std, network; prior_individual = initial_individuals)
                if accepted
                    at_least_one_accepted = true
                end
                p_individuals[i] = (1 - gamma) * p_individuals[i] + gamma * p_indiv_new # stochastic update
            end
            acceptance_count += at_least_one_accepted ? 1 : 0
            loglikelihood += individual_log_likelihood(p_individuals[i], p_neural, individual, network, σ)

        end

        # update the population parameters
        p_neural_new = update_population_parameters(p_individuals, p_neural, individuals, σ, Ω, network)
        p_neural = (1 - gamma) .* p_neural .+ gamma .* p_neural_new

        # update the covariance matrix
        Ω = (1 - gamma) * Ω + gamma * var(p_individuals)

        # update diagnostic monitoring
        acceptance_rate = acceptance_count / (length(individuals))
        push!(total_nll_values, -loglikelihood)
        push!(acceptance_rates, acceptance_rate)

        # update proposal standard deviation
        log_proposal_std = log(proposal_std) + gamma * (acceptance_rate - target_acceptance_rate)
        proposal_std = iteration <= n_burnin ? proposal_std : clamp(exp(log_proposal_std), 1e-3, 1.0)

        # update progress bar
        next!(prog, showvalues = [("Total Negative Log-Likelihood", -loglikelihood), 
                                   ("Acceptance Rate", acceptance_rate),
                                   ("Proposal Std", proposal_std)])

    end

    return (
        p_neural = p_neural,
        p_individuals = p_individuals,
        Ω = Ω,
        total_nll_values = total_nll_values,
        acceptance_rates = acceptance_rates
    )
end


function FOCE(
    individuals,
    initial_neural_params,
    prior_Ω,
    σ,
    network;
    initial_individuals = 0.0,
    iterations = 50
)
    println("Initializing FOCE...")
    p_neural = initial_neural_params
    p_individuals = fill(initial_individuals, length(individuals))
    Ω = copy(prior_Ω)

    total_nll_values = Float64[]
    prog = Progress(iterations)

    for iter in 1:iterations
        # Step 1: Get MAP estimates of random effects
        p_individuals = compute_individual_maps(p_individuals, p_neural, individuals, σ, Ω, network; prior_individual=mean(p_individuals))

        # Step 2: Update neural parameters
        p_neural = update_population_parameters(p_individuals, p_neural, individuals, σ, Ω, network)

        # Step 3: Update variance of random effects
        Ω = var(p_individuals)

        # Track diagnostics
        push!(total_nll_values, total_nll(p_individuals, p_neural, individuals, σ, network))
        next!(prog, showvalues = [("Negative Log-Likelihood", total_nll_values[end])])
    end

    return (
        p_neural = p_neural,
        p_individuals = p_individuals,
        Ω = Ω,
        total_nll_values = total_nll_values
    )
end

# load the neural network
rng = StableRNG(232705)
network = chain(4, 2, tanh)


# pretrain with MLE on a random small subset of individuals
train_individuals_subset = rand(train_individuals, 25)
objective_mle(p, (individuals, σ, network)) = total_nll(p.individuals, p.neural, individuals, σ, network)

# initialize the neural network parameters and select the best 10 samples
initial_neural_params = [init_params(network, rng=rng) for _ in 1:2500]
initial_samples = [objective_mle(ComponentArray(neural = pars, individuals = zeros(length(train_individuals_subset))), (train_individuals_subset, 1.0, network)) for pars in initial_neural_params]

# top 5 samples
top_5_samples = partialsortperm(initial_samples,1:5)

optfunc_mle = OptimizationFunction(objective_mle, AutoForwardDiff())
results = []

# run MLE on the top 5 samples
for sample in top_5_samples
    try
        optprob_mle = OptimizationProblem(optfunc_mle, ComponentArray(neural = initial_neural_params[sample], individuals = zeros(length(train_individuals_subset))), (train_individuals_subset, 1.0, network))

        optsol_mle = Optimization.solve(optprob_mle, Optimisers.Adam(1e-3), maxiters=500)

        optprob_mle_2 = OptimizationProblem(optfunc_mle, optsol_mle.u, (train_individuals_subset, 1.0, network))

        optsol_mle_2 = Optimization.solve(optprob_mle_2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=500)

        push!(results, optsol_mle_2)
    catch
        println("Optimization failed for sample ", sample)
    end
end

# select the sample with the lowest objective function
best_sample_index = argmin([optsol.objective for optsol in results])
optsol_mle = results[best_sample_index]
std(optsol_mle.u.individuals)

# run FOCE on the best sample
foce_result = FOCE(
    train_individuals,
    Vector{Float64}(initial_neural_params[top_5_samples[1]]),
    0.5,
    0.25,
    network;
    initial_individuals = 0.0,
    iterations = 100
)

foce_result.p_individuals

# run SAEM on the best sample
# saem_result = SAEM(
#     train_individuals,
#     Vector{Float64}(optsol_mle.u.neural),
#     std(optsol_mle.u.individuals),
#     0.25,
#     network;
#     initial_individuals = mean(optsol_mle.u.individuals),
#     iterations = 100,
#     proposal_std = 0.3,
#     n_burnin = 20,
#     α = 0.7,
#     n_mcmc_steps = 10,
#     initial_mcmc_steps = 100,
#     target_acceptance_rate = 0.25
# )

COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
    )
# saem_result.p_individuals
# saem_result.p_population
figure = let f = Figure()
    ax1 = Axis(f[1,1], xlabel="Param", ylabel="Covariate")
    for (i, type) in enumerate(unique(train_data.types))
        type_indices = train_data.types .== type
        scatter!(ax1,saem_result.p_individuals[type_indices], train_data.first_phase[type_indices], color=(COLORS[type], 0.8), markersize=12, label=type)
    end
    
    f
end

figure_simulation = let f = Figure()
    ax1 = Axis(f[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title="Model fit")
    for (i, individual) in enumerate(train_individuals)
        y_pred = simulate(saem_result.p_neural, saem_result.p_individuals[i], individual, network; timepoints = individual.timepoints[1]:0.1:individual.timepoints[end])
        lines!(ax1, individual.timepoints[1]:0.1:individual.timepoints[end], Array(y_pred), color=(COLORS[individual.condition], 0.1), label=individual.subjectid)
    end
    scatter!(ax1, train_data.timepoints, mean(train_data.cpeptide, dims=1)[:], color=:black, markersize=12, label="Data")

    f
end