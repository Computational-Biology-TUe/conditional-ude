using StableRNGs, CairoMakie, Distributions, JLD2, OrdinaryDiffEq, Optimization, OptimizationOptimisers, OptimizationOptimJL, ProgressMeter, LinearAlgebra, DataInterpolations, LineSearches
using ComponentArrays: ComponentArray
using SciMLBase: successful_retcode
using StatsBase


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
    production = network([glucose(t)-glucose(0); exp(p.conditional)], p.neural)[1] - network([0.0; exp(p.conditional)], p.neural)[1]

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
        return -Inf
    end

    n_i = length(individual.timepoints)
    return -(n_i/2) * log(σ^2) - sum((individual.cpeptide - Array(y_pred)).^2) / (2 * σ^2)
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

function mcmc_step(p_individual, p_neural, individual, σ, Ω, proposal_std, network; prior_individual = 0.0, temperature = 1.0)
    # Propose new parameters
    p_individual_new = p_individual + randn() * proposal_std

    # prior ratio
    prior_ratio = logpdf(Normal(prior_individual, Ω), p_individual_new) - logpdf(Normal(prior_individual, Ω), p_individual)

    # Compute the log-likelihood of the proposed parameters
    log_likelihood_new = individual_log_likelihood(p_individual_new, p_neural, individual, network, σ)

    # Compute the log-likelihood of the current parameters
    log_likelihood_current = individual_log_likelihood(p_individual, p_neural, individual, network, σ)

    # Compute the acceptance ratio
    likelihood_ratio = log_likelihood_new/temperature - log_likelihood_current/temperature

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

function update_population_parameters(p_individuals, p_neural, individuals, σ, Ω, network; use_LBFGS = false)
    objective(p, (p_individuals, individuals, Ω, network)) = total_nll(p_individuals, p.neural, individuals, p.sigma, network)
    optfunc = OptimizationFunction(objective, AutoForwardDiff())
    optprob_train = OptimizationProblem(optfunc, ComponentArray(neural=p_neural, sigma=σ), (p_individuals, individuals, Ω, network))
    # training step 1 (Adam)
    if use_LBFGS
            # training step 2 (LBFGS)
        optsol_train = Optimization.solve(optprob_train, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=5)
        return optsol_train.u
    else
        optsol_train = Optimization.solve(optprob_train, Optimisers.Adam(1e-2), maxiters=5)
        return optsol_train.u
    end
end


function SAEM(
    individuals,
    initial_neural_params,
    network;
    σ = 1.0,
    prior_η = 0.0,
    prior_Ω = 1.0,
    iterations = 500,
    n_burnin_iterations = 100,
    proposal_std = 0.1,
    proposal_std_bounds = (1e-3, 1.0),
    α = 0.7,
    n_mcmc_steps = 1,
    initial_mcmc_steps = n_mcmc_steps,
    target_acceptance_rate = 0.25,
    initial_temperature = 10.0,
    temperature_decay = 0.05,
    Ω_learning_rate = 0.04,
)

    println("Initializing the SAEM algorithm...")

    # initialize the random effect parameters
    p_individuals = fill(prior_η, length(individuals))
    p_neural = initial_neural_params
    Ω = copy(prior_Ω)

    # initialize diagnostic monitoring
    total_nll_values = Float64[]
    acceptance_rates = Float64[]

    # initialize the progress bar
    prog = Progress(iterations, desc="Running SAEM")

    for iteration in 1:iterations

        loglikelihood = 0.0
        gamma = iteration <= n_burnin_iterations ? 1.0 : 1.0 / (iteration - n_burnin_iterations)^α
        acceptance_count = 0
        n_mcmc_steps_iter = iteration <= n_burnin_iterations ? initial_mcmc_steps : n_mcmc_steps
        temperature = max(1, initial_temperature * exp(-temperature_decay * iteration))

        # loop over individuals
        for (i, individual) in enumerate(individuals)

            # MCMC steps
            for step in 1:n_mcmc_steps_iter
                p_indiv_new, accepted = mcmc_step(p_individuals[i], p_neural, individual, σ, Ω, proposal_std, network; prior_individual = prior_η, temperature = temperature)
                if accepted
                    acceptance_count += 1
                end
                p_individuals[i] = (1 - gamma) * p_individuals[i] + gamma * p_indiv_new # stochastic update
            end
            loglikelihood += individual_log_likelihood(p_individuals[i], p_neural, individual, network, σ)

        end

        # update the population parameters
        if iteration <= n_burnin_iterations
            p_pop_new = update_population_parameters(p_individuals, p_neural, individuals, σ, Ω, network; use_LBFGS = false)
            p_neural_new = p_pop_new.neural
            σ = p_pop_new.sigma
        else
            p_pop_new = update_population_parameters(p_individuals, p_neural, individuals, σ, Ω, network)
            p_neural_new = p_pop_new.neural
            σ = p_pop_new.sigma
        end
        p_neural = (1 - gamma) .* p_neural .+ gamma .* p_neural_new

        # update the covariance matrix
        Ω = (1 - Ω_learning_rate) * Ω + Ω_learning_rate * var(p_individuals)
        prior_η = (1 - Ω_learning_rate) * prior_η + Ω_learning_rate * mean(p_individuals)

        # update diagnostic monitoring
        acceptance_rate = acceptance_count / (length(individuals)*n_mcmc_steps_iter)
        push!(total_nll_values, -loglikelihood)
        push!(acceptance_rates, acceptance_rate)
        fixed_effect_norm = sum(abs2, p_neural)
        random_effect_variance = var(p_individuals)

        # update proposal standard deviation
        log_proposal_std = log(proposal_std) + gamma * (acceptance_rate - target_acceptance_rate)
        proposal_std = iteration <= n_burnin_iterations ? proposal_std : clamp(exp(log_proposal_std), proposal_std_bounds[1], proposal_std_bounds[2])

        # update progress bar
        next!(prog, showvalues = [("Total Negative Log-Likelihood", -loglikelihood), 
                                   ("Acceptance Rate", acceptance_rate),
                                   ("Proposal Std", proposal_std),
                                   ("Fixed effect norm", fixed_effect_norm),
                                   ("Random effect variance", random_effect_variance),
                                   ("σ", σ)])

    end

    return (
        p_neural = p_neural,
        p_individuals = p_individuals,
        Ω = Ω,
        σ = σ,
        η = prior_η,
        total_nll_values = total_nll_values,
        acceptance_rates = acceptance_rates
    )
end