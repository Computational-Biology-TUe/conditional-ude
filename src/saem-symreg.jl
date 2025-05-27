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

function c_peptide_ode!(du, u, p, t, k0, k1, k2, glucose, c0)
    # production
    production = 1.78 * (glucose(t) - glucose(0)) / (glucose(t) - glucose(0) + p)

    du[1] = -k2*u[1] + k1*u[2] - k0 * (u[1] - c0) + production * (glucose(t) > glucose(0) ? 1.0 : 0.0)
    du[2] = k2*u[1] - k1*u[2]
end

function simulate(kM_indiv, individual; timepoints = individual.timepoints)
	# map parameters
	k0, k1, k2 = van_cauter_parameters(individual)

	# interpolate glucose
	gi = LinearInterpolation(individual.glucose, individual.timepoints)

	# initial cpeptide
	c0 = individual.cpeptide[1]

	# initial conditions
	u0 = [c0, (k2/k1)*c0]

	# problem definition
	prob = ODEProblem((du, u, p, t) -> c_peptide_ode!(
		du, u, p, t, k0, k1, k2, gi, c0), u0, (timepoints[1], timepoints[end]), kM_indiv)

	return solve(prob, saveat=timepoints, save_idxs=1)
end

function map_individual(km_pop, eta)
	km_pop * exp(eta)
end

function individual_log_likelihood(kM_indiv, individual, σ)

	y_pred = simulate(kM_indiv, individual)

    # if !successful_retcode(y_pred)
    #     # If the solver fails, return infinity
    #     return -Inf
    # end

    n_i = length(individual.timepoints)
    return -(n_i/2) * log(σ^2) - sum((individual.cpeptide - Array(y_pred)).^2) / (2 * σ^2)
end

function map_objective(km_pop, individual, σ, eta, Ω)
    kM_indiv = map_individual(km_pop, eta)
    ll = individual_log_likelihood(kM_indiv, individual, σ)
    prior = logpdf(Normal(0.0, Ω), eta)
    return -(ll + prior) # negative for minimization
end

# function compute_individual_maps(p_individuals, p_neural, individuals, σ, Ω, network; prior_individual=0.0)
#     new_p_individuals = similar(p_individuals)

#     optfunc = OptimizationFunction((x, (p_neural, individual, σ, Ω, network)) -> map_objective(x[1], p_neural, individual, σ, Ω, network; prior_individual=prior_individual), AutoForwardDiff())

#     for (i, individual) in enumerate(individuals)
#         result = Optimization.solve(OptimizationProblem(optfunc, [p_individuals[i]], (p_neural, individual, σ, Ω, network)), LBFGS())
#         new_p_individuals[i] = result.u[1]
#     end
#     return new_p_individuals
# end

function mcmc_step(η_current, km_pop, individual, σ, Ω, proposal_std; temperature = 1.0)
    # Propose new parameters
    η_proposed = η_current + randn() * proposal_std

    km_current = map_individual(km_pop, η_current)
    km_proposed = map_individual(km_pop, η_proposed)

    # prior ratio
    prior_ratio = logpdf(Normal(0, Ω), η_proposed) - logpdf(Normal(0, Ω), η_current)

    # Compute the log-likelihood of the proposed parameters
    log_likelihood_new = individual_log_likelihood(km_proposed, individual, σ)

    # Compute the log-likelihood of the current parameters
    log_likelihood_current = individual_log_likelihood(km_current, individual, σ)

    # Compute the acceptance ratio
    likelihood_ratio = log_likelihood_new/temperature - log_likelihood_current/temperature

    # Accept or reject the proposed parameters
    if log(rand()) < (prior_ratio + likelihood_ratio)
        return η_proposed, true
    else
        return η_current, false
    end
end

function total_nll(km_pop, η, individuals, σ)
    nll = 0.0
    for (η_individual ,individual) in zip(η, individuals)
        km_indiv = map_individual(km_pop[1], η_individual)
        nll += -individual_log_likelihood(km_indiv, individual, σ)
    end
    return nll
end

function update_population_parameters(η, initial_estimate, individuals, σ, Ω)
    objective(p, (η, individuals)) = total_nll(p.km, η, individuals, p.sigma)
    optfunc = OptimizationFunction(objective, AutoForwardDiff())
    optprob_train = OptimizationProblem(optfunc, ComponentArray(km=[initial_estimate], sigma=σ), (η, individuals))

    optsol_train = Optimization.solve(optprob_train, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=5)
    return optsol_train.u

end


function SAEM(
    individuals,
    initial_population_parameter;
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
    η = fill(prior_η, length(individuals))
    km_pop = initial_population_parameter
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
                p_indiv_new, accepted = mcmc_step(η[i], km_pop, individual, σ, Ω, proposal_std, temperature = temperature)
                if accepted
                    acceptance_count += 1
                end
                η[i] = (1 - gamma) * η[i] + gamma * p_indiv_new # stochastic update
            end
            loglikelihood += individual_log_likelihood(map_individual(km_pop, η[i]), individual, σ)

        end

        # update the population parameters
        p_pop_new = update_population_parameters(η, km_pop, individuals, σ, Ω)
        km_pop_new = p_pop_new.km[1]
        σ = p_pop_new.sigma

        km_pop = (1 - gamma) .* km_pop .+ gamma .* km_pop_new

        # update the covariance matrix
        Ω = (1 - Ω_learning_rate) * Ω + Ω_learning_rate * var(η)

        # update diagnostic monitoring
        acceptance_rate = acceptance_count / (length(individuals)*n_mcmc_steps_iter)
        push!(total_nll_values, -loglikelihood)
        push!(acceptance_rates, acceptance_rate)
        fixed_effect_norm = sum(abs2, km_pop)
        random_effect_variance = var(η)

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
        km_pop = km_pop,
        η = η,
        Ω = Ω,
        σ = σ,
        total_nll_values = total_nll_values,
        acceptance_rates = acceptance_rates
    )
end