using OrdinaryDiffEq
using CairoMakie
using SimpleChains: init_params
using ComponentArrays: ComponentArray
using Optimization, OptimizationOptimisers, OptimizationOptimJL, SciMLSensitivity, Zygote
using LineSearches
using StableRNGs
using Statistics
using StatsBase: corspearman
using DataFrames, CSV
using ProgressMeter: Progress, next!
using SciMLBase: OptimizationSolution

softplus(x) = log(1 + exp(x))

function lsup!(du, u, p, t)
    du[1] = - p[1]*u[1]
    du[2] = p[1]*u[1] - p[2]*u[2] / (1 + p[4]*u[3])
    du[3] = p[2]*u[2] / (1 + p[4]*u[3]) - p[3]*u[3]
end

function create_progressbar_callback_with_loss(its, run, losses)
    prog = Progress(its; dt=1, desc="Optimizing run $(run) ", showspeed=true, color=:blue)
    function callback(_, l)
        push!(losses, l)
        next!(prog)
        false
    end

    return callback
end

function get_group_parameters(μ_sup, n_samples; rng= StableRNG(232705))
    μ = [0.4, 0.9, 0.3, μ_sup]
    std = [0.1, 0.1, 0.1, μ_sup/8]
    return max.(μ .+ std .* randn(rng, 4, n_samples),0.05) # sample from normal distribution
end

function generate_data(group_means, group_sizes, timepoints; noise_additive = 0.0, noise_multiplicative = 0.0, rng = StableRNG(232705))

    data = zeros(3, length(timepoints), sum(group_sizes))
    gt_sup_param = Float64[]
    individual_idx = 1

    for (i, group_mean) in enumerate(group_means)
        group_params = get_group_parameters(group_mean, group_sizes[i]; rng=rng)
        for j in axes(group_params, 2)
            u0 = [10.0, 0.0, 0.0]
            tspan = (0.0, timepoints[end])
            prob = ODEProblem(lsup!, u0, tspan, group_params[:,j])
            sol = Array(solve(prob, Tsit5(), saveat=timepoints))

            # add noise
            sol += noise_additive * randn(rng, size(sol)) + noise_multiplicative * sol .* randn(rng, size(sol))

            data[:,:,individual_idx] = max.(sol, 0.0)
            individual_idx += 1
        end
        append!(gt_sup_param, group_params[4,:])
    end

    return data, gt_sup_param
end

"""
neural_network_model(depth::Int, width::Int; input_dims::Int = 2)

Constructs a neural network model with a given depth and width. The input dimensions are set to 2 by default.

# Arguments
- `depth::Int`: The depth of the neural network.
- `width::Int`: The width of the neural network.
- `input_dims::Int`: The number of input dimensions. Default is 2.

# Returns
- `SimpleChain`: A neural network model.
"""
function neural_network_model(depth::Int, width::Int; input_dims::Int = 2)

    layers = []
    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    push!(layers, TurboDense{true}(softplus, 1))

    SimpleChain(static(input_dims), layers...)
end
 
# Define the hybrid model
function ude_lsup!(du, u, p, t, p_true, network)

  û = network([u; exp.(p.conditional)], p.neural)[1] # Network prediction
  du[1] = -p_true[1]*u[1]
  du[2] = p_true[1]*u[1] - û[1]
  du[3] = û[1] - p_true[3]*u[3]

end

function get_prob_func(u0mat, theta, p_neural)
    function prob_fun(prob, i, _)
        prob = remake(prob, p=ComponentArray(
            neural = p_neural,
            conditional = theta[i]
        ), u0 = u0mat[:,i])
        return prob
    end
end

function simul(p, prob, individual_data, timepoints)
    
    u0mat = individual_data[:,1,:]
    prob_func = get_prob_func(u0mat, p.theta, p.neural)
    ensemble = EnsembleProblem(prob, prob_func = prob_func)

    solve(ensemble, Tsit5(), EnsembleThreads(), saveat = timepoints, trajectories = length(p.theta))

end

function suppression_loss(p, (prob, individual_data, timepoints, λ))

    u0mat = individual_data[:,1,:]
    prob_func = get_prob_func(u0mat, p.theta, p.neural)
    ensemble = EnsembleProblem(prob, prob_func = prob_func)

    sims = Array(solve(ensemble, Tsit5(), EnsembleThreads(), saveat = timepoints, trajectories = length(p.theta), sensealg=ForwardDiffSensitivity()))

    # scale
    scale = mean(maximum(individual_data, dims=2), dims=3)[:]

    sum(abs2, (sims - individual_data) ./ scale) / size(individual_data, 3) + λ * sum(abs2, p.neural) # regularization term

end

function fit_suppression_model(p_init, prob, data, timepoints, λ; select_best_n = 1)

    # select best initials
    initial_losses = [suppression_loss(p, (prob, data, timepoints, λ)) for p in p_init]

    # select best 
    if select_best_n > 1
        best_idxs = partialsortperm(initial_losses, 1:select_best_n)
        p_init_best = p_init[best_idxs]
    else
        best_idx = argmin(initial_losses)
        p_init_best = [p_init[best_idx]]
    end

    println("Selected best $(length(p_init_best)) initials")


    optsols = OptimizationSolution[]
    loss_traces = Vector{Float64}[]
    for (run, p_init_selected) in enumerate(p_init_best)
        losses = Float64[]
        try
            # Define the optimization problem
            adtype = Optimization.AutoForwardDiff()
            optf = Optimization.OptimizationFunction(suppression_loss, adtype)
            optprob = Optimization.OptimizationProblem(optf, p_init_selected, (prob, data, timepoints, λ))

            # iterations
            ADAM_iters = 2000
            LBFGS_iters = 2000

            # Step 1: train using Adam
            res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(), callback = create_progressbar_callback_with_loss(ADAM_iters, "$(run)-ADAM", losses), maxiters = ADAM_iters)

            # Step 2: train using L-BFGS
            optprob2 = Optimization.OptimizationProblem(optf, res1.u, (prob, data, timepoints, λ))
            res2 = Optimization.solve(optprob2, LBFGS(linesearch = BackTracking()), callback = create_progressbar_callback_with_loss(LBFGS_iters, "$(run)-LBFGS", losses), maxiters = LBFGS_iters)
            println("Final training loss after $(length(losses)) iterations: $(losses[end])")
            push!(optsols, res2)
        catch
            println("Optimization failed")
        end
        push!(loss_traces, losses)
    end
    return optsols, loss_traces
end

function validate_suppression_model(p_init, prob, data, timepoints, network_params)

    function loss_valid(p_c, (prob, data, timepoints))
        p = ComponentArray(
            theta = p_c,
            neural = network_params
        )
        suppression_loss(p, (prob, data, timepoints, 0.0))
    end

    # select best initials
    initial_losses = [loss_valid(p, (prob, data, timepoints)) for p in p_init]
    best_idx = argmin(initial_losses)
    p_init_best = p_init[best_idx]

    losses = Float64[]

    callback = function (p, l)
        push!(losses, l)
        if length(losses) % 50 == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        return false
    end

    # Define the optimization problem
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction(loss_valid, adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init_best, (prob, data, timepoints))

    # iterations
    LBFGS_iters = 2000

    # Step 1: train using Adam
    try
        res1 = Optimization.solve(optprob, LBFGS(linesearch = BackTracking()), callback = callback, maxiters = LBFGS_iters)
        println("Training loss after $(length(losses)) iterations: $(losses[end])")

        return res1.u, res1.objective
    catch
        println("Optimization failed")
        return p_init_best, Inf
    end
end

function validate_suppression_model_sigma(p_init, prob, data, timepoints, network_params)

    function loss_valid(p_c, (prob, data, timepoints))
        p = ComponentArray(
            conditional = p_c.ode,
            neural = network_params
        )

        simulation = Array(solve(prob, Tsit5(), p=p, u0=data[:,1], saveat = timepoints, sensealg=ForwardDiffSensitivity()))
        nll = 0.0
        for i in axes(data, 1)
            error = sum(abs2, (simulation[i,:] - data[i,:]))
            n = length(timepoints)
            nll += (n/2) * log(p_c.sigma[i]^2) + (1/(2*p_c.sigma[i]^2)) * error
        end
        return nll
    end

    # select best initials
    initial_losses = [loss_valid(ComponentArray(ode=p, sigma=[1.0, 1.0, 1.0]), (prob, data, timepoints)) for p in p_init]
    best_idx = argmin(initial_losses)
    p_init_best = p_init[best_idx]

    losses = Float64[]

    callback = function (p, l)
        push!(losses, l)
        if length(losses) % 50 == 0
            println("Current loss after $(length(losses)) iterations: $(losses[end])")
        end
        return false
    end

    # Define the optimization problem
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction(loss_valid, adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentArray(ode= p_init_best, sigma=[1.0, 1.0, 1.0]), (prob, data, timepoints))

    # iterations
    LBFGS_iters = 2000

    # Step 1: train using Adam
    try
        res1 = Optimization.solve(optprob, LBFGS(linesearch = BackTracking()), callback = callback, maxiters = LBFGS_iters)
        println("Training loss after $(length(losses)) iterations: $(losses[end])")

        return res1.u, res1.objective
    catch
        println("Optimization failed")
        return p_init_best, Inf
    end
end
