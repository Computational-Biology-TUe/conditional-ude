include("types.jl")
include("c-peptide-models.jl")
using SciMLBase: successful_retcode, OptimizationFunction, OptimizationProblem, OptimizationSolution
using QuasiMonteCarlo: LatinHypercubeSample, sample
using ComponentArrays: ComponentArray
using Random
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity, LineSearches

using ProgressMeter: Progress, next!

### INITIALIZATION FUNCTIONS ###

"""
Samples initial parameters for a neural network model.

# Arguments
- `chain::SimpleChain`: The neural network model.
- `n_initials::Int`: The number of initial parameter sets to sample.
- `rng::AbstractRNG`: The random number generator. Default is `Random.default_rng()`.
"""
function initial_parameters(chain::SimpleChain, n_initials::Int; rng::AbstractRNG = Random.default_rng())
    return [init_params(chain, rng=rng) for _ in 1:n_initials]
end

"""
Samples initial parameters for an ODE model.

# Arguments
- `n_models::Int`: The number of models.
- `lhs_lb::T`: The lower bound for the Latin hypercube sampling.
- `lhs_ub::T`: The upper bound for the Latin hypercube sampling.
- `n_initials::Int`: The number of initial parameter sets to sample.
- `rng::AbstractRNG`: The random number generator. Default is `Random.default_rng()`.
"""
function initial_parameters(n_models::Int, lhs_lb::T, lhs_ub::T, n_initials, rng::AbstractRNG = Random.default_rng()) where T <: Real
    return sample(n_initials, repeat([lhs_lb], n_models), repeat([lhs_ub], n_models), LatinHypercubeSample(rng))
end

### LOSS FUNCTIONS ###

"""
loss(θ, (model, timepoints, cpeptide_data))

Sum of squared errors loss function for the c-peptide model.

# Arguments
- `θ`: The parameter vector.
- `model::CPeptideModel`: The c-peptide model.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractVector{T}`: The c-peptide data.

# Returns
- `Real`: The sum of squared errors.
"""
function loss(θ, (model, timepoints, cpeptide_data)::Tuple{M, AbstractVector{T}, AbstractVector{T}}) where T <: Real where M <: CPeptideModel

    # solve the ODE problem
    sol = solve(model.problem, p=θ, saveat=timepoints, save_idxs=1)
    
    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end

    # Calculate the mean squared error
    return sum(abs2, Array(sol) - cpeptide_data)
end

function loss_sigma(θ, (model, timepoints, cpeptide_data)::Tuple{M, AbstractVector{T}, AbstractVector{T}}) where T <: Real where M <: CPeptideModel

    error = loss(θ, (model, timepoints, cpeptide_data))
    n = length(timepoints)
    return (n/2) * log(θ.sigma^2) + (1/(2*θ.sigma^2)) * error
end

"""
loss(θ, (models, timepoints, cpeptide_data, neural_network_parameters))

Sum of squared errors loss function for the conditional UDE c-peptide model with known neural network parameters.

# Arguments
- `θ`: The parameter vector.
- `p`: The tuple containing the following elements:
    - `models::CPeptideCUDEModel`: The conditional c-peptide models.
    - `timepoints::AbstractVector{T}`: The timepoints.
    - `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.
    - `neural_network_parameters::AbstractVector{T}`: The neural network parameters.

# Returns
- `Real`: The sum of squared errors.
"""
function loss(θ, (model, timepoints, cpeptide_data, neural_network_parameters)::Tuple{CPeptideConditionalUDEModel, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}) where T <: Real

    # construct the parameter vector
    p = ComponentArray(conditional = θ, neural = neural_network_parameters)

    return loss(p, (model, timepoints, cpeptide_data))
end

function loss_sigma(θ, (model, timepoints, cpeptide_data, neural_network_parameters)::Tuple{CPeptideConditionalUDEModel, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}) where T <: Real

    # construct the parameter vector
    p = ComponentArray(conditional = θ.ode, neural = neural_network_parameters)

    error = loss(p, (model, timepoints, cpeptide_data))
    n = length(timepoints)
    return (n/2) * log(θ.sigma^2) + (1/(2*θ.sigma^2)) * error
end

"""
loss(θ, (models, timepoints, cpeptide_data))

Sum of squared errors loss function for the conditional UDE c-peptide model with multiple models.

# Arguments
- `θ`: The parameter vector.
- `p`: The tuple containing the following elements:
    - `models::AbstractVector{CPeptideCUDEModel}`: The conditional c-peptide models.
    - `timepoints::AbstractVector{T}`: The timepoints.
    - `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.

# Returns
- `Real`: The sum of squared errors.
"""
function loss(θ, (models, timepoints, cpeptide_data)::Tuple{AbstractVector{CPeptideConditionalUDEModel}, AbstractVector{T}, AbstractMatrix{T}}) where T <: Real
    # calculate the loss for each model
    error = 0.0
    for (i, model) in enumerate(models)
        p_model = ComponentArray(conditional = θ.conditional[i], neural=θ.neural)
        error += loss(p_model, (model, timepoints, cpeptide_data[i,:]))

        # check if the solver was successful
        if isinf(error)
            return error
        end

    end
    return error / length(models)
end

### OPTIMIZE FUNCTIONS ###

function _optimize(optfunc::OptimizationFunction, initial_parameters, model::CPeptideUDEModel, 
    timepoints::AbstractVector{T}, cpeptide_data::AbstractVector{T}, number_of_iterations_adam::Int,
    number_of_iterations_lbfgs::Int, learning_rate_adam::Real) where T <: Real

    # training step 1 (Adam)
    optprob_train = OptimizationProblem(optfunc, initial_parameters, (model, timepoints, cpeptide_data))
    optsol_train = Optimization.solve(optprob_train, Optimisers.Adam(learning_rate_adam), maxiters=number_of_iterations_adam)
    
    # training step 2 (LBFGS)
    optprob_train_2 = OptimizationProblem(optfunc, optsol_train.u, (model, timepoints, cpeptide_data))
    optsol_train_2 = Optimization.solve(optprob_train_2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=number_of_iterations_lbfgs)

    return optsol_train_2
end

function _optimize(optfunc::OptimizationFunction, initial_parameters, model::CPeptideConditionalUDEModel, 
    timepoints::AbstractVector{T}, cpeptide_data::AbstractVector{T}, neural_network_parameters::AbstractVector{T},
    lower_bound, upper_bound, number_of_iterations_lbfgs::Int) where T <: Real

    optprob = OptimizationProblem(optfunc, initial_parameters, (model, timepoints, cpeptide_data, neural_network_parameters),
    lb = lower_bound, ub = upper_bound)
    optsol = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=number_of_iterations_lbfgs)

    return optsol
end

function _optimize(optfunc::OptimizationFunction, initial_parameters, models::AbstractVector{CPeptideConditionalUDEModel}, 
    timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, number_of_iterations_adam::Int, 
    number_of_iterations_lbfgs::Int, learning_rate_adam::Real) where T <: Real

    # training step 1 (Adam)
    optprob_train = OptimizationProblem(optfunc, initial_parameters, (models, timepoints, cpeptide_data))
    optsol_train = Optimization.solve(optprob_train, Optimisers.Adam(learning_rate_adam), maxiters=number_of_iterations_adam)
    
    # training step 2 (LBFGS)
    optprob_train_2 = OptimizationProblem(optfunc, optsol_train.u, (models, timepoints, cpeptide_data))
    optsol_train_2 = Optimization.solve(optprob_train_2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=number_of_iterations_lbfgs)

    return optsol_train_2
end

### TRAINING FUNCTIONS ###

"""
train(model::CPeptideUDEModel, timepoints::AbstractVector{T}, cpeptide_data::AbstractVector{T}, rng::AbstractRNG; 
    initial_guesses::Int = 10_000,
    selected_initials::Int = 10,
    number_of_iterations_adam::Int = 1000,
    number_of_iterations_lbfgs::Int = 1000,
    learning_rate_adam::Real = 1e-2) where T <: Real

Trains a c-peptide model with a neural network for c-peptide production using the conventional UDE framework.

# Arguments
- `model::CPeptideUDEModel`: The c-peptide model.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractVector{T}`: The c-peptide data.
- `rng::AbstractRNG`: The random number generator.
- `initial_guesses::Int`: The number of initial guesses. Default is 10,000.
- `selected_initials::Int`: The number of selected initials. Default is 10.
- `number_of_iterations_adam::Int`: The number of iterations for the Adam optimizer. Default is 1,000.
- `number_of_iterations_lbfgs::Int`: The number of iterations for the L-BFGS optimizer. Default is 1,000.
- `learning_rate_adam::Real`: The learning rate for the Adam optimizer. Default is 1e-2.

# Returns
- `AbstractVector{OptimizationSolution}`: The optimization solutions.
"""
function train(model::CPeptideUDEModel, timepoints::AbstractVector{T}, cpeptide_data::AbstractVector{T}, rng::AbstractRNG;
    initial_guesses::Int = 10_000,
    selected_initials::Int = 10,
    number_of_iterations_adam::Int = 1000,
    number_of_iterations_lbfgs::Int = 1000,
    learning_rate_adam::Real = 1e-2) where T <: Real

    # sample initial parameters
    initial_p = Vector{Float64}.(initial_parameters(model.chain, initial_guesses; rng=rng))

    # preselect initial parameters
    losses_initial = Float64[]
    prog = Progress(initial_guesses; dt=0.01, desc="Evaluating initial guesses... ", showspeed=true, color=:firebrick)
    for p in initial_p
        loss_value = loss(p, (model, timepoints, cpeptide_data))
        push!(losses_initial, loss_value)
        next!(prog)
    end

    optsols = OptimizationSolution[]
    optfunc = OptimizationFunction(loss, AutoForwardDiff())
    prog = Progress(selected_initials; dt=1.0, desc="Optimizing...", color=:blue)
    for param_indx in partialsortperm(losses_initial, 1:selected_initials)
        try 
            optsol_train_2 = _optimize(optfunc, initial_p[param_indx], 
                                       model, timepoints, cpeptide_data, number_of_iterations_adam, 
                                       number_of_iterations_lbfgs, learning_rate_adam)
            push!(optsols, optsol_train_2)
        catch
            println("Optimization failed... Skipping")
        end
        next!(prog)
    end

    return optsols

end

"""
train(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, neural_network_parameters::AbstractVector{T}; 
    initial_beta::Real = -2.0,
    lbfgs_lower_bound::Real = -4.0,
    lbfgs_upper_bound::Real = 1.0,
    lbfgs_iterations::Int = 1000) where T <: Real

Trains a c-peptide model with a conditional neural network for c-peptide production using the conditional UDE framework. This function is used when the neural network parameters are known
and fixed. Only the conditional parameter(s) are optimized.

# Arguments
- `models::AbstractVector{CPeptideCUDEModel}`: The c-peptide models.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.
- `neural_network_parameters::AbstractVector{T}`: The neural network parameters.
- `initial_beta::Real`: The initial beta value. Default is -2.0.
- `lbfgs_lower_bound::Real`: The lower bound for the L-BFGS optimizer. Default is -4.0.
- `lbfgs_upper_bound::Real`: The upper bound for the L-BFGS optimizer. Default is 1.0.
- `lbfgs_iterations::Int`: The number of iterations for the L-BFGS optimizer. Default is 1,000.

# Returns
- `AbstractVector{OptimizationSolution}`: The optimization solutions.
"""
function train(models::AbstractVector{CPeptideConditionalUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, 
    neural_network_parameters::AbstractVector{T};
    initial_beta = -2.0,
    lbfgs_lower_bound::V = -4.0,
    lbfgs_upper_bound::V = 1.0,
    lbfgs_iterations::Int = 1000
    ) where T <: Real where V <: Real

    optsols = OptimizationSolution[]
    optfunc = OptimizationFunction(loss, AutoForwardDiff())
    for (i,model) in enumerate(models)
        optsol = _optimize(optfunc, [initial_beta],  model, timepoints, cpeptide_data[i,:], neural_network_parameters, lbfgs_lower_bound, lbfgs_upper_bound, lbfgs_iterations)
        push!(optsols, optsol)
    end

    return optsols
end

function train_with_sigma(models::AbstractVector{CPeptideConditionalUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, 
    neural_network_parameters::AbstractVector{T};
    initial_beta = -2.0,
    lbfgs_lower_bound::V = -4.0,
    lbfgs_upper_bound::V = 1.0,
    lbfgs_iterations::Int = 1000
    ) where T <: Real where V <: Real

    optsols = OptimizationSolution[]
    optfunc = OptimizationFunction(loss_sigma, AutoForwardDiff())
    for (i,model) in enumerate(models)
        initials = ComponentArray(ode = [initial_beta], sigma = 1.0)
        optsol = _optimize(optfunc, initials,  model, timepoints, cpeptide_data[i,:], neural_network_parameters, [lbfgs_lower_bound, -Inf], [lbfgs_upper_bound, Inf], lbfgs_iterations)
        push!(optsols, optsol)
    end

    return optsols
end

"""
train(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, rng::AbstractRNG; 
    initial_guesses::Int = 25_000,
    selected_initials::Int = 25,
    lhs_lower_bound::V = -2.0,
    lhs_upper_bound::V = 0.0,
    n_conditional_parameters::Int = 1,
    number_of_iterations_adam::Int = 1000,
    number_of_iterations_lbfgs::Int = 1000,
    learning_rate_adam::Real = 1e-2) where T <: Real where V <: Real

Trains a c-peptide model with a conditional neural network for c-peptide production using the conditional UDE framework. This function is used when the neural network parameters are unknown.
Both the neural network and conditional parameters are optimized.

# Arguments
- `models::AbstractVector{CPeptideCUDEModel}`: The c-peptide models.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.
- `rng::AbstractRNG`: The random number generator.
- `initial_guesses::Int`: The number of initial guesses. Default is 25,000.
- `selected_initials::Int`: The number of selected initials. Default is 25.
- `lhs_lower_bound::V`: The lower bound for the LHS sampling. Default is -2.0.
- `lhs_upper_bound::V`: The upper bound for the LHS sampling. Default is 0.0.
- `n_conditional_parameters::Int`: The number of conditional parameters. Default is 1.
- `number_of_iterations_adam::Int`: The number of iterations for the Adam optimizer. Default is 1,000.
- `number_of_iterations_lbfgs::Int`: The number of iterations for the L-BFGS optimizer. Default is 1,000.
- `learning_rate_adam::Real`: The learning rate for the Adam optimizer. Default is 1e-2.

# Returns
- `AbstractVector{OptimizationSolution}`: The optimization solutions.
"""
function train(models::AbstractVector{CPeptideConditionalUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractVecOrMat{T}, rng::AbstractRNG; 
    initial_guesses::Int = 25_000,
    selected_initials::Int = 25,
    lhs_lower_bound::V = -2.0,
    lhs_upper_bound::V = 0.0,
    n_conditional_parameters::Int = 1,
    number_of_iterations_adam::Int = 1000,
    number_of_iterations_lbfgs::Int = 1000,
    learning_rate_adam::Real = 1e-2) where T <: Real where V <: Real

    # sample initial parameters
    initial_neural_params = initial_parameters(models[1].chain, initial_guesses;rng=rng)
    initial_ode_params = initial_parameters(length(models), lhs_lower_bound, lhs_upper_bound, initial_guesses, rng)

    initials = [ComponentArray(
        neural = initial_neural_params[i],
        conditional = repeat(initial_ode_params[:,i],1, n_conditional_parameters)
    ) for i in eachindex(initial_neural_params)]

    # preselect initial parameters
    losses_initial = Float64[]
    prog = Progress(initial_guesses; dt=0.01, desc="Evaluating initial guesses... ", showspeed=true, color=:firebrick)
    for p in initials
        loss_value = loss(p, (models, timepoints, cpeptide_data))
        push!(losses_initial, loss_value)
        next!(prog)
    end

    println("Initial parameters evaluated. Optimizing for the best $(selected_initials) initial parameters.")
    optsols = OptimizationSolution[]
    optfunc = OptimizationFunction(loss, AutoForwardDiff())
    prog = Progress(selected_initials; dt=1.0, desc="Optimizing...", color=:blue)
    for param_indx in partialsortperm(losses_initial, 1:selected_initials)
        try 
            optsol_train_2 = _optimize(optfunc, initials[param_indx], 
                                       models, timepoints, cpeptide_data, number_of_iterations_adam, 
                                       number_of_iterations_lbfgs, learning_rate_adam)
            push!(optsols, optsol_train_2)
        catch
            println("Optimization failed... Skipping")
        end
        next!(prog)
    end

    return optsols

end

### MODEL SELECTION FUNCTIONS ###

"""
select_model(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, neural_network_parameters, betas_train)

Selects the best model based on the data and the neural network parameters. This evaluates the neural network parameters on each individual in the 
validation set and selects the model that performs best on each individual. The model that is most frequently selected as the best model is returned.

# Arguments
- `models::AbstractVector{CPeptideCUDEModel}`: The c-peptide models.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.
- `neural_network_parameters`: The neural network parameters.
- `betas_train`: The training data for the conditional parameters.

# Returns
- `Int`: The index of the best model.
"""
function evaluate_model(
    models::AbstractVector{CPeptideConditionalUDEModel},
    timepoints::AbstractVector{T},
    cpeptide_data::AbstractMatrix{T},
    neural_network_parameters,
    betas_train::AbstractVector{<:AbstractVector{T}}) where T<:Real

    model_objectives = []
    for (betas, p_nn) in zip(betas_train, neural_network_parameters)
        try
            initial = mean(betas)

            optsols_valid = train(
                models, timepoints, cpeptide_data, p_nn;
                initial_beta = initial, lbfgs_lower_bound=-Inf,
                lbfgs_upper_bound=Inf
            )
            objectives = [sol.objective for sol in optsols_valid]
            push!(model_objectives, objectives)
        catch
            push!(model_objectives, repeat([Inf], length(models)))
        end
    end

    model_objectives = hcat(model_objectives...)

    return model_objectives
end