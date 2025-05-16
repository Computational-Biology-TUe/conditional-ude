include("types.jl")
using DataInterpolations: LinearInterpolation
using OrdinaryDiffEq

### KINETIC MODEL ###

function c_peptide_kinetics!(du, u, k0, k1, k2, c_peptide_0)
    # plasma c-peptide
    du[1] = -(k0 + k2) * u[1] + k1 * u[2] + k0 * c_peptide_0

    # interstitial c-peptide
    du[2] = -k1*u[2] + k2*u[1]
    return nothing
end

"""
Calculates the kinetic parameters for the c-peptide model based on the age and the presence of type 2 diabetes. The
parameters are based on the van Cauter model. [1]

# Arguments
- `age::Real`: The age of the individual.
- `t2dm::Bool`: A boolean indicating whether the individual has type 2 diabetes.

# Returns
- `Tuple`: A tuple containing the kinetic parameters k0, k1, and k2.

---
[1]: Van Cauter, E., Mestrez, F., Sturis, J., Polonsky, K. S. (1992). Estimation of insulin secretion rates from C-peptide levels. Comparison of individual and standard kinetic parameters for C-peptide clearance. Diabetes, 41(3), 368-377.
"""
function van_cauter_parameters(age::Real, t2dm::Bool)

    # set "van Cauter" parameters
    short_half_life = t2dm ? 4.52 : 4.95
    fraction = t2dm ? 0.78 : 0.76
    long_half_life = 0.14 * age + 29.2

    k1 = fraction * (log(2)/long_half_life) + (1-fraction) * (log(2)/short_half_life)
    k0 = (log(2)/short_half_life)*(log(2)/long_half_life)/k1
    k2 = (log(2)/short_half_life) + (log(2)/long_half_life) - k0 - k1

    return k0, k1, k2
end

"""
Generates a `DifferentialEquations` compatible function for the van Cauter model of c-peptide kinetics.

# Arguments
- `c_peptide_0::Real`: The initial (steady-state) concentration of c-peptide.
- `age::Real`: The age of the individual.
- `t2dm::Bool`: A boolean indicating whether the individual has type 2 diabetes.

# Returns
- `Function`: A function that takes the current state and time, and returns the rate of change of the state.
- `Tuple`: A tuple containing the kinetic parameters k0, k1, and k2.
"""
function van_cauter_model(c_peptide_0, age, t2dm)
    # get the kinetic parameters
    k0, k1, k2 = van_cauter_parameters(age, t2dm)

    # construct the ode function
    ode!(du, u, _, _) = c_peptide_kinetics!(du, u, k0, k1, k2, c_peptide_0)

    return ode!, (k0, k1, k2)
end

### PRODUCTION MODELS ###

function analytic_production(::T, p, t,
    production::Function, glucose::LinearInterpolation; t0 = 0.0) where T

    ΔG = glucose(t) - glucose(t0)
    plasma_production = production(ΔG, p)

    return [plasma_production, 0.0]
end

function neural_network_production(::T, p, t,
    network::SimpleChain, glucose::LinearInterpolation; t0 = 0.0) where T

    ΔG = glucose(t) - glucose(t0)
    plasma_production = network([ΔG], p)[1] - network([0.0], p)[1]

    return T([plasma_production, 0.0])
end

function conditional_production(::T, p, t, 
    network::SimpleChain, glucose::LinearInterpolation; t0 = 0.0) where T

    ΔG = glucose(t) - glucose(t0)
    β = exp.(p.conditional)
    plasma_production = network([ΔG; β], p.neural)[1] - network([0.0; β], p.neural)[1]

    return T([plasma_production, 0.0])
end

function conditional_covariate_production(::T, p, t, 
    network::SimpleChain, glucose::LinearInterpolation, age; t0 = 0.0) where T

    ΔG = glucose(t) - glucose(t0)
    β = exp.(p.conditional)
    plasma_production = network([ΔG; β; age], p.neural)[1] - network([0.0; β; age], p.neural)[1]

    return T([plasma_production, 0.0])
end

### TOTAL C-PEPTIDE MODEL ###

function combine(kinetics!, production)
    function combined!(du, u, p, t)
        kinetics!(du, u, p, t)
        du .+= production(u, p, t)
    end
    return combined!
end

### C-Peptide Models ###

function CPeptideODEModel(glucose_data::AbstractVector{T}, glucose_timepoints::AbstractVector{T}, age::Real, 
    production_function::Function, cpeptide_data::AbstractVector{T}, t2dm::Bool) where T <: Real

    # basal c-peptide
    c_0 = cpeptide_data[1]

    # kinetics
    kinetics!, (_, k1, k2) = van_cauter_model(c_0, age, t2dm)

    # production
    # interpolate glucose data
    glucose = LinearInterpolation(glucose_data, glucose_timepoints)
    production(u, p, t) = analytic_production(u, p, t, production_function, glucose; t0 = glucose_timepoints[1])

    # initial conditions
    u0 = [c_0, (k2/k1)*c_0]

    # time span
    tspan = (glucose_timepoints[1], glucose_timepoints[end])

    # construct the ode problem
    ode = ODEProblem(combine(kinetics!, production), u0, tspan)

    return CPeptideODEModel(ode, production_function)
end

function CPeptideUDEModel(glucose_data::AbstractVector{T}, glucose_timepoints::AbstractVector{T}, age::Real, 
    network::SimpleChain, cpeptide_data::AbstractVector{T}, t2dm::Bool) where T <: Real

    # basal c-peptide
    c_0 = cpeptide_data[1]

    # kinetics
    kinetics!, (_, k1, k2) = van_cauter_model(c_0, age, t2dm)

    # production
    # interpolate glucose data
    glucose = LinearInterpolation(glucose_data, glucose_timepoints)
    production(u, p, t) = neural_network_production(u, p, t, network, glucose; t0 = glucose_timepoints[1])

    # initial conditions
    u0 = [c_0, (k2/k1)*c_0]

    # time span
    tspan = (glucose_timepoints[1], glucose_timepoints[end])

    # construct the ode problem
    ode = ODEProblem(combine(kinetics!, production), u0, tspan)

    return CPeptideUDEModel(ode, network)
end

function CPeptideConditionalUDEModel(glucose_data::AbstractVector{T}, glucose_timepoints::AbstractVector{T}, age::Real, 
    network::SimpleChain, cpeptide_data::AbstractVector{T}, t2dm::Bool) where T <: Real

    # basal c-peptide
    c_0 = cpeptide_data[1]

    # kinetics
    kinetics!, (_, k1, k2) = van_cauter_model(c_0, age, t2dm)

    # production
    # interpolate glucose data
    glucose = LinearInterpolation(glucose_data, glucose_timepoints)
    production(u, p, t) = conditional_production(u, p, t, network, glucose; t0 = glucose_timepoints[1])

    # initial conditions
    u0 = [c_0, (k2/k1)*c_0]

    # time span
    tspan = (glucose_timepoints[1], glucose_timepoints[end])

    # construct the ode problem
    ode = ODEProblem(combine(kinetics!, production), u0, tspan)

    return CPeptideConditionalUDEModel(ode, network)
end

function CPeptideConditionalCovariateUDEModel(glucose_data::AbstractVector{T}, glucose_timepoints::AbstractVector{T}, age::Real, 
    network::SimpleChain, cpeptide_data::AbstractVector{T}, t2dm::Bool) where T <: Real

    # basal c-peptide
    c_0 = cpeptide_data[1]

    # kinetics
    kinetics!, (_, k1, k2) = van_cauter_model(c_0, age, t2dm)

    # production
    # interpolate glucose data
    glucose = LinearInterpolation(glucose_data, glucose_timepoints)
    production(u, p, t) = conditional_covariate_production(u, p, t, network, glucose, age; t0 = glucose_timepoints[1])

    # initial conditions
    u0 = [c_0, (k2/k1)*c_0]

    # time span
    tspan = (glucose_timepoints[1], glucose_timepoints[end])

    # construct the ode problem
    ode = ODEProblem(combine(kinetics!, production), u0, tspan)

    return CPeptideConditionalUDEModel(ode, network)
end
