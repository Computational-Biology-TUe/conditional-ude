using SimpleChains: SimpleChain
using SciMLBase: ODEProblem

abstract type CPeptideModel end

struct CPeptideODEModel <: CPeptideModel
    problem::ODEProblem
    production::Function
end

struct CPeptideUDEModel <: CPeptideModel
    problem::ODEProblem
    chain::SimpleChain
end

struct CPeptideConditionalUDEModel <: CPeptideModel
    problem::ODEProblem
    chain::SimpleChain
end



