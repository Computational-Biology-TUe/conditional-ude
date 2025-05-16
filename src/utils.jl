using Random
using ProgressMeter: Progress, next!

"""
Stratified split of the data into training and testing sets by retaining the proportion of each type.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `types::AbstractVector`: The types of the individuals.
- `f_train::Real`: The fraction of the data to use for training.

# Returns
- `Tuple`: A tuple containing the training and testing indices.
"""
function stratified_split(rng::AbstractRNG, types::AbstractVector{T}, f_train::Real)::Tuple{AbstractVector{Int}, AbstractVector{Int}} where T

    training_indices = Int[]
    for type in unique(types)
        type_indices = findall(types .== type)
        n_train = Int(round(f_train * length(type_indices)))
        selection = StatsBase.sample(rng, type_indices, n_train, replace=false)
        append!(training_indices, selection)
    end

    training_indices = sort(training_indices)

    # Get the testing indices by taking the complement of the training indices
    testing_indices = setdiff(1:length(types), training_indices)

    training_indices, testing_indices
end

function create_progressbar_callback(its, run)
    prog = Progress(its; dt=1, desc="Optimizing run $(run) ", showspeed=true, color=:blue)
    function callback(_, _)
        next!(prog)
        false
    end

    return callback
end

function argmedian(x)
    return argmin(abs.(x .- median(x)))
end