using SimpleChains: SimpleChain, TurboDense, static, init_params
using Random

"""
Softplus activation function.

The softplus function is a smooth approximation of the ReLU function, defined as:

```math
\\text{softplus}(x) = \\log\\left[1 + e^{x}\\right]
```
"""
function softplus(x)
    log(1 + exp(x))
end

"""
Constructs a `SimpleChain` neural network with specified widths and activation functions.

# Arguments
- `widths::AbstractVector{Int}`: A vector of integers specifying the widths of each layer.
- `activation_functions::AbstractVector{Function}`: A vector of activation functions for each layer.

# Keyword Arguments
- `input_dims::Int = 2`: The number of input dimensions. Default is 2.
- `output_dims::Int = 1`: The number of output dimensions. Default is 1.
- `output_activation::Function = softplus`: The activation function for the output layer. Default is `softplus`.

# Returns
- `SimpleChain`: A neural network model.

# Example
```julia
using SimpleChains

neural_net = chain([10, 20, 30], [tanh, relu, softplus]; input_dims=4, output_dims=1)
```
# Raises
- `ArgumentError`: If the number of widths does not match the number of activation functions.
- `ArgumentError`: If the widths vector is empty.
"""
function chain(widths::AbstractVector{Int}, activation_functions::AbstractVector{<:Function}; input_dims::Int = 2, output_dims::Int = 1, output_activation::Function = softplus)

    if isempty(widths)
        throw(ArgumentError("Input widths must be non-empty."))
    end

    if length(widths) != length(activation_functions)
        throw(ArgumentError("The number of widths must match the number of activation functions."))
    end

    layers = TurboDense[]
    append!(layers, [TurboDense{true}(func, width) for (func, width) in zip(activation_functions, widths)])
    push!(layers, TurboDense{true}(output_activation, output_dims))

    SimpleChain(static(input_dims), layers...)

end

"""
Constructs a `SimpleChain` neural network with specified widths and a single activation function for all layers.

# Arguments
- `widths::AbstractVector{Int}`: A vector of integers specifying the widths of each layer.
- `activation_function::Function`: A single activation function to be used for all layers.

# Keyword Arguments
- `input_dims::Int = 2`: The number of input dimensions. Default is 2.
- `output_dims::Int = 1`: The number of output dimensions. Default is 1.
- `output_activation::Function = softplus`: The activation function for the output layer. Default is `softplus`.

# Returns
- `SimpleChain`: A neural network model.

# Example
```julia
using SimpleChains

neural_net = chain([10, 20, 30], relu; input_dims=4, output_dims=1)
```

# Raises
- `ArgumentError`: If the widths vector is empty.
"""
function chain(widths::AbstractVector{Int}, activation_function::Function; input_dims::Int = 2, output_dims::Int = 1, output_activation::Function = softplus)
    chain(widths, fill(activation_function, length(widths)); input_dims=input_dims, output_dims=output_dims, output_activation=output_activation)
end

"""
Constructs a `SimpleChain` neural network with specified width and depth, and a single activation function for all layers.

# Arguments
- `width::Int`: An integer specifying the width of each layer.
- `depth::Int`: An integer specifying the number of layers.
- `activation_function::Function`: A single activation function to be used for all layers.

# Keyword Arguments
- `input_dims::Int = 2`: The number of input dimensions. Default is 2.
- `output_dims::Int = 1`: The number of output dimensions. Default is 1.
- `output_activation::Function = softplus`: The activation function for the output layer. Default is `softplus`.

# Returns
- `SimpleChain`: A neural network model.
"""
function chain(width::Int, depth::Int, activation_function::Function; input_dims::Int = 2, output_dims::Int = 1, output_activation::Function = softplus)
    chain(fill(width, depth), fill(activation_function, depth); input_dims=input_dims, output_dims=output_dims, output_activation=output_activation)
end




