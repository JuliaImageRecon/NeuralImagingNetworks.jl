export ConvAsymPadding, Crop, BatchNormWrap,
       UNetConvBlock,
       UNetExpansiveBlock, UNetContractingBlock, UNetBottleneckBlock, UNetFinalBlock,
       Split, Join

### ConvAsymPadding Layer ###

struct ConvAsymPadding{T, N}
    c::T
    pad::NTuple{N,Int}
end
function ConvAsymPadding(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
    init = Flux.glorot_uniform,  stride = 1, pad = 0, dilation = 1, bias=true) where N
    length(pad) < 2N || all(i -> length(unique(pad[i:i+1])) == 1, 1:2:2N) == 1&& return Conv(k, ch , σ, init=init, stride=stride, pad=pad, dilation=dilation, bias=bias)

    pad_manual = Tuple(map(i -> abs(pad[i] - pad[i+1]), 1:2:2N))
    pad_auto = Tuple(map(i -> minimum(pad[i:i+1]), 1:2:2N))
    return ConvAsymPadding(Conv(k, ch, σ, init=init, stride=stride, pad=pad_auto, dilation=dilation), pad_manual)
end
function (c::ConvAsymPadding)(x::AbstractArray)
    # Maybe there are faster ways to do this as well...
    padding = similar(x, c.pad..., size(x)[end-1:end]...)
    fill!(padding, 0)
    c.c(cat(x, padding, dims=1:length(c.pad)))
end

Flux.@functor ConvAsymPadding


### Cropping Layer ###

struct Crop{N}
    crop::NTuple{N,Int}
end

function (c::Crop{4})(x::AbstractArray)
  return x[(1+c.crop[1]):(end-c.crop[2]), (1+c.crop[3]):(end-c.crop[4]),:,:]
end

function (c::Crop{6})(x::AbstractArray)
  return x[(1+c.crop[1]):(end-c.crop[2]), (1+c.crop[3]):(end-c.crop[4]), (1+c.crop[5]):(end-c.crop[6]),:,:]
end

Flux.@functor Crop

### BatchNormWrap ###

expand_dims(x,n::Int) = reshape(x,ones(Int64,n)...,size(x)...)
	
function _random_normal(shape...)
  return Float32.(rand(Distributions.Normal(0.f0,0.02f0),shape...)) |> dev()
end
	
function BatchNormWrap(out_ch)
    Chain(x->expand_dims(x,3),
	  BatchNorm(out_ch),
	  x->reshape(x, size(x)[4:end]))
end

### UNetConvBlock ###

UNetConvBlock(in_chs, out_chs; kernel = (3,3,3), pad = (1,1,1), stride=(1,1,1)) =
  Chain(Conv(kernel, in_chs=>out_chs, pad = pad, stride=stride; init=_random_normal),
	BatchNormWrap(out_chs),
	x->leakyrelu.(x,0.02f0))


### UNetExpansiveBlock
UNetExpansiveBlock(in_chs, mid_chs, out_chs; kernel_size=3) =
  Chain(
    Conv((kernel_size,kernel_size,kernel_size),
      in_chs => mid_chs, 
      pad = 1, 
      gelu
    ),
	  BatchNorm(mid_chs),
    Conv((kernel_size,kernel_size,kernel_size),
     mid_chs => out_chs,
     pad=1,
     gelu
    ),
    BatchNorm(out_chs),
    ConvTranspose((kernel_size,kernel_size,kernel_size),
      out_chs => out_chs,
      pad=SamePad(),
      stride=2,
    )
  )

### UNetContractingBlock
UNetContractingBlock(in_chs,out_chs;kernel_size=3) =
  Chain(
    Conv((kernel_size,kernel_size,kernel_size),
      in_chs => out_chs, 
      pad = 1, 
      gelu
    ),
	  BatchNorm(out_chs),
    Conv((kernel_size,kernel_size,kernel_size),
     out_chs => out_chs,
     pad=1,
     gelu
    ),
    BatchNorm(out_chs),
  )

### UNetFinalBlock
UNetFinalBlock(in_chs,mid_chs,out_chs; kernel_size=3) = 
  Chain(
    Conv((kernel_size,kernel_size,kernel_size),
      in_chs => mid_chs,
      pad=1,
      gelu
    ),
    BatchNorm(mid_chs),
    Conv((kernel_size,kernel_size,kernel_size),
      mid_chs => mid_chs,
      pad=1,
      gelu
    ),
    BatchNorm(mid_chs),
    Conv((kernel_size,kernel_size,kernel_size),
      mid_chs => out_chs,
      pad=1,
      gelu
    ),
    BatchNorm(out_chs),
  )

  ### UNetBottleneckBlock
  UNetBottleneckBlock(in_chs,mid_chs,out_chs; kernel_size=3) = 
    Chain(
      Conv((kernel_size,kernel_size,kernel_size),
        in_chs=>mid_chs,
        pad=1,
        gelu
      ),
      BatchNorm(mid_chs),
      Conv((kernel_size,kernel_size,kernel_size),
        mid_chs=>mid_chs,
        pad=1,
        gelu
      ),
      BatchNorm(mid_chs),
      ConvTranspose((kernel_size,kernel_size,kernel_size),
        mid_chs=>out_chs,
        pad=SamePad(),
        stride=2
      ),
      
    )














# Join layer
"""
    Join(dim::Int64)
    Join(dim = dim::Int64)
Concatenates a tuple of arrays along a dimension `dim`. A convenient and type stable way of using `x -> cat(x..., dims = dim)`.
"""
struct Join{D}
    dim::Int64
    function Join(dim)
        if dim>4
            throw(DimensionMismatch("Dimension should be 1, 2, 3 or 4."))
        end
        new{dim}(dim)
    end
    function Join(;dim)
        if dim>4
            throw(DimensionMismatch("Dimension should be 1, 2, 3 or 4."))
        end
        new{dim}(dim)
    end
end
(m::Join{D})(x::NTuple{N,AbstractArray}) where {D,N} = cat(x..., dims = Val(D))

function Base.show(io::IO, l::Join)
    print(io, "Join(", "dim = ",l.dim, ")")
end


# Split layer
"""
    Split(outputs::Int64, dim::Int64)
    Split(outputs::Int64, dim = dim::Int64)
Breaks an array into a number of arrays which is equal to `outputs` along a dimension `dim`. `dim` should we divisible by `outputs` without a remainder.
"""
struct Split{O,D}
    outputs::Int64
    dim::Int64
    function Split(outputs,dim)
        if dim>4
            throw(DimensionMismatch("Dimension should be 1, 2, 3 or 4."))
        elseif outputs<2
            throw(DomainError(outputs, "The number of outputs should be 2 or more."))
        end
        new{outputs,dim}(outputs,dim)
    end
    function Split(outputs;dim)
        if dim>4
            throw(DimensionMismatch("Dimension should be 1, 2, 3 or 4."))
        elseif outputs<2
            throw(DomainError(outputs, "The number of outputs should be 2 or more."))
        end
        new{outputs,dim}(outputs,dim)
    end
end

function Split_func(x::T,m::Split{O,D}) where {O,D,T<:AbstractArray{<:AbstractFloat,2}}
    if D!=1
        throw(DimensionMismatch("Dimension should be 1."))
    end
    step_var = Int64(size(x, D) / O)
    f = i::Int64 -> (1+(i-1)*step_var):(i)*step_var
    vals = ntuple(i -> i, O)
    inds_vec = map(f,vals)
    x_out = map(inds -> x[inds,:],inds_vec)
    return x_out
end

function get_part(x::T,inds::NTuple{O,UnitRange{Int64}},d::Type{Val{1}}) where {O,T<:AbstractArray{<:AbstractFloat,5}}
    x_out = map(inds -> x[inds,:,:,:,:],inds)
end
function get_part(x::T,inds::NTuple{O,UnitRange{Int64}},d::Type{Val{2}}) where {O,T<:AbstractArray{<:AbstractFloat,5}}
    x_out = map(inds -> x[:,inds,:,:,:],inds)
end
function get_part(x::T,inds::NTuple{O,UnitRange{Int64}},d::Type{Val{3}}) where {O,T<:AbstractArray{<:AbstractFloat,5}}
    x_out = map(inds -> x[:,:,inds,:,:],inds)
end
function get_part(x::T,inds::NTuple{O,UnitRange{Int64}},d::Type{Val{4}}) where {O,T<:AbstractArray{<:AbstractFloat,5}}
  x_out = map(inds -> x[:,:,:,inds,:],inds)
end
function Split_func(x::T,m::Split{O,D}) where {O,D,T<:AbstractArray{<:AbstractFloat,5}}
    step_var = Int64(size(x, D) / O)
    f = i::Int64 -> (1+(i-1)*step_var):(i)*step_var
    vals = ntuple(i -> i, O)
    inds_tuple = map(f,vals)
    x_out = get_part(x,inds_tuple,Val{D})
    return x_out
end
(m::Split{O,D})(x::T) where {O,D,T<:AbstractArray} = Split_func(x,m)

function Base.show(io::IO, l::Split)
    print(io, "Split(", l.outputs,", dim = ",l.dim, ")")
end

# Makes Parallel layer type stable when used after Split
(m::Parallel)(xs::NTuple{N,AbstractArray}) where N = map((f,x) -> f(x), m.layers,xs)


# Addition layer
"""
    Addition()
A convenient way of using `x -> sum(x)`.
"""
struct Addition 
end
(m::Addition)(x::NTuple{N,AbstractArray}) where N = sum(x)


# Activation layer
"""
    Activation(f::Function)
A convenient way of using `x -> f(x)`.
"""
struct Activation{F}
    f::F
    Activation(f::Function) = new{typeof(f)}(f)
end
(m::Activation{F})(x::AbstractArray) where F = m.f.(x)

function Base.show(io::IO, l::Activation)
    print(io, "Activation(",l.f, ")")
end


# Flatten layer
"""
    Flatten()
Flattens an array. A convenient way of using `x -> Flux.flatten(x)`.
"""
struct Flatten 
end
(m::Flatten)(x::AbstractArray) = Flux.flatten(x)


# Identity layer
"""
    Identity()
Returns its input without changes. Should be used with a `Parallel` layer if one wants to have a branch that does not change its input.
"""
struct Identity
end
(m::Identity)(x::AbstractArray) = x
