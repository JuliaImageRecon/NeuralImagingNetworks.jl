using Flux

export DnCNN


struct _DnCNNBlock
    # public
    in_channels::Integer
    out_channels::Integer
    kernel_size::Integer
    padding::Integer
    bias::Bool
    batch_norm::Bool
    # private
    layers

    function _DnCNNBlock(;
        in_channels::Integer,
        out_channels::Integer,
        kernel_size::Integer,
        padding::Integer,
        bias::Bool,
        batch_norm::Bool
    )
        layers = [
            Conv((kernel_size, kernel_size), in_channels => out_channels, pad=padding, bias=bias),
            relu
        ]
        if batch_norm
            insert!(layers, 2, BatchNorm(out_channels))
        end
        layers = Chain(layers)
        new(in_channels, out_channels, kernel_size, padding, bias, batch_norm, layers)
    end
end
@Flux.functor _DnCNNBlock

function (model::_DnCNNBlock)(x)
    return model.layers(x)
end


struct DnCNN
    # public
    in_channels::Integer
    out_channels::Integer
    n_features::Integer
    n_layers::Integer
    kernel_size::Integer
    padding::Integer
    bias::Bool
    batch_norm::Bool
    # private
    layers

    function DnCNN(;
        in_channels::Integer=1,
        out_channels::Integer=1,
        n_features::Integer=64,
        n_layers::Integer=17,
        kernel_size::Integer=3,
        padding::Integer=1,
        bias::Bool=false,
        batch_norm::Bool=true
    )
        layers = []
        push!(layers, _DnCNNBlock(in_channels=in_channels, out_channels=n_features, kernel_size=kernel_size, padding=padding, bias=bias, batch_norm=false))
        for _ in 1:n_layers-2
            push!(layers, _DnCNNBlock(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size, padding=padding, bias=bias, batch_norm=batch_norm))
        end
        push!(layers, Conv((kernel_size, kernel_size), n_features => out_channels, pad=padding, bias=bias))
        layers = Chain(layers)
        new(
            in_channels,
            out_channels,
            n_features,
            n_layers,
            kernel_size,
            padding,
            bias,
            batch_norm,
            layers
        )
    end
end
@Flux.functor DnCNN

function (model::DnCNN)(x)
    return model.layers(x)
end
