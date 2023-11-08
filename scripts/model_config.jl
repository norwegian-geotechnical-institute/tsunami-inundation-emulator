module ModelConfig

using Flux, Serialization

export get_model, reshape_for_lstm, last_out, save, load

# get_model specifies the architecture of the current model. The get_model_X is functions are 
# used to store different model configurations.

# MaxPool model mc8
function get_model_mc8()
    encode = Chain(
        Conv((3,3), 1 => 8, leakyrelu; stride = (1,1), bias=false),
        MaxPool((3,3); stride = (1,2)),
    
        Conv((3,5), 8 => 16, leakyrelu; stride = (1,1), bias=false),
        MaxPool((3,5); stride = (2, 3)),
    
        Conv((3,5), 16 => 32, leakyrelu; stride = (1,1), pad=1, bias=false),
        MaxPool((3,5); stride = (2,3)),
    
        Conv((1,1), 32 => 8, leakyrelu; stride = (1,1), bias=true),
    
        Flux.flatten,
        Dropout(0.5),
        Dense(8*24 => 4, leakyrelu, bias=true)
    )
    decode = Chain(
        Dense(4 => 32, leakyrelu; bias=true),
        Dense(32 => 418908, leakyrelu; bias=true)
    )
    Chain(encode, decode)
end

# MaxPool model mc8_l8_4d
function get_model_mc8_l8_4d()
    encode = Chain(
        Conv((3,3), 1 => 8, leakyrelu; stride = (1,1), bias=false),
        Dropout(0.1),
        MaxPool((3,3); stride = (1,2)),
    
        Conv((3,5), 8 => 16, leakyrelu; stride = (1,1), bias=false),
        Dropout(0.1),
        MaxPool((3,5); stride = (2, 3)),
    
        Conv((3,5), 16 => 32, leakyrelu; stride = (1,1), pad=1, bias=false),
        Dropout(0.1),
        MaxPool((3,5); stride = (2,3)),
    
        Conv((1,1), 32 => 32, leakyrelu; stride = (1,1), bias=true),
    
        Flux.flatten,
        Dropout(0.5),
        Dense(32*24 => 8, leakyrelu, bias=true)
    )
    decode = Chain(
        Dense(8 => 32, leakyrelu; bias=true),
        Dense(32 => 64, leakyrelu; bias=true),
        Dense(64 => 418908, leakyrelu; bias=true)
    )
    Chain(encode, decode)
end


# MaxPool model mc32
function get_model()
    encode = Chain(
        Conv((3,3), 1 => 32, leakyrelu; stride = (1,1), bias=false),
        MaxPool((3,3); stride = (1,2)),
    
        Conv((3,5), 32 => 64, leakyrelu; stride = (1,1), bias=false),
        MaxPool((3,5); stride = (2, 3)),
    
        Conv((3,5), 64 => 128, leakyrelu; stride = (1,1), pad=1, bias=false),
        MaxPool((3,5); stride = (2,3)),
    
        Conv((1,1), 128 => 32, leakyrelu; stride = (1,1), bias=true),
    
        Flux.flatten,
        Dropout(0.5),
        Dense(32*24 => 16, leakyrelu, bias=true),
    )
    decode = Chain(
        Dense(16 => 32, leakyrelu; bias=true),
        Dense(32 => 418908, leakyrelu; bias=true)
    )
    Chain(encode, decode)
end

# MaxPool model mc64
function get_model_mc64()
    encode = Chain(
        Conv((3,3), 1 => 128, leakyrelu; stride = (1,1), bias=false),
        MaxPool((3,3); stride = (1,2)),
    
        Conv((3,5), 128 => 256, leakyrelu; stride = (1,1), bias=false),
        MaxPool((3,5); stride = (2, 3)),
    
        Conv((3,5), 256 => 512, leakyrelu; stride = (1,1), pad=1, bias=false),
        MaxPool((3,5); stride = (2,3)),
    
        Conv((1,1), 512 => 32, leakyrelu; stride = (1,1), bias=true),
    
        Flux.flatten,
        Dropout(0.5),
        Dense(768 => 32, leakyrelu, bias=true),
    )
    decode = Chain(
        Dense(32 => 32, leakyrelu; bias=true),
        Dense(32 => 418908, leakyrelu; bias=true)
    )
    Chain(encode, decode)
end
# Strided convolution model.
function get_model_sc16()
    encode = Chain(
        Conv((3,3), 1 => 16, leakyrelu; stride = (1,1), bias=false),
        Conv((3,3), 16 => 16, leakyrelu; stride = (1,2), bias=false),
    
        Conv((3,5), 16 => 32, leakyrelu; stride = (1, 1), bias=false),
        Conv((3,5), 32 => 32, leakyrelu; stride = (2, 3), bias=false),
    
        Conv((3,5), 32 => 64, leakyrelu; stride = (1,1), pad=1, bias=false),
        Conv((3,5), 64 => 64, leakyrelu; stride = (2,3), bias=false),
    
        Conv((1,1), 64 => 64, leakyrelu; stride = (1,1), bias=true),
    
        Flux.flatten,
        Dropout(0.5),
        Dense(1536 => 32, leakyrelu, bias=true)
    )
    decode = Chain(
        Dense(32 => 64, leakyrelu; bias=true),
        Dense(64 => 418908, leakyrelu; bias=true)
    )
    Chain(encode, decode)
end



# Shallow strided model.
function get_model_x()
    encode_ts = Chain(
        Conv((4,5), 1 => 8, leakyrelu; stride = (1,2), pad = 1, bias=false),
        Conv((5,8), 8 => 16, leakyrelu; stride = (1,3), pad = 0, bias=false),
        Conv((11,9), 16 => 32, leakyrelu; stride = (1,3), pad = 0, bias=false),
        MaxPool((1,6); stride=(1,3), pad=0),
        Dropout(0.2),
        Flux.flatten,
        Dense(175 => 60, leakyrelu, bias=false)
    )
    decode_ts = Chain(
        Dense(60 => 100, leakyrelu; bias=true),
        Dense(100 => 14585, leakyrelu; bias=true),
    )
    Chain(encode_ts, decode_ts)
end

function save(model, filename)
    @info "Writes model weights to file: $filename"
    Flux.reset!(model)
    serialize(filename, collect(Flux.params(cpu(model))))
end

function load(filename)
    @info "Loads model from file: $filename"
    model = get_model()
    weights = deserialize(filename)
    Flux.loadparams!(model, weights)
    return model
end
    
end
