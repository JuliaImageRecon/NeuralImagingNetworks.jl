export make_model_resDenseNet

"""
Generate *Residual Dense Net* according to
Zhang et al. 2018
'Residual Dense Network for Image Super-Resolution'

* `filter` is a tuple of Integers denoting the kernel size used in local feature extraction
* `numRDB` denotes the number of residual dense blocks (`D` in paper)
* `numRDBLayer` denotes the number of convolutional layers per RDB (not including 1x1 conv) (`C` in paper)
* `growthRate` denotes the number of feature channels produced by each RDB layer (`G` in paper)
* `inChannel` is the number of channels of the input
* `localChannel` is the number of channels for local feature extraction (`G_0` in paper)
* `σ` is the activation function in the RDBs
* `catFct` dictates how local feature maps are concatenated
"""
function make_model_resDenseNet(filter = (3,3,3); numRDB::Integer=5, numRDBLayers::Integer = 4,
                                growthRate::Integer=8,  inChannel::Integer=1, localChannel::Integer=10,
                                σ = relu)#, catFct = (x,y)->cat(x,y,dims=4))
    catFct(x,y) = cat(x,y,dims=4)
    # create inner net consisting of all RDBs (except the first one since its input is not used for concatenation)
    makeRDB() = resDenseBlock(filter, localChannel, growthRate, catFct; numLayers = numRDBLayers, σ = σ)
    rdBlock = makeRDB()
    model_rdb_inner = SkipConnection(rdBlock, catFct)
    for i in 2:numRDB-1
        rdBlock = makeRDB()
        model_rdb_inner = SkipConnection(Chain(rdBlock, model_rdb_inner), catFct)
    end
    rdBlock = makeRDB()
    # create net from second conv to second to last one
    model_DF = SkipConnection(Chain(Conv(filter, localChannel=>localChannel, pad=SamePad()),
                                    rdBlock,
                                    model_rdb_inner,
                                    Conv((1,1,1), localChannel*numRDB=>localChannel),
                                    Conv(filter, localChannel=>localChannel, pad=SamePad())), +)
    # create net entire net
    model_HQ = SkipConnection(Chain(Conv(filter, inChannel=>localChannel, pad=SamePad()), 
                                    model_DF,
                                    Conv(filter, localChannel=>inChannel, pad=SamePad())), +)
    return model_HQ
end

"""
Create a residual dense block (RDB)

* `filter` as above
* `numRDBChannels` is the number of channels that denote the state of an RDB (in/output channels) (`G_0` in paper)
* `growthRate` is the number of output channels of each convolutional layer (`G` in paper)
* `numLayer` is the number of convolutions (`C` in paper)
* `σ` activation functionA
* `catFct` denotes how new feature maps are concatenated 
"""
function resDenseBlock(filter, numRDBChannels, growthRate, catFct; numLayers=5, σ = relu)
    layers = Any[]
    push!(layers, SkipConnection(Conv(filter, numRDBChannels=>growthRate, σ; pad=SamePad()), catFct))
    for i in 2:numLayers
        push!(layers, SkipConnection(Conv(filter, numRDBChannels+(i-1)*growthRate=>growthRate, σ; pad=SamePad()), catFct))
    end
    push!(layers, Conv((1,1,1), numRDBChannels + growthRate*numLayers=>numRDBChannels))
    rdBlock = SkipConnection(Chain(layers...), +)
    return rdBlock
end