export make_model_unet, make_model_res_unet, make_model_unet_skip

### make U-Net ###

function make_model_unet(N, inChan = 1)
  layers = Any[]
  maxDepth = minimum(round.(Int,log2.(N)).-1)
  numLayers = min(maxDepth, 3)
  
  H = zeros(Int, numLayers, length(N))
  H[1,:] .= N
  for l=1:numLayers-1
    H[l+1,:] .= ceil.(Int, vec(H[l,:])./2)
  end
  needCrop = Int.(isodd.(H))
  
  #interp = :nearest #trilinear
  interp = :trilinear

  outChan = 64
  for l=1:numLayers
    push!(layers,UNetConvBlock(inChan,outChan,kernel=(3,3,3), stride=(2,2,2), pad=1))
    push!(layers,UNetConvBlock(outChan,outChan,kernel=(3,3,3), pad=1))
    inChan = outChan
    outChan *= 2
  end
  outChan = inChan

  for l=0:(numLayers-1)
    push!(layers, Upsample(interp,scale=(2,2,2))) 
    if sum(needCrop[:,end-l])>0
      push!(layers, Crop( (0,needCrop[end-l,1],0,needCrop[end-l,2],0,needCrop[end-l,3]) ) )
    end  
    push!(layers, UNetConvBlock(inChan,outChan,kernel=(3,3,3), pad=1))
    push!(layers, UNetConvBlock(outChan,outChan,kernel=(1,1,1), pad=0))
    inChan = outChan
    outChan = outChan ÷ 2
  end
  push!(layers, Conv((1, 1, 1), 64=>1, pad = 0, gelu ;init=_random_normal))

  return Chain(layers...)
end



function make_model_unet_skip(N, inChan_ = 1; depth = 3, inputResidual=true, baseChan = 4 )
  catChannels(x,y) = cat(x,y,dims=4)

  maxDepth = minimum(round.(Int,log2.(N)).-1)
  numLayers = min(maxDepth, depth)
  
  H = zeros(Int, numLayers, length(N))
  H[1,:] .= N
  for l=1:numLayers-1
    H[l+1,:] .= ceil.(Int, vec(H[l,:])./2)
  end
  needCrop = Int.(isodd.(H))

  #interp = :nearest #trilinear
  interp = :trilinear

  

  chan = baseChan * 2^numLayers
  currNet = Chain(
    UNetConvBlock(chan, 2*chan, kernel=(3,3,3), pad=1),
    UNetConvBlock(2*chan, chan, kernel=(3,3,3), pad=1)
  )

  for l=1:numLayers
    chan =  baseChan * 2^(numLayers-l)
    inChan = (l==numLayers) ? inChan_ : chan
 
    if sum(needCrop[:,end-l+1]) > 0
     innerNet = Chain(  MaxPool((2,2,2),pad=(needCrop[end-l+1,1],needCrop[end-l+1,2],needCrop[end-l+1,3])),
                        currNet,
                        Upsample(interp, scale=(2,2,2)),
                        Crop( (0,needCrop[end-l+1,1],0,needCrop[end-l+1,2],0,needCrop[end-l+1,3]) ) )
    else
      innerNet = Chain(  MaxPool((2,2,2)), 
                         currNet,
                         Upsample(interp, scale=(2,2,2)) ) 
    end

    currNet = Chain(  UNetConvBlock(inChan, 2*chan, kernel=(3,3,3), pad=1),
                      UNetConvBlock(2*chan, 2*chan, kernel=(3,3,3), pad=1),
                      SkipConnection( innerNet,
                                      catChannels),     
                      UNetConvBlock(4*chan,2*chan,kernel=(3,3,3), pad=1),
                      UNetConvBlock(2*chan,chan,kernel=(3,3,3), pad=1)
          )
  end

  currNet = Chain(
    currNet,
    Conv((1, 1, 1), baseChan=>1, pad = 0, gelu ;init=_random_normal)
  )

  if inputResidual
    return Chain(
        SkipConnection(
          Chain(
            currNet, 
            Conv((1, 1, 1), 1=>1, identity; pad = 0, init=_random_normal),
           ),
          (x,y) -> x .+ y[:,:,:,1:1,:] #mean(y, dims=4) #Add channel one onto the output
        )
      )
  else
    return currNet
  end
end

export make_parallel_channel_unet_skip

function make_parallel_channel_unet_skip(N, inChan::Integer; depth = 3, inputResidual=true, baseChan = 4 )

  networks = ntuple( d->make_model_unet_skip(N; depth, inputResidual, baseChan ), inChan ) 
  return make_parallel_channel_unet_skip(networks)
end


function make_parallel_channel_unet_skip(networks::Tuple)

  inChan = length(networks)

  model = Chain(Split(inChan,4),
                Parallel(tuple, networks), 
                Join(4),
                Conv((3, 3, 3), inChan=>1, gelu; pad =(1,1,1), init=_random_normal)
                #make_model_unet_skip(N, inChan_; depth, inputResidual, baseChan=baseChan*2 )
                )
  return model
end

export make_parallel_channel_unet_skip2


function make_parallel_channel_unet_skip2(networks::Tuple, N)

  inChan = length(networks)

  model = Chain(Split(inChan,4),
                Parallel(tuple, networks), 
                Join(4),
                Conv((3, 3, 3), inChan=>1, gelu; pad =(1,1,1), init=_random_normal),
                make_model_unet_skip(N, 1; depth=3, inputResidual=true, baseChan=16 )
                )
  return model
end
