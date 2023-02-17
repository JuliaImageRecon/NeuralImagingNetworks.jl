module NeuralImagingNetworks


# ML Stuff
import Distributions
using Flux
using CUDA
using MLUtils

export setDevice, getDevice, dev

const device_ = Ref{Function}(gpu)
function setDevice(dev::Symbol)
  if dev == :cpu
    device_[] = cpu
  elseif dev == :gpu
    device_[] = gpu
  else
    error("Device not supported!")
  end
end
dev() = device_[]

function getDevice()
    if dev() == cpu 
        return :cpu
    elseif dev() == gpu 
        return :gpu
    else
        error("Something is broken!")
    end
end

include("layers.jl")
include("unets.jl")

end
