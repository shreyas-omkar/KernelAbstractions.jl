module CUDAExt


import KernelAbstractions.KernelInterface as KI
import CUDA
using CUDA: @device_override


# NVPTX does not select scoped atomic fences (the generic `KI.device_fence` acquire-release
# fence), so provide the device-scope fence with the native `membar.gl` threadfence.
@device_override @inline KI.device_fence() = CUDA.threadfence()


end   # module CUDAExt
