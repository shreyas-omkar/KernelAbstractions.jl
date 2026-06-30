module CUDAExt

using KernelAbstractions
import KernelAbstractions.KernelIntrinsics as KI
import CUDA
using CUDA: @device_override

# ── Host-side ────────────────────────────────────────────────────────────────

KI.sub_group_size(::CUDA.CUDABackend) = 32

KI.shfl_down_types(::CUDA.CUDABackend) = [
    Int32, UInt32, Int64, UInt64, Float32, Float64, Float16,
]

# ── Device-side sub-group queries ─────────────────────────────────────────────

@device_override @inline KI.get_sub_group_size()     = UInt32(32)
@device_override @inline KI.get_max_sub_group_size() = UInt32(32)
@device_override @inline KI.get_sub_group_local_id() = UInt32(CUDA.laneid())
@device_override @inline KI.get_sub_group_id()       =
    UInt32((Int(CUDA.threadIdx().x) - 1) ÷ 32 + 1)
@device_override @inline KI.get_num_sub_groups()     =
    UInt32(cld(Int(CUDA.blockDim().x), 32))

# ── Device-side sub-group operations ─────────────────────────────────────────

@device_override @inline KI.sub_group_barrier() = CUDA.sync_warp()

@device_override @inline function KI.shfl_down(val::T, offset::Integer) where {T}
    CUDA.shfl_down_sync(0xFFFFFFFF, val, offset)
end

@device_override @inline KI.sub_group_ballot(pred::Bool) =
    CUDA.vote_ballot_sync(0xFFFFFFFF, pred)

end
