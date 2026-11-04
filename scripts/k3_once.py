"""K3's load-time ops on top of kern's `once`: the derive tables K3's
checkpoint does not hold (`source/k3_weight_prep.cu`, `k3_moe_prefill.cu`).
kern dropped them when the K3 line moved to this repository; importing this
module registers them again, before any generator calls `Once.call`.
"""
import once

once.KERNELS.update({
    "fill_bf16": ("k3_weight_prep", "kern_fill_bf16", ["out buffer<bf16>", "i32", "f32"]),
    "k3_conv_taps": ("k3_weight_prep", "kern_k3_conv_taps",
                     ["in buffer<f32>", "in buffer<f32>", "in buffer<f32>", "out buffer<f32>", "i32"]),
    "k3_scoring": ("k3_weight_prep", "kern_k3_scoring", ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32"]),
    "k3_wsm": ("k3_weight_prep", "kern_k3_wsm", ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"]),
    "k3_kvb_aug": ("k3_weight_prep", "kern_k3_kvb_aug", ["in buffer<bf16>", "out buffer<bf16>", "i32"]),
    "k3_moe_w_shuffle": ("k3_moe_prefill", "kern_k3_moe_w_shuffle",
                         ["in buffer<u8>", "out buffer<u8>", "i32", "i32", "i32", "i32"]),
    "k3_moe_sf_shuffle": ("k3_moe_prefill", "kern_k3_moe_sf_shuffle",
                          ["in buffer<u8>", "out buffer<u8>", "i32", "i32", "i32", "i32"]),
})
once.FILL["bf16"] = ("fill_bf16", "f32")
