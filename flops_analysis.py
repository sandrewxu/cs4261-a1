import torch
from thop import profile

# Simple benchmark with fixed batch size

from gqa import GPTModel as GQAModel
from mla import GPTModel as MLAModel
from moe import GPTModel as MOEModel

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "num_experts": 8,
    "num_experts_per_tok": 2,
}

model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "hidden_dim":768*4,  "n_layers": 12, "n_heads": 12, "n_kv_groups": 3, "latent_dim": 512},
    "gpt-medium (355M)": {"emb_dim": 1024, "hidden_dim":1024*4, "n_layers": 24, "n_heads": 16, "n_kv_groups": 4, "latent_dim": 768},
    "gpt-large (774M)": {"emb_dim": 1280, "hidden_dim":1280*4, "n_layers": 36, "n_heads": 20, "n_kv_groups": 5, "latent_dim": 960},
    "gpt-xl (1558M)": {"emb_dim": 1600, "hidden_dim":1600*4, "n_layers": 48, "n_heads": 25, "n_kv_groups": 5, "latent_dim": 1200},
}

flops_per_second = {
    # https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23
    "H200": {
        torch.float32: 67e12,  # 67 TFLOPs for FP32 on NVIDIA H200 SXM
        torch.float16: 1979e12,  # 1979 TFLOPs for FP16 on NVIDIA H200 SXM
        torch.bfloat16: 1979e12
    },
    # https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf
    "B200": {
        torch.float32: 80e12,  # 80 teraFLOPs for FP32 on NVIDIA GB200 NVL 72
        torch.float16: 5e15,  # 5 petaFLOPs for FP16 on NVIDIA GB200 NVL 72
        torch.bfloat16: 5e15
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
input_tensor = torch.randint(0, 50257, (batch_size, 1024)).to(device)

models = {
    "GQA": GQAModel,
    "MLA": MLAModel,
    "MoE": MOEModel,
}

mfu = 0.5  # 50% Model FLOPs Utilization

for model_name, ModelClass in models.items():
    print(f"\n{'='*80}")
    print(f"{model_name} Model Benchmarks")
    print(f"{'='*80}")
    
    for size in model_configs:
        config = {**BASE_CONFIG, **model_configs[size]}

        model = ModelClass(config).bfloat16()
        model.to(device)

        # MACS = multiply-accumulate operations
        # MACS are typically counted as two FLOPS (one multiply and one accumulate)
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops = 2 * macs

        print(f"\n{size}")
        print(f"  Total FLOPs: {flops:.2e}")
        print(f"  Parameters: {params/1e6:.1f}M")

        # Calculate time for each GPU and dtype at 50% MFU
        for gpu_name, gpu_specs in flops_per_second.items():
            print(f"\n  {gpu_name} (at {int(mfu*100)}% MFU):")
            for dtype, theoretical_flops in gpu_specs.items():
                actual_flops = theoretical_flops * mfu
                time_seconds = flops / actual_flops

                dtype_name = str(dtype).split('.')[-1]
                print(f"    {dtype_name:8}: {time_seconds*1000:.2f} ms")
        del model
        torch.cuda.empty_cache()
