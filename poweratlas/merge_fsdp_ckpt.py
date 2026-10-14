# -*- coding: utf-8 -*-
"""
把 verl 存的 FSDP 分片 GRPO checkpoint 合并成标准 HF 模型(可 from_pretrained)。
每个权重是 DTensor(Shard(dim=0)):rank0=上半、rank1=下半,沿 dim0 拼起来裁到全局形状即可。
tokenizer 从 SFT ckpt 拷(GRPO 没动 tokenizer,且 actor/ 里 tokenizer 文件可能不全)。

用法:
  python merge_fsdp_ckpt.py --ckpt checkpoints/eccs-v2/grpo-4b-v2/global_step_60 \
      --out checkpoints/eccs-v2/grpo-4b-v2-hf/global_step_60
"""
import argparse, os, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".../global_step_N (含 actor/ 子目录)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="checkpoints/eccs-sft/qwen4b/global_step_72")
    ap.add_argument("--world_size", type=int, default=2)
    args = ap.parse_args()

    actor = os.path.join(args.ckpt, "actor")
    if not os.path.isdir(actor):
        actor = args.ckpt

    print(f"[merge] loading {args.world_size} shards from {actor}")
    shards = [torch.load(os.path.join(actor, f"model_world_size_{args.world_size}_rank_{r}.pt"),
                         weights_only=False, map_location="cpu") for r in range(args.world_size)]

    full = {}
    for k in shards[0]:
        v = shards[0][k]
        if type(v).__name__ == "DTensor":
            p = v._spec.placements[0]
            if hasattr(p, "dim"):                       # Shard(dim=d)
                dim = p.dim
                merged = torch.cat([s[k]._local_tensor for s in shards], dim=dim)
                sl = [slice(None)] * merged.dim()
                sl[dim] = slice(0, v.shape[dim])        # 裁掉 FSDP 的 padding
                full[k] = merged[tuple(sl)].contiguous()
            else:                                        # Replicate / Partial
                full[k] = v._local_tensor.contiguous()
        else:
            full[k] = v
    print(f"[merge] merged {len(full)} tensors")

    cfg = AutoConfig.from_pretrained(actor)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(full, strict=False)
    print(f"[merge] missing keys   : {list(missing)}")     # 期望只有 lm_head.weight(与 embed 权重绑定)
    print(f"[merge] unexpected keys: {list(unexpected)}")   # 期望空

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True).save_pretrained(args.out)
    print(f"[merge] saved HF model -> {args.out}")


if __name__ == "__main__":
    main()
