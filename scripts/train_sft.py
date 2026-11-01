"""LoRA SFT of Qwen3-VL-4B on one experiment config with ms-swift (tested against ms-swift 4.6 APIs).

    python scripts/train_sft.py --config c3_no_material --data data/sft_v1
    python scripts/train_sft.py --config c3_no_material --loss weighted_ntl
    python scripts/train_sft.py --config c1_full_aux -- --num_train_epochs 1   # extra swift args after --

Logs go to tensorboard and to wandb project `pixels2physics` (run `wandb login` once on the
machine, or set WANDB_API_KEY); runs are tagged with the config and loss mode.

Loss modes:
    ce            plain token cross-entropy over the whole response.
    weighted      CE with per-token weights: the <physics> block x W_PHYSICS, and numeric
                  literals in the code whose value appears in the <physics> block x W_NUMBER.
    weighted_ntl  weighted + Number Token Loss: for every upweighted digit token, the expected
                  distance E_p[|d - d*|] under the model's distribution over the ten digit tokens.
                  Unlike CE, predicting 4 for a 5 costs less than predicting 9.

This file is also passed to ms-swift as an external plugin, which registers the
`physics` loss_scale and the `physics_ntl` loss at import time.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch
from swift.loss import BaseLoss, loss_map
from swift.loss_scale.base import LossScale
from swift.loss_scale.mapping import loss_scale_map

PHYSICS_BLOCK = re.compile(r"<physics>.*?</physics>\n?", re.S)
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")
TRIVIAL = {0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0}  # MPM boilerplate constants (B-spline weights, dim, indices)


def _key(text: str) -> float:
    return float(f"{float(text):.4g}")


class PhysicsLossScale(LossScale):
    """Split a response into (text, weight) segments; ms-swift tokenizes each segment separately.

    Segments end right before a digit run or right after it, which is where the Qwen
    pre-tokenizer already splits, so tokenization is unchanged.
    """

    is_binary = False

    def get_loss_scale(self, context, **kwargs):
        if not isinstance(context, str):
            return [context], [1.0]
        w_physics = float(os.environ.get("P2P_W_PHYSICS", "2.0"))
        w_number = float(os.environ.get("P2P_W_NUMBER", "3.0"))
        block = PHYSICS_BLOCK.search(context)
        if block is None:
            return [context], [1.0]
        values = {_key(m.group()) for m in NUMBER.finditer(block.group())} - TRIVIAL
        texts, weights = [], []
        if block.start():
            texts.append(context[: block.start()])
            weights.append(1.0)
        texts.append(block.group())
        weights.append(w_physics)
        code, cursor = context[block.end():], 0
        for match in NUMBER.finditer(code):
            if _key(match.group()) not in values:
                continue
            if match.start() > cursor:
                texts.append(code[cursor: match.start()])
                weights.append(1.0)
            texts.append(match.group())
            weights.append(w_number)
            cursor = match.end()
        if cursor < len(code):
            texts.append(code[cursor:])
            weights.append(1.0)
        return texts, weights


class PhysicsNTLLoss(BaseLoss):
    """Weighted CE (from the loss_scale) + P2P_NTL_WEIGHT x Number Token Loss on upweighted digits."""

    digit_ids: torch.Tensor | None = None

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, trainer=None, **kwargs):
        from swift.trainers import per_token_loss_func

        trainer = trainer or self.trainer
        token_loss = outputs.loss
        if token_loss is None:
            token_loss = per_token_loss_func(outputs, labels)
            if loss_scale is not None:
                token_loss = token_loss * loss_scale
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        loss = token_loss.sum() / num_items_in_batch

        target = torch.roll(labels, shifts=-1, dims=-1).view(-1)
        scale = loss_scale.view(-1) if loss_scale is not None else torch.ones_like(target, dtype=torch.float)
        if self.digit_ids is None:
            tokenizer = trainer.template.tokenizer
            self.digit_ids = torch.tensor([tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)])
        digit_ids = self.digit_ids.to(target.device)
        weighted = scale > 1.0
        mode = "train" if trainer.model.training else "eval"
        if weighted.any():
            # Unweighted CE on physics tokens: the number to watch for "is it learning the physics part".
            trainer.custom_metrics[mode]["ce_physics"].update((token_loss.view(-1)[weighted] / scale[weighted]).mean().detach())
        mask = torch.isin(target, digit_ids) & (weighted if loss_scale is not None else True)
        ntl_weight = float(os.environ.get("P2P_NTL_WEIGHT", "0"))
        if ntl_weight > 0 and mask.any():
            logits = outputs.logits.view(-1, outputs.logits.shape[-1])[mask][:, digit_ids].float()
            truth = (target[mask].unsqueeze(1) == digit_ids).float().argmax(dim=1)
            distance = (torch.arange(10, device=logits.device).unsqueeze(0) - truth.unsqueeze(1)).abs().float()
            ntl = (logits.softmax(dim=-1) * distance).sum(dim=-1)
            ntl_loss = (ntl * scale[mask]).sum() / num_items_in_batch
            trainer.custom_metrics[mode]["ntl"].update((ntl.mean()).detach())
            loss = loss + ntl_weight * ntl_loss
        return loss


loss_scale_map["physics"] = PhysicsLossScale
loss_map["physics_ntl"] = PhysicsNTLLoss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="c0_video_only | c1_full_aux | c2_no_position | c3_no_material")
    parser.add_argument("--data", type=Path, default=Path("data/sft_v1"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--loss", choices=("ce", "weighted", "weighted_ntl"), default="weighted")
    parser.add_argument("--w-physics", type=float, default=2.0)
    parser.add_argument("--w-number", type=float, default=3.0)
    parser.add_argument("--ntl-weight", type=float, default=0.3)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=12288)
    parser.add_argument("--attn-impl", default="sdpa", help="sdpa or flash_attn (if installed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="pixels2physics", help="empty string: log to tensorboard only")
    parser.add_argument("--run-name", help="wandb run name (default: output dir name)")
    parser.add_argument("swift_args", nargs=argparse.REMAINDER, help="extra ms-swift arguments after --")
    args = parser.parse_args()

    dataset = args.data / "swift" / args.config
    if not (dataset / "train.jsonl").is_file():
        parser.error(f"missing {dataset}/train.jsonl; run build_sft_data.py first")
    output_dir = args.output_dir or Path("runs") / f"{args.config}__{args.loss}"
    report_to = ["tensorboard"]
    if args.wandb_project:
        report_to.append("wandb")
        # Tags make the runs filterable by experiment axis in the wandb UI.
        os.environ.update(WANDB_PROJECT=args.wandb_project, WANDB_TAGS=f"{args.config},{args.loss}", WANDB_LOG_MODEL="false")
    os.environ.update(
        P2P_W_PHYSICS=str(args.w_physics),
        P2P_W_NUMBER=str(args.w_number),
        P2P_NTL_WEIGHT=str(args.ntl_weight if args.loss == "weighted_ntl" else 0.0),
    )
    swift_args = [
        "--model", args.model, "--use_hf", "true",
        "--train_type", "lora", "--lora_rank", str(args.lora_rank), "--lora_alpha", str(2 * args.lora_rank),
        "--lora_dropout", "0.05", "--target_modules", "all-linear", "--freeze_vit", "true", "--freeze_aligner", "true",
        "--torch_dtype", "bfloat16", "--attn_impl", args.attn_impl, "--gradient_checkpointing", "true",
        "--dataset", str(dataset / "train.jsonl"), "--val_dataset", str(dataset / "validation.jsonl"),
        "--max_length", str(args.max_length), "--truncation_strategy", "delete",
        "--num_train_epochs", str(args.epochs), "--learning_rate", str(args.lr),
        "--lr_scheduler_type", "cosine", "--warmup_ratio", "0.05",
        "--per_device_train_batch_size", str(args.batch_size), "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--eval_strategy", "steps", "--eval_steps", "25", "--save_steps", "25", "--save_total_limit", "3",
        "--load_best_model_at_end", "true", "--logging_steps", "5",
        "--dataloader_num_workers", "4", "--dataset_num_proc", "4", "--report_to", *report_to,
        "--run_name", args.run_name or output_dir.name, "--seed", str(args.seed), "--output_dir", str(output_dir), "--external_plugins", __file__,
    ]
    if args.loss != "ce":
        swift_args += ["--loss_scale", "physics", "--loss_type", "physics_ntl"]
    extra = args.swift_args[1:] if args.swift_args[:1] == ["--"] else args.swift_args

    from swift.pipelines import sft_main

    sft_main(swift_args + extra)


if __name__ == "__main__":
    main()
