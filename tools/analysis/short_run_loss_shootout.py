#!/usr/bin/env python3
"""Few-step loss shootout from intermediate checkpoints.

Purpose:
- Compare candidate loss families under identical short-run budgets.
- Start from neutral/intermediate checkpoints instead of final old-objective basin.
- Record held-out offline metrics and optionally save final checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from probe_loss_redesign_small_experiments import (
    SH_L0_INDICES,
    SH_L1_INDICES,
    build_dataset,
    build_fibonacci_dirs,
    build_random_dirs,
    build_sh_basis_from_dirs,
    charbonnier_loss,
    infer_split_meta,
    load_checkpoint_model,
    maybe_inverse_scaler,
    parse_profile,
    profile_forward,
    render_irradiance_samples,
    render_radiance_mix_loss,
    render_radiance_samples,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run short-run objective shootout from intermediate checkpoints.")
    p.add_argument(
        "--start-checkpoints",
        nargs="+",
        required=True,
        help="Intermediate checkpoints used as starting states.",
    )
    p.add_argument(
        "--objectives",
        nargs="+",
        default=["old_full", "observable_only", "observable_lf", "proposed_hybrid"],
        choices=[
            "old_full",
            "observable_only",
            "observable_lf",
            "proposed_hybrid",
            "soft_consistency_light",
        ],
    )
    p.add_argument("--data-root", default="1_data_generation/output/bistro_clean_v2")
    p.add_argument("--train-split", default="train", choices=["train", "val", "test"])
    p.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    p.add_argument("--train-max-samples", type=int, default=4096)
    p.add_argument("--eval-max-samples", type=int, default=2048)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--radiance-samples", type=int, default=64)
    p.add_argument("--irradiance-samples", type=int, default=64)
    p.add_argument("--teacher-profile", default="soft8_t018")
    p.add_argument("--soft3-profile", default="soft3_t018")
    p.add_argument("--hard3-profile", default="hard3")
    p.add_argument("--save-final-checkpoints", action="store_true")
    p.add_argument(
        "--output-dir",
        default="3_experiments/results/analysis/short_run_loss_shootout/latest",
    )
    return p.parse_args()


def _to_device_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _sample_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2)
    if float(mse.item()) <= 1e-12:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(1.0, device=pred.device, dtype=pred.dtype) / mse).item())


class ShootoutRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = (
            torch.device(args.device)
            if args.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.teacher = parse_profile(args.teacher_profile)
        self.soft3 = parse_profile(args.soft3_profile)
        self.hard3 = parse_profile(args.hard3_profile)
        ref = torch.empty((1, 27), device=self.device, dtype=torch.float32)
        self.radiance_basis = build_sh_basis_from_dirs(
            build_random_dirs(args.radiance_samples, args.seed, ref)
        )
        self.irradiance_basis = build_sh_basis_from_dirs(
            build_fibonacci_dirs(args.irradiance_samples, args.seed + 1, ref)
        )
        self.l0l1_idx = list(SH_L0_INDICES) + list(SH_L1_INDICES)

    def objective_weights(self, objective: str, step: int, total_steps: int) -> Dict[str, float]:
        if objective == "old_full":
            return {"recon_full": 1.0, "radiance": 1.0, "irradiance": 1.0, "recon_l0l1": 0.0, "cons_soft3": 0.0, "cons_hard3": 0.0}
        if objective == "observable_only":
            return {"recon_full": 0.0, "radiance": 1.0, "irradiance": 1.0, "recon_l0l1": 0.0, "cons_soft3": 0.0, "cons_hard3": 0.0}
        if objective == "observable_lf":
            return {"recon_full": 0.0, "radiance": 1.0, "irradiance": 1.0, "recon_l0l1": 0.2, "cons_soft3": 0.0, "cons_hard3": 0.0}
        if objective == "soft_consistency_light":
            soft_ramp = min(1.0, step / max(1.0, total_steps * 0.5))
            return {
                "recon_full": 0.0,
                "radiance": 1.0,
                "irradiance": 1.0,
                "recon_l0l1": 0.2,
                "cons_soft3": 0.02 * soft_ramp,
                "cons_hard3": 0.0,
            }
        if objective == "proposed_hybrid":
            soft_ramp = min(1.0, step / max(1.0, total_steps * 0.33))
            hard_ramp = 0.0
            if step > total_steps * 0.5:
                hard_ramp = min(1.0, (step - total_steps * 0.5) / max(1.0, total_steps * 0.5))
            return {
                "recon_full": 0.0,
                "radiance": 1.0,
                "irradiance": 1.0,
                "recon_l0l1": 0.2,
                "cons_soft3": 0.05 * soft_ramp,
                "cons_hard3": 0.01 * hard_ramp,
            }
        raise KeyError(objective)

    def build_loaders(self, meta: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
        train_ratio, val_ratio = infer_split_meta(meta)
        train_ds = build_dataset(
            self.args.data_root,
            self.args.train_split,
            train_ratio,
            val_ratio,
            max_samples=self.args.train_max_samples,
            seed=self.args.seed,
        )
        eval_ds = build_dataset(
            self.args.data_root,
            self.args.eval_split,
            train_ratio,
            val_ratio,
            max_samples=self.args.eval_max_samples,
            seed=self.args.seed + 17,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader, eval_loader

    def forward_components(
        self,
        model: torch.nn.Module,
        scaler: Any,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pos = batch["probe_position"]
        light_params = batch["light_params"]
        light_mask = batch["light_mask"]
        gt = batch["sh_coeffs"]

        pred_teacher = maybe_inverse_scaler(profile_forward(model, pos, light_params, light_mask, self.teacher), scaler)
        pred_soft3 = maybe_inverse_scaler(profile_forward(model, pos, light_params, light_mask, self.soft3), scaler)
        pred_hard3 = maybe_inverse_scaler(profile_forward(model, pos, light_params, light_mask, self.hard3), scaler)

        gt_rad = render_radiance_samples(gt, self.radiance_basis)
        teacher_rad = render_radiance_samples(pred_teacher, self.radiance_basis)
        hard_rad = render_radiance_samples(pred_hard3, self.radiance_basis)

        gt_irr = render_irradiance_samples(gt, self.irradiance_basis)
        teacher_irr = render_irradiance_samples(pred_teacher, self.irradiance_basis)
        hard_irr = render_irradiance_samples(pred_hard3, self.irradiance_basis)

        return {
            "gt": gt,
            "pred_teacher": pred_teacher,
            "pred_soft3": pred_soft3,
            "pred_hard3": pred_hard3,
            "gt_rad": gt_rad,
            "teacher_rad": teacher_rad,
            "hard_rad": hard_rad,
            "gt_irr": gt_irr,
            "teacher_irr": teacher_irr,
            "hard_irr": hard_irr,
        }

    def objective_and_logs(
        self,
        model: torch.nn.Module,
        scaler: Any,
        batch: Dict[str, torch.Tensor],
        objective: str,
        step: int,
        total_steps: int,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        c = self.forward_components(model, scaler, batch)
        gt = c["gt"]
        pred_teacher = c["pred_teacher"]
        weights = self.objective_weights(objective, step, total_steps)

        loss_recon_full = charbonnier_loss(pred_teacher, gt)
        loss_recon_l0l1 = charbonnier_loss(pred_teacher[..., self.l0l1_idx], gt[..., self.l0l1_idx])
        loss_radiance = render_radiance_mix_loss(pred_teacher, gt, self.radiance_basis, epsilon=1e-3)
        loss_irradiance = charbonnier_loss(c["teacher_irr"], c["gt_irr"])
        loss_cons_soft3 = charbonnier_loss(pred_teacher, c["pred_soft3"])
        loss_cons_hard3 = charbonnier_loss(pred_teacher, c["pred_hard3"])

        total = (
            weights["recon_full"] * loss_recon_full
            + weights["recon_l0l1"] * loss_recon_l0l1
            + weights["radiance"] * loss_radiance
            + weights["irradiance"] * loss_irradiance
            + weights["cons_soft3"] * loss_cons_soft3
            + weights["cons_hard3"] * loss_cons_hard3
        )
        log = {
            "loss_total": float(total.detach().item()),
            "loss_recon_full": float(loss_recon_full.detach().item()),
            "loss_recon_l0l1": float(loss_recon_l0l1.detach().item()),
            "loss_radiance": float(loss_radiance.detach().item()),
            "loss_irradiance": float(loss_irradiance.detach().item()),
            "loss_cons_soft3": float(loss_cons_soft3.detach().item()),
            "loss_cons_hard3": float(loss_cons_hard3.detach().item()),
            "w_recon_full": float(weights["recon_full"]),
            "w_recon_l0l1": float(weights["recon_l0l1"]),
            "w_radiance": float(weights["radiance"]),
            "w_irradiance": float(weights["irradiance"]),
            "w_cons_soft3": float(weights["cons_soft3"]),
            "w_cons_hard3": float(weights["cons_hard3"]),
        }
        return total, log

    @torch.inference_mode()
    def evaluate(self, model: torch.nn.Module, scaler: Any, eval_loader: DataLoader) -> Dict[str, float]:
        sums: Dict[str, float] = {}
        count = 0
        for batch in eval_loader:
            batch = _to_device_batch(batch, self.device)
            c = self.forward_components(model, scaler, batch)
            gt = c["gt"]
            pred_teacher = c["pred_teacher"]
            pred_hard3 = c["pred_hard3"]

            metrics = {
                "teacher_recon_full": float(charbonnier_loss(pred_teacher, gt).item()),
                "teacher_recon_l0l1": float(charbonnier_loss(pred_teacher[..., self.l0l1_idx], gt[..., self.l0l1_idx]).item()),
                "teacher_radiance_mix": float(render_radiance_mix_loss(pred_teacher, gt, self.radiance_basis, epsilon=1e-3).item()),
                "teacher_irradiance": float(charbonnier_loss(c["teacher_irr"], c["gt_irr"]).item()),
                "teacher_proxy_psnr": float(_sample_psnr(c["teacher_rad"], c["gt_rad"])),
                "hard3_proxy_psnr": float(_sample_psnr(c["hard_rad"], c["gt_rad"])),
                "teacher_irr_psnr": float(_sample_psnr(c["teacher_irr"], c["gt_irr"])),
                "hard3_irr_psnr": float(_sample_psnr(c["hard_irr"], c["gt_irr"])),
                "route_gap_soft3": float(charbonnier_loss(pred_teacher, c["pred_soft3"]).item()),
                "route_gap_hard3": float(charbonnier_loss(pred_teacher, pred_hard3).item()),
            }
            bsz = int(gt.shape[0])
            count += bsz
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v * bsz
        denom = max(1, count)
        return {k: float(v / denom) for k, v in sums.items()}

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        meta: Dict[str, Any],
        path: Path,
        step: int,
        objective: str,
    ) -> None:
        payload = {
            "step": int(step),
            "objective": str(objective),
            "model_state_dict": model.state_dict(),
            "meta": meta,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))

    def run_case(self, start_checkpoint: Path, objective: str, case_dir: Path) -> Dict[str, Any]:
        model, meta, scaler = load_checkpoint_model(start_checkpoint, device=self.device)
        model.train()
        train_loader, eval_loader = self.build_loaders(meta)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.args.lr), weight_decay=float(self.args.weight_decay))
        train_iter = cycle(train_loader)

        trace: List[Dict[str, Any]] = []
        initial_metrics = self.evaluate(model, scaler, eval_loader)
        trace.append({"step": 0, "eval": initial_metrics})

        for step in range(1, int(self.args.steps) + 1):
            batch = next(train_iter)
            batch = _to_device_batch(batch, self.device)
            optimizer.zero_grad(set_to_none=True)
            loss, log = self.objective_and_logs(model, scaler, batch, objective, step, int(self.args.steps))
            loss.backward()
            if float(self.args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.args.grad_clip))
            optimizer.step()

            record: Dict[str, Any] = {"step": step, "train": log}
            if step % int(self.args.eval_every) == 0 or step == int(self.args.steps):
                model.eval()
                record["eval"] = self.evaluate(model, scaler, eval_loader)
                model.train()
            trace.append(record)

        final_metrics = trace[-1].get("eval")
        if final_metrics is None:
            model.eval()
            final_metrics = self.evaluate(model, scaler, eval_loader)
            model.train()

        if self.args.save_final_checkpoints:
            self.save_checkpoint(model, meta, case_dir / "final_checkpoint.pt", int(self.args.steps), objective)

        return {
            "start_checkpoint": str(start_checkpoint),
            "objective": objective,
            "steps": int(self.args.steps),
            "lr": float(self.args.lr),
            "trace": trace,
            "initial_eval": initial_metrics,
            "final_eval": final_metrics,
        }

    def run(self) -> Dict[str, Any]:
        out_dir = Path(self.args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        summary: Dict[str, Any] = {
            "device": str(self.device),
            "teacher_profile": self.teacher.name,
            "soft3_profile": self.soft3.name,
            "hard3_profile": self.hard3.name,
            "steps": int(self.args.steps),
            "lr": float(self.args.lr),
            "cases": [],
        }

        for start_ckpt_str in self.args.start_checkpoints:
            start_ckpt = Path(start_ckpt_str).resolve()
            if start_ckpt.name == "checkpoint.pt":
                epoch_name = start_ckpt.parent.name
                exp_name = start_ckpt.parents[3].name if len(start_ckpt.parents) > 3 else start_ckpt.parent.parent.name
                stem = f"{exp_name}__{epoch_name}"
            else:
                stem = start_ckpt.parent.name
            for objective in self.args.objectives:
                case_name = f"{stem}__{objective}"
                case_dir = out_dir / case_name
                case_dir.mkdir(parents=True, exist_ok=True)
                result = self.run_case(start_ckpt, objective, case_dir)
                (case_dir / "trace.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

                initial_eval = result["initial_eval"]
                final_eval = result["final_eval"]
                deltas = {k: float(final_eval[k] - initial_eval[k]) for k in initial_eval.keys()}
                row = {
                    "case_name": case_name,
                    "start_checkpoint": str(start_ckpt),
                    "objective": objective,
                    "initial_eval": initial_eval,
                    "final_eval": final_eval,
                    "delta": deltas,
                }
                summary["cases"].append(row)

        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[OK] short-run shootout written to: {out_dir}")
        return summary


def main() -> None:
    args = parse_args()
    ShootoutRunner(args).run()


if __name__ == "__main__":
    main()
