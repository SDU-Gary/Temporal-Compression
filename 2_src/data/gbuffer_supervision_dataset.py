from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _pick_first_key(payload: Dict[str, Any], candidates: Sequence[str]) -> tuple[Any, str]:
    for key in candidates:
        if key in payload:
            return payload[key], key
    raise KeyError(f"Missing required key. candidates={list(candidates)}")


def _to_float_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().to(dtype=torch.float32, device="cpu")
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=0.0, max=1.0)
    threshold = 0.04045
    low = x / 12.92
    high = torch.pow((x + 0.055) / 1.055, 2.4)
    return torch.where(x <= threshold, low, high)


def _dedupe_existing(paths: Sequence[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _read_manifest_paths(manifest_path: Path) -> list[Path]:
    entries: list[Path] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        path = Path(text)
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        entries.append(path)
    return _dedupe_existing(entries)


def _to_path_list(root: Path) -> list[Path]:
    path = Path(root)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in {".pt", ".npz"}:
            return [path]
        if path.name == "samples_manifest.txt":
            return _read_manifest_paths(path)
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                manifest = payload.get("sample_manifest", "")
                if isinstance(manifest, str) and manifest.strip():
                    manifest_path = Path(manifest)
                    if not manifest_path.is_absolute():
                        manifest_path = (path.parent / manifest_path).resolve()
                    if manifest_path.exists():
                        return _read_manifest_paths(manifest_path)
                out_dir = payload.get("output_dir", "")
                if isinstance(out_dir, str) and out_dir.strip():
                    out_dir_path = Path(out_dir)
                    if not out_dir_path.is_absolute():
                        out_dir_path = (path.parent / out_dir_path).resolve()
                    if out_dir_path.exists():
                        return _to_path_list(out_dir_path)
        return []

    direct = sorted(path.glob("*.npz")) + sorted(path.glob("*.pt"))
    if direct:
        return _dedupe_existing(direct)

    manifest_path = path / "samples_manifest.txt"
    if manifest_path.exists():
        manifest_files = _read_manifest_paths(manifest_path)
        if manifest_files:
            return manifest_files

    nested = sorted(path.rglob("*.npz")) + sorted(path.rglob("*.pt"))
    return _dedupe_existing(nested)


def _extract_scalar_int(value: Any, *, label: str) -> int:
    arr = np.asarray(value)
    if arr.size <= 0:
        raise ValueError(f"{label} must be a scalar/int-like value, got empty array")
    return int(arr.reshape(-1)[0])


_FRAME_RE = re.compile(r"[fF](\d+)")


def _parse_frame_idx_from_path(path: Path) -> int:
    name = path.stem
    match = _FRAME_RE.search(name)
    if match is not None:
        return int(match.group(1))
    return -1


class GBufferSupervisionDataset(Dataset):
    """Offline GBuffer supervision dataset for auxiliary image loss."""

    def __init__(
        self,
        data_root: str | Path,
        pixel_sample_count: int = 8192,
        *,
        strict_keys: bool = True,
        pos_key: str = "posW",
        normal_key: str = "normW",
        albedo_key: str = "albedo",
        gt_linear_key: str = "gt_linear",
        light_params_key: str = "light_params",
        light_mask_key: str = "light_mask",
        valid_mask_key: str = "valid_mask",
        frame_idx_key: str = "frame_idx",
        config_idx_key: str = "config_idx",
        require_gt_linear: bool = True,
        require_light_params: bool = True,
        gt_color_space: str = "linear",
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"GBuffer dataset root not found: {self.data_root}")
        self.files = _to_path_list(self.data_root)
        if not self.files:
            raise FileNotFoundError(f"No .pt/.npz samples found under: {self.data_root}")
        self.pixel_sample_count = max(1, int(pixel_sample_count))
        self.strict_keys = bool(strict_keys)
        self.pos_key = str(pos_key).strip() or "posW"
        self.normal_key = str(normal_key).strip() or "normW"
        self.albedo_key = str(albedo_key).strip() or "albedo"
        self.gt_linear_key = str(gt_linear_key).strip() or "gt_linear"
        self.light_params_key = str(light_params_key).strip() or "light_params"
        self.light_mask_key = str(light_mask_key).strip() or "light_mask"
        self.valid_mask_key = str(valid_mask_key).strip() or "valid_mask"
        self.frame_idx_key = str(frame_idx_key).strip() or "frame_idx"
        self.config_idx_key = str(config_idx_key).strip() or "config_idx"
        self.require_gt_linear = bool(require_gt_linear)
        self.require_light_params = bool(require_light_params)
        self.gt_color_space = str(gt_color_space).strip().lower() or "linear"
        if self.gt_color_space not in {"linear", "srgb"}:
            raise ValueError(f"Unsupported gt_color_space: {self.gt_color_space}")

    def __len__(self) -> int:
        return len(self.files)

    def _load_payload(self, path: Path) -> Dict[str, Any]:
        if path.suffix.lower() == ".pt":
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected dict payload in {path}, got {type(obj)}")
            return obj
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as npz_obj:
                return {k: npz_obj[k] for k in npz_obj.files}
        raise ValueError(f"Unsupported sample format: {path}")

    @staticmethod
    def _flatten_rgb(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.reshape(-1, x.shape[-1])
        if x.ndim == 2:
            return x
        raise ValueError(f"Expected tensor with ndim 2 or 3, got shape={tuple(x.shape)}")

    def _sample_pixels(self, count: int) -> torch.Tensor:
        target = int(self.pixel_sample_count)
        if count <= target:
            return torch.randint(0, count, (target,), dtype=torch.long)
        return torch.randperm(count, dtype=torch.long)[:target]

    def _pick_required(
        self,
        payload: Dict[str, Any],
        *,
        primary_key: str,
        fallback_keys: Sequence[str],
        label: str,
    ) -> tuple[Any, str]:
        if primary_key in payload:
            return payload[primary_key], primary_key
        if self.strict_keys:
            raise KeyError(
                f"Missing required key '{primary_key}' for {label}. "
                f"strict_keys=True prevents fallback keys {list(fallback_keys)}"
            )
        return _pick_first_key(payload, (primary_key, *fallback_keys))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_path = self.files[int(index)]
        payload = self._load_payload(sample_path)

        pos_obj, _ = self._pick_required(
            payload,
            primary_key=self.pos_key,
            fallback_keys=("pos", "world_pos"),
            label="pos",
        )
        normal_obj, _ = self._pick_required(
            payload,
            primary_key=self.normal_key,
            fallback_keys=("normal", "world_normal"),
            label="normal",
        )
        albedo_obj, _ = self._pick_required(
            payload,
            primary_key=self.albedo_key,
            fallback_keys=("diffuse", "base_color", "diffuseOpacity"),
            label="albedo",
        )
        pos = self._flatten_rgb(_to_float_tensor(pos_obj))
        normal = self._flatten_rgb(_to_float_tensor(normal_obj))
        albedo = self._flatten_rgb(_to_float_tensor(albedo_obj))
        gt_linear: torch.Tensor | None = None
        if self.require_gt_linear:
            gt_obj, _ = self._pick_required(
                payload,
                primary_key=self.gt_linear_key,
                fallback_keys=("gt", "target", "gt_image"),
                label="gt_linear",
            )
            gt_linear = self._flatten_rgb(_to_float_tensor(gt_obj))
            if self.gt_color_space == "srgb":
                gt_linear = _srgb_to_linear(gt_linear)
        else:
            gt_obj = payload.get(self.gt_linear_key, None)
            if gt_obj is not None:
                gt_linear = self._flatten_rgb(_to_float_tensor(gt_obj))
                if self.gt_color_space == "srgb":
                    gt_linear = _srgb_to_linear(gt_linear)

        if pos.shape[-1] > 3:
            pos = pos[:, :3]
        if normal.shape[-1] > 3:
            normal = normal[:, :3]
        if albedo.shape[-1] > 3:
            albedo = albedo[:, :3]
        if gt_linear is not None and gt_linear.shape[-1] > 3:
            gt_linear = gt_linear[:, :3]

        valid_mask_obj = payload.get(self.valid_mask_key, None)
        if valid_mask_obj is not None:
            valid_mask = _to_float_tensor(valid_mask_obj).reshape(-1) > 0.5
        else:
            valid_mask = torch.ones(pos.shape[0], dtype=torch.bool)

        if valid_mask.shape[0] != pos.shape[0]:
            raise ValueError(
                f"valid_mask size mismatch for {sample_path}: "
                f"{valid_mask.shape[0]} vs {pos.shape[0]}"
            )

        pos = pos[valid_mask]
        normal = normal[valid_mask]
        albedo = albedo[valid_mask]
        if gt_linear is not None:
            gt_linear = gt_linear[valid_mask]
        if pos.numel() == 0:
            raise ValueError(f"No valid pixels after masking: {sample_path}")

        idx = self._sample_pixels(int(pos.shape[0]))
        pos = pos[idx]
        normal = normal[idx]
        albedo = albedo[idx]
        if gt_linear is not None:
            gt_linear = gt_linear[idx]

        light_params: torch.Tensor | None = None
        light_mask: torch.Tensor | None = None
        if self.require_light_params:
            light_params_obj, _ = self._pick_required(
                payload,
                primary_key=self.light_params_key,
                fallback_keys=(),
                label="light_params",
            )
            light_params = _to_float_tensor(light_params_obj)
            if light_params.ndim == 1:
                light_params = light_params.unsqueeze(0)
            if light_params.ndim != 2:
                raise ValueError(f"light_params must be [S,F] or [F], got shape={tuple(light_params.shape)}")
            light_mask_obj = payload.get(self.light_mask_key, None)
            if light_mask_obj is None:
                light_mask = torch.ones(light_params.shape[0], dtype=torch.float32)
            else:
                light_mask = _to_float_tensor(light_mask_obj).reshape(-1)
                if light_mask.shape[0] != light_params.shape[0]:
                    raise ValueError(
                        f"light_mask size mismatch for {sample_path}: "
                        f"{light_mask.shape[0]} vs {light_params.shape[0]}"
                    )
        else:
            light_params_obj = payload.get(self.light_params_key, None)
            if light_params_obj is not None:
                light_params = _to_float_tensor(light_params_obj)
                if light_params.ndim == 1:
                    light_params = light_params.unsqueeze(0)
                if light_params.ndim != 2:
                    raise ValueError(f"light_params must be [S,F] or [F], got shape={tuple(light_params.shape)}")
                light_mask_obj = payload.get(self.light_mask_key, None)
                if light_mask_obj is None:
                    light_mask = torch.ones(light_params.shape[0], dtype=torch.float32)
                else:
                    light_mask = _to_float_tensor(light_mask_obj).reshape(-1)
                    if light_mask.shape[0] != light_params.shape[0]:
                        raise ValueError(
                            f"light_mask size mismatch for {sample_path}: "
                            f"{light_mask.shape[0]} vs {light_params.shape[0]}"
                        )

        frame_idx = -1
        if self.frame_idx_key in payload:
            frame_idx = _extract_scalar_int(payload[self.frame_idx_key], label=self.frame_idx_key)
        elif "frame" in payload:
            frame_idx = _extract_scalar_int(payload["frame"], label="frame")
        elif "frame_id" in payload:
            frame_idx = _extract_scalar_int(payload["frame_id"], label="frame_id")
        else:
            frame_idx = _parse_frame_idx_from_path(sample_path)

        config_idx = -1
        if self.config_idx_key in payload:
            config_idx = _extract_scalar_int(payload[self.config_idx_key], label=self.config_idx_key)

        out: Dict[str, Any] = {
            "posW": pos.contiguous(),
            "normW": normal.contiguous(),
            "albedo": albedo.contiguous(),
            "frame_idx": torch.tensor(int(frame_idx), dtype=torch.long),
            "config_idx": torch.tensor(int(config_idx), dtype=torch.long),
            "source_path": str(sample_path),
        }
        if gt_linear is not None:
            out["gt_linear"] = gt_linear.contiguous()
        if light_params is not None:
            out["light_params"] = light_params.contiguous()
        if light_mask is not None:
            out["light_mask"] = light_mask.contiguous()
        return out


def create_gbuffer_supervision_dataloader(
    *,
    data_root: str | Path,
    pixel_sample_count: int,
    strict_keys: bool = True,
    pos_key: str = "posW",
    normal_key: str = "normW",
    albedo_key: str = "albedo",
    gt_linear_key: str = "gt_linear",
    light_params_key: str = "light_params",
    light_mask_key: str = "light_mask",
    valid_mask_key: str = "valid_mask",
    frame_idx_key: str = "frame_idx",
    config_idx_key: str = "config_idx",
    require_gt_linear: bool = True,
    require_light_params: bool = True,
    gt_color_space: str = "linear",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    dataset = GBufferSupervisionDataset(
        data_root=data_root,
        pixel_sample_count=pixel_sample_count,
        strict_keys=strict_keys,
        pos_key=pos_key,
        normal_key=normal_key,
        albedo_key=albedo_key,
        gt_linear_key=gt_linear_key,
        light_params_key=light_params_key,
        light_mask_key=light_mask_key,
        valid_mask_key=valid_mask_key,
        frame_idx_key=frame_idx_key,
        config_idx_key=config_idx_key,
        require_gt_linear=require_gt_linear,
        require_light_params=require_light_params,
        gt_color_space=gt_color_space,
    )
    loader_kwargs: Dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": bool(shuffle),
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(dataset, **loader_kwargs)
