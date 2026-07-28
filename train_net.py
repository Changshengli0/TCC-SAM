import copy
import itertools
import logging
import math
import os
import time
import warnings
warnings.filterwarnings("ignore")

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

import torch
from tqdm import tqdm
from torch.utils.data import Subset
import random

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog, build_detection_test_loader
from detectron2.data.samplers import RandomSubsetTrainingSampler
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import get_event_storage
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    verify_results,
)
import cv2
import matplotlib.pyplot as plt
import weakref

from datasets import (
    OpenWorldSAM2InstanceDatasetMapper,
    OpenWorldSAM2InstanceDatasetMapperAll,
    OpenWorldSAM2PanopticDatasetMapper,
    OpenWorldSAM2PanopticDatasetMapperAll,
    ScanNetPanoDatasetMapper,
    OpenWorldSAM2SemanticDatasetMapper,
    RefCOCODatasetMapper,
)

from evaluation import (
    # InstanceSegEvaluator,
    COCOPanopticEvaluator,
    SemSegEvaluator,
    GroundingEvaluator
)

from model import (
    add_open_world_sam2_config,
)

from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)

import numpy as np

# Add imports for our new OpenWorldSAM2WithPaliGemma model and mapper
# from datasets.dataset_mappers.open_world_sam_panoptic_dataset_mapper_paligemma import OpenWorldSAM2PanopticDatasetMapperPaliGemma


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _safe_stat_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_stat_max(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(max(values))


def _format_stat(value: Optional[float]) -> str:
    return "None" if value is None else f"{value:.6e}"


def _param_name_matches_output_head(name: str) -> bool:
    keys = (
        "sam_mask_decoder.iou_prediction_head",
        "sam_mask_decoder.output_hypernetworks_mlps",
        "sam_mask_decoder.mask_tokens",
        "sam_mask_decoder.iou_token",
        "sam_mask_decoder.obj_score_token",
        "sam_mask_decoder.conv_s0",
        "sam_mask_decoder.conv_s1",
        "sam_mask_decoder.pred_obj_score_head",
    )
    return any(key in name for key in keys)


def _extract_lora_layer_index(name: str) -> Optional[int]:
    prefixes = (
        "evf_sam2.mm_extractor.beit3.encoder.layers.",
        "evf_sam2.mm_extractor.beit3.layers.",
        "mm_extractor.beit3.encoder.layers.",
        "mm_extractor.beit3.layers.",
    )
    for prefix in prefixes:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        layer_idx_str = suffix.split(".", 1)[0]
        if layer_idx_str.isdigit():
            return int(layer_idx_str)
    return None


def _ows_compat_is_baseline_param_name(name: str) -> bool:
    if name == "positional_tokens":
        return True
    if name.startswith("text_hidden_fcs."):
        return True
    if name.startswith("cross_attention_transformer."):
        return True
    prompt_roots = (
        "evf_sam2.visual_model.sam_prompt_encoder.",
        "visual_model.sam_prompt_encoder.",
    )
    mask_roots = (
        "evf_sam2.visual_model.sam_mask_decoder.",
        "visual_model.sam_mask_decoder.",
    )
    if any(name.startswith(root) for root in prompt_roots):
        return not (
            name.startswith("evf_sam2.visual_model.sam_prompt_encoder.pe_layer.")
            or name.startswith("visual_model.sam_prompt_encoder.pe_layer.")
        )
    if any(name.startswith(root) for root in mask_roots):
        return not (
            name.startswith("evf_sam2.visual_model.sam_mask_decoder.transformer.")
            or name.startswith("visual_model.sam_mask_decoder.transformer.")
        )
    return False


SLOT_PARAM_MATCH_KEYS = (
    "fusion.semantic_slot_extractor",
    "fusion.slot_routes",
    "fusion.summarizer",
)

SLOT_GRAD_PROBE_NAMES = (
    "fusion.semantic_slot_extractor.slot_queries",
    "fusion.semantic_slot_extractor.q_proj.weight",
    "fusion.semantic_slot_extractor.k_proj.weight",
    "fusion.semantic_slot_extractor.v_proj.weight",
    "fusion.slot_routes.0.q_proj.weight",
    "fusion.slot_routes.0.refine.refine.2.weight",
    "fusion.slot_routes.0.refine.refine.2.bias",
    "fusion.slot_routes.1.q_proj.weight",
    "fusion.slot_routes.1.refine.refine.2.weight",
    "fusion.slot_routes.1.refine.refine.2.bias",
    "fusion.slot_routes.2.refine.refine.2.weight",
    "fusion.slot_routes.3.refine.refine.2.weight",
)


def _is_slot_param_name(name: str) -> bool:
    return any(key in name for key in SLOT_PARAM_MATCH_KEYS)


def _format_debug_value(value: Any) -> str:
    if value is None:
        return "None"
    if torch.is_tensor(value):
        v = value.detach().float().cpu()
        if v.numel() == 1:
            return f"{float(v.item()):.6e}"
        flat = v.flatten()
        preview = [float(x) for x in flat[:8].tolist()]
        return f"shape={tuple(v.shape)} preview={preview}"
    if isinstance(value, (list, tuple)):
        preview = list(value[:8])
        return f"{type(value).__name__}={preview}"
    return str(value)


def _grad_abs_mean(param) -> Optional[float]:
    if param is None or param.grad is None:
        return None
    return float(param.grad.detach().abs().mean().item())


def _grad_norm(param) -> Optional[float]:
    if param is None or param.grad is None:
        return None
    return float(param.grad.detach().norm().item())


def _format_optional_float(value: Optional[float]) -> str:
    return "None" if value is None else f"{value:.6e}"


def _log_slot_mech_grad_debug_once(trainer, model) -> None:
    if not bool(getattr(trainer, "slot_mech_grad_debug", False)):
        return
    if bool(getattr(trainer, "_slot_mech_grad_debug_logged", False)):
        return
    model_obj = _unwrap_model(model)
    fusion = getattr(model_obj, "fusion", None)
    logger = logging.getLogger("detectron2")
    if fusion is None:
        logger.info("[SLOT-MECH-GRAD] fusion=None")
        trainer._slot_mech_grad_debug_logged = True
        return

    slot_extractor = getattr(fusion, "semantic_slot_extractor", None)
    slot_queries = getattr(slot_extractor, "slot_queries", None) if slot_extractor is not None else None
    logger.info(
        "[SLOT-MECH-GRAD] iter=%s fusion.semantic_slot_extractor.slot_queries.grad_norm=%s",
        getattr(trainer, "iter", None),
        _format_optional_float(_grad_norm(slot_queries)),
    )
    for idx, route in enumerate(getattr(fusion, "slot_routes", [])):
        last_refine = getattr(getattr(route, "refine", None), "refine", [None])[-1]
        refine_weight = getattr(last_refine, "weight", None)
        refine_bias = getattr(last_refine, "bias", None)
        q_proj_weight = getattr(getattr(route, "q_proj", None), "weight", None)
        logger.info(
            "[SLOT-MECH-GRAD] iter=%s route=%d refine_last_weight_grad_abs_mean=%s "
            "refine_last_bias_grad_abs_mean=%s q_proj_weight_grad_abs_mean=%s",
            getattr(trainer, "iter", None),
            int(idx),
            _format_optional_float(_grad_abs_mean(refine_weight)),
            _format_optional_float(_grad_abs_mean(refine_bias)),
            _format_optional_float(_grad_abs_mean(q_proj_weight)),
        )
    trainer._slot_mech_grad_debug_logged = True


def _collect_param_rows(named_params: Dict[str, torch.nn.Parameter], names: List[str]) -> Dict[str, Any]:
    grad_norms: List[float] = []
    param_norms: List[float] = []
    param_count = 0
    grad_count = 0
    for name in names:
        p = named_params.get(name)
        if p is None:
            continue
        param_count += p.numel()
        try:
            param_norms.append(float(p.detach().float().norm().item()))
        except Exception:
            pass
        if p.grad is None:
            continue
        try:
            grad_norms.append(float(p.grad.detach().float().norm().item()))
            grad_count += 1
        except Exception:
            continue
    return {
        "param_count": int(param_count),
        "grad_param_count": int(grad_count),
        "grad_norm_mean": _safe_stat_mean(grad_norms),
        "grad_norm_max": _safe_stat_max(grad_norms),
        "param_norm_mean": _safe_stat_mean(param_norms),
        "param_norm_max": _safe_stat_max(param_norms),
    }

class _MDFAMPTrainer(AMPTrainer):
    """AMP trainer with explicit no-grad diagnostics for MDF adapter-only optimization."""

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    def _collect_optimizer_grad_stats(self):
        total = 0
        with_grad = 0
        nonzero = 0
        for group in self.optimizer.param_groups:
            for p in group.get("params", []):
                total += 1
                g = p.grad
                if g is None:
                    continue
                with_grad += 1
                try:
                    if float(g.detach().abs().sum().item()) > 0.0:
                        nonzero += 1
                except Exception:
                    pass
        return {
            "optimizer_total_params": int(total),
            "optimizer_params_with_grad": int(with_grad),
            "optimizer_params_with_nonzero_grad": int(nonzero),
        }

    def _collect_named_param_grad_probe(self, model_obj, names: List[str]):
        named_params = dict(model_obj.named_parameters())
        opt_param_ids = {id(p) for g in self.optimizer.param_groups for p in g.get("params", [])}
        rows = []
        with_grad_cnt = 0
        for n in names:
            p = named_params.get(n, None)
            if p is None:
                rows.append(
                    {
                        "name": n,
                        "exists": False,
                        "requires_grad": False,
                        "grad_is_none": True,
                        "grad_norm": None,
                        "is_in_optimizer": False,
                    }
                )
                continue
            g = p.grad
            grad_is_none = g is None
            grad_norm = None
            grad_all_finite = None
            if g is not None:
                try:
                    grad_norm = float(g.detach().norm().item())
                except Exception:
                    grad_norm = None
                try:
                    grad_all_finite = bool(torch.isfinite(g.detach()).all().item())
                except Exception:
                    grad_all_finite = None
                with_grad_cnt += 1
            rows.append(
                {
                    "name": n,
                    "exists": True,
                    "requires_grad": bool(p.requires_grad),
                    "grad_is_none": bool(grad_is_none),
                    "grad_norm": grad_norm,
                    "grad_all_finite": grad_all_finite,
                    "is_in_optimizer": bool(id(p) in opt_param_ids),
                }
            )
        return rows, with_grad_cnt

    def _slot_anomaly_enabled(self) -> bool:
        return bool(getattr(self, "slot_detect_anomaly", False))

    def _log_slot_anomaly_once(self):
        if not self._slot_anomaly_enabled():
            return
        if bool(getattr(self, "_slot_anomaly_logged", False)):
            return
        logging.getLogger("detectron2").info("[SLOT-ANOMALY] detect_anomaly=True (forward+backward)")
        self._slot_anomaly_logged = True

    def run_step(self):
        assert self.model.training, "[MDF] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        self._log_slot_anomaly_once()
        with torch.autograd.set_detect_anomaly(self._slot_anomaly_enabled()):
            with torch.cuda.amp.autocast():
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

            self.optimizer.zero_grad()
            self.grad_scaler.scale(losses).backward()

        _log_slot_mech_grad_debug_once(self, self.model)
        model_obj = self._unwrap_model(self.model)
        eftpg_active = bool(
            getattr(model_obj, "eftpg_on", False) and getattr(model_obj, "eftpg_train_on", False)
        )
        if eftpg_active and hasattr(model_obj, "get_eftpg_trainable_param_names"):
            eftpg_probe_names = [
                "eftpg_visual_proj.weight",
                "eftpg_text_proj.weight",
                "eftpg_cat_fuse.0.weight",
                "eftpg_attr_fuse.0.weight",
                "eftpg_disamb_fuse.0.weight",
                "eftpg_cat_to_prompt.weight",
                "eftpg_attr_to_prompt.weight",
                "eftpg_disamb_to_prompt.weight",
                "eftpg_scale",
            ]
            eftpg_rows, eftpg_with_grad = self._collect_named_param_grad_probe(model_obj, eftpg_probe_names)
            if not hasattr(self, "_eftpg_grad_probe_logged"):
                self._eftpg_grad_probe_logged = False
            if not self._eftpg_grad_probe_logged:
                logger = logging.getLogger("detectron2")
                for row in eftpg_rows:
                    logger.info(
                        "[EFTPG-GRAD] name=%s exists=%s requires_grad=%s grad_is_none=%s grad_norm=%s grad_all_finite=%s is_in_optimizer=%s",
                        row["name"],
                        row["exists"],
                        row["requires_grad"],
                        row["grad_is_none"],
                        "None" if row["grad_norm"] is None else f"{row['grad_norm']:.6e}",
                        row.get("grad_all_finite", None),
                        row["is_in_optimizer"],
                    )
                eftpg_dbg = {}
                if hasattr(model_obj, "_last_eftpg_debug") and isinstance(model_obj._last_eftpg_debug, dict):
                    eftpg_dbg = model_obj._last_eftpg_debug
                logger.info(
                    "[EFTPG-FINITE] eftpg_debug_available=%s visual_summary_isfinite=%s "
                    "fused_cat_isfinite=%s fused_attr_isfinite=%s fused_disamb_isfinite=%s "
                    "p_cat_isfinite=%s p_attr_isfinite=%s p_disamb_isfinite=%s "
                    "prompt_feat_after_isfinite=%s injection_output_isfinite=%s first_nonfinite_stage=%s",
                    isinstance(eftpg_dbg, dict),
                    eftpg_dbg.get("visual_summary_isfinite", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("fused_cat_isfinite", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("fused_attr_isfinite", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("fused_disamb_isfinite", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("p_cat_isfinite", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("p_attr_isfinite", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("p_disamb_isfinite", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("prompt_feat_after_isfinite", eftpg_dbg.get("sparse_prompt_after_isfinite", None))
                    if isinstance(eftpg_dbg, dict)
                    else None,
                    eftpg_dbg.get("injection_output_isfinite", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("first_nonfinite_stage", None) if isinstance(eftpg_dbg, dict) else None,
                )
                logger.info(
                    "[EFTPG-ALIGN] row_count=%s total_prompt_count=%s aligned_prompt_count=%s "
                    "rowwise_injection_allowed=%s skip_reason_alignment=%s valid_prompt_count=%s valid_prompt_indices=%s",
                    eftpg_dbg.get("row_count", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("total_prompt_count", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("aligned_prompt_count", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("rowwise_injection_allowed", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("skip_reason_alignment", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("valid_prompt_count", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("valid_prompt_indices", None) if isinstance(eftpg_dbg, dict) else None,
                )
                logger.info(
                    "[EFTPG-BWD] first_nonfinite_backward_stage=%s p_cat_grad_isfinite=%s p_attr_grad_isfinite=%s "
                    "p_disamb_grad_isfinite=%s prompt_feat_before_grad_isfinite=%s "
                    "prompt_feat_after_grad_isfinite=%s injection_output_grad_isfinite=%s",
                    eftpg_dbg.get("first_nonfinite_backward_stage", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("p_cat_grad_isfinite", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("p_attr_grad_isfinite", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("p_disamb_grad_isfinite", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("prompt_feat_before_grad_isfinite", eftpg_dbg.get("sparse_prompt_before_grad_isfinite", None))
                    if isinstance(eftpg_dbg, dict)
                    else None,
                    eftpg_dbg.get("prompt_feat_after_grad_isfinite", eftpg_dbg.get("sparse_prompt_after_grad_isfinite", None))
                    if isinstance(eftpg_dbg, dict)
                    else None,
                    eftpg_dbg.get("injection_output_grad_isfinite", None) if isinstance(eftpg_dbg, dict) else None,
                )
                logger.info(
                    "[EFTPG-INJECT-LEVEL] inject_level_name=%s inject_tensor_shape=%s inject_tensor_first_dim=%s "
                    "total_prompt_count=%s is_real_prompt_level=%s",
                    eftpg_dbg.get("inject_level_name", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("inject_tensor_shape", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("inject_tensor_first_dim", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("total_prompt_count", None) if isinstance(eftpg_dbg, dict) else None,
                    eftpg_dbg.get("is_real_prompt_level", None) if isinstance(eftpg_dbg, dict) else None,
                )
                self._eftpg_grad_probe_logged = True
            non_finite_grad_names = [
                row["name"]
                for row in eftpg_rows
                if row.get("exists", False)
                and (not row.get("grad_is_none", True))
                and (row.get("grad_all_finite", True) is False)
            ]
            if len(non_finite_grad_names) > 0:
                eftpg_dbg = {}
                if hasattr(model_obj, "_last_eftpg_debug") and isinstance(model_obj._last_eftpg_debug, dict):
                    eftpg_dbg = model_obj._last_eftpg_debug
                bwd_stage = eftpg_dbg.get("first_nonfinite_backward_stage", None) if isinstance(eftpg_dbg, dict) else None
                if bwd_stage in (
                    "prompt_feat_after_grad_nonfinite",
                    "sparse_prompt_after_grad_nonfinite",
                    "injection_output_grad_nonfinite",
                ):
                    raise ValueError(
                        "[EFTPG] Non-finite backward starts after prompt feature modulation. "
                        "Check prompt alignment / decoder prompt compatibility. "
                        f"non_finite_params={non_finite_grad_names} eftpg_debug={eftpg_dbg}"
                    )
                raise ValueError(
                    "[EFTPG] Non-finite gradients detected on iter=0. "
                    "Check EFTPG prompt fusion / injection numerics. "
                    f"non_finite_params={non_finite_grad_names} eftpg_debug={eftpg_dbg}"
                )
            if eftpg_with_grad == 0:
                eftpg_dbg = {}
                if hasattr(model_obj, "_last_eftpg_debug") and isinstance(model_obj._last_eftpg_debug, dict):
                    eftpg_dbg = model_obj._last_eftpg_debug
                raise ValueError(
                    "[EFTPG] No gradients reached EFTPG parameters on iter=0. "
                    "Check prompt injection graph / no-op conditions. "
                    f"eftpg_debug={eftpg_dbg}"
                )

        pdqg_active = bool(
            getattr(model_obj, "pdqg_on", False) and getattr(model_obj, "pdqg_train_on", False)
        )
        if pdqg_active and hasattr(model_obj, "get_pdqg_trainable_param_names"):
            probe_names = [
                "pdqg_cat_proj.weight",
                "pdqg_attr_proj.weight",
                "pdqg_disamb_proj.weight",
                "pdqg_cat_to_query.weight",
                "pdqg_attr_to_query.weight",
                "pdqg_disamb_to_query.weight",
                "pdqg_cat_tokens",
                "pdqg_attr_tokens",
                "pdqg_disamb_tokens",
                "pdqg_scale",
                "pdqg_disamb_score_proj.weight",
                "pdqg_disamb_score_region_proj.weight",
                "pdqg_disamb_score_head.0.weight",
                "pdqg_disamb_score_head.2.weight",
                "fusion.disamb_summary_mlp.0.weight",
                "fusion.disamb_summary_mlp.2.weight",
            ]
            probe_rows, pdqg_with_grad = self._collect_named_param_grad_probe(model_obj, probe_names)
            scoring_probe_names = {
                "pdqg_disamb_score_proj.weight",
                "pdqg_disamb_score_region_proj.weight",
                "pdqg_disamb_score_head.0.weight",
                "pdqg_disamb_score_head.2.weight",
            }
            scoring_with_grad = 0
            for row in probe_rows:
                if row.get("name") in scoring_probe_names and (not row.get("grad_is_none", True)):
                    scoring_with_grad += 1
            if not hasattr(self, "_pdqg_grad_probe_logged"):
                self._pdqg_grad_probe_logged = False
            if not self._pdqg_grad_probe_logged:
                logger = logging.getLogger("detectron2")
                for row in probe_rows:
                    logger.info(
                        "[PDQG-GRAD] name=%s exists=%s requires_grad=%s grad_is_none=%s grad_norm=%s is_in_optimizer=%s",
                        row["name"],
                        row["exists"],
                        row["requires_grad"],
                        row["grad_is_none"],
                        "None" if row["grad_norm"] is None else f"{row['grad_norm']:.6e}",
                        row["is_in_optimizer"],
                    )
                pdqg_v2_dbg = {}
                if hasattr(model_obj, "_last_pdqg_v2_debug") and isinstance(model_obj._last_pdqg_v2_debug, dict):
                    pdqg_v2_dbg = model_obj._last_pdqg_v2_debug
                loss_pdqg = loss_dict.get("loss_pdqg_ambig_rank", None) if isinstance(loss_dict, dict) else None
                loss_pdqg_val = None
                loss_pdqg_isfinite = None
                if isinstance(loss_pdqg, torch.Tensor):
                    try:
                        loss_pdqg_val = float(loss_pdqg.detach().item())
                        loss_pdqg_isfinite = bool(torch.isfinite(loss_pdqg.detach()).item())
                    except Exception:
                        loss_pdqg_val = None
                        loss_pdqg_isfinite = None
                elif isinstance(loss_pdqg, (float, int)):
                    loss_pdqg_val = float(loss_pdqg)
                    loss_pdqg_isfinite = bool(math.isfinite(loss_pdqg_val))
                active_cnt = int(pdqg_v2_dbg.get("pdqg_v2_active_count", 0))
                total_cnt = int(pdqg_v2_dbg.get("pdqg_v2_total_count", 0))
                skipped_cnt = int(pdqg_v2_dbg.get("pdqg_v2_skipped_count", max(total_cnt - active_cnt, 0)))
                should_have_grad = bool(active_cnt > 0 and (loss_pdqg_isfinite is not False))
                logger.info(
                    "[PDQG-V2] active_count=%d total_count=%d skipped_count=%d all_inactive_reason=%s "
                    "scoring_head_should_have_grad=%s scoring_head_with_grad=%d loss_pdqg_ambig_rank=%s isfinite=%s "
                    "region_feat_isfinite=%s disamb_summary_isfinite=%s skip_reason_v2=%s",
                    active_cnt,
                    total_cnt,
                    skipped_cnt,
                    pdqg_v2_dbg.get("pdqg_v2_all_inactive_reason", None),
                    should_have_grad,
                    scoring_with_grad,
                    "None" if loss_pdqg_val is None else f"{loss_pdqg_val:.6e}",
                    loss_pdqg_isfinite,
                    pdqg_v2_dbg.get("region_feat_isfinite", None),
                    pdqg_v2_dbg.get("disamb_summary_isfinite", None),
                    pdqg_v2_dbg.get("skip_reason_v2", None),
                )
                self._pdqg_grad_probe_logged = True
            pdqg_v2_dbg = {}
            if hasattr(model_obj, "_last_pdqg_v2_debug") and isinstance(model_obj._last_pdqg_v2_debug, dict):
                pdqg_v2_dbg = model_obj._last_pdqg_v2_debug
            active_cnt = int(pdqg_v2_dbg.get("pdqg_v2_active_count", 0))
            if active_cnt > 0 and scoring_with_grad == 0:
                raise ValueError(
                    "[PDQG-V2] Ambiguity active samples exist but scoring head got no gradient on iter=0. "
                    f"pdqg_v2_debug={pdqg_v2_dbg}"
                )
            if pdqg_with_grad == 0:
                pdqg_dbg = {}
                if hasattr(model_obj, "_last_pdqg_debug") and isinstance(model_obj._last_pdqg_debug, dict):
                    pdqg_dbg = model_obj._last_pdqg_debug
                raise ValueError(
                    "[PDQG] No gradients reached PDQG parameters on iter=0. "
                    "Check query injection graph / no-op conditions. "
                    f"pdqg_debug={pdqg_dbg}"
                )

        self._write_metrics(loss_dict, data_time)
        # STA adapter-only：确保 GradScaler 能正常运行
        # 即使 gate 接近 0 导致梯度数值极小，也不应崩溃
        _sta_active = bool(
            getattr(model_obj, "sta_on", False)
            and getattr(model_obj, "sta_train_on", False)
        )
        if _sta_active:
            # 手动注册一次 inf check，避免 GradScaler 因无参数有梯度而崩溃
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


class _SlotAnomalySimpleTrainer(SimpleTrainer):
    """SimpleTrainer with optional anomaly detection around forward and backward."""

    def _slot_anomaly_enabled(self) -> bool:
        return bool(getattr(self, "slot_detect_anomaly", False))

    def _log_slot_anomaly_once(self):
        if not self._slot_anomaly_enabled():
            return
        if bool(getattr(self, "_slot_anomaly_logged", False)):
            return
        logging.getLogger("detectron2").info("[SLOT-ANOMALY] detect_anomaly=True (forward+backward)")
        self._slot_anomaly_logged = True

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        self._log_slot_anomaly_once()
        with torch.autograd.set_detect_anomaly(self._slot_anomaly_enabled()):
            if self.zero_grad_before_forward:
                self.optimizer.zero_grad()
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())
            if not self.zero_grad_before_forward:
                self.optimizer.zero_grad()
            losses.backward()

        _log_slot_mech_grad_debug_once(self, self.model)
        self.after_backward()
        if self.async_write_metrics:
            self.concurrent_executor.submit(self._write_metrics, loss_dict, data_time, iter=self.iter)
        else:
            self._write_metrics(loss_dict, data_time)
        self.optimizer.step()


class _SlotAnomalyAMPTrainer(AMPTrainer):
    """AMPTrainer with optional anomaly detection around forward and scaled backward."""

    def _slot_anomaly_enabled(self) -> bool:
        return bool(getattr(self, "slot_detect_anomaly", False))

    def _log_slot_anomaly_once(self):
        if not self._slot_anomaly_enabled():
            return
        if bool(getattr(self, "_slot_anomaly_logged", False)):
            return
        logging.getLogger("detectron2").info("[SLOT-ANOMALY] detect_anomaly=True (forward+backward)")
        self._slot_anomaly_logged = True

    def run_step(self):
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        self._log_slot_anomaly_once()
        with torch.autograd.set_detect_anomaly(self._slot_anomaly_enabled()):
            if self.zero_grad_before_forward:
                self.optimizer.zero_grad()
            with autocast(dtype=self.precision):
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

            if not self.zero_grad_before_forward:
                self.optimizer.zero_grad()
            self.grad_scaler.scale(losses).backward()

        _log_slot_mech_grad_debug_once(self, self.model)
        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()
        if self.async_write_metrics:
            self.concurrent_executor.submit(self._write_metrics, loss_dict, data_time, iter=self.iter)
        else:
            self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        logger.info("Model on device:\n{}".format(model.device))
        model.print_trainable_parameters()
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        lr_scheduler = self.build_lr_scheduler(cfg, optimizer)

        model = create_ddp_model(model, broadcast_buffers=False)
        use_mdf_amp_debug = bool(
            cfg.SOLVER.AMP.ENABLED
            and (
                (
                    getattr(cfg.MODEL.OpenWorldSAM2, "MULTISCALE_DEEP_FUSION_ON", False)
                    and getattr(cfg.MODEL.OpenWorldSAM2, "MULTISCALE_DEEP_FUSION_TRAIN_ON", False)
                )
                or (
                    getattr(cfg.MODEL.OpenWorldSAM2, "EFTPG_ON", False)
                    and getattr(cfg.MODEL.OpenWorldSAM2, "EFTPG_TRAIN_ON", False)
                )
                or (
                    getattr(cfg.MODEL.OpenWorldSAM2, "PDQG_ON", False)
                    and getattr(cfg.MODEL.OpenWorldSAM2, "PDQG_TRAIN_ON", False)
                )
                or (
                    getattr(cfg.MODEL.OpenWorldSAM2, "STA_ON", False)
                    and getattr(cfg.MODEL.OpenWorldSAM2, "STA_TRAIN_ON", False)
                )
            )
        )
        slot_detect_anomaly = bool(getattr(cfg.SOLVER, "SLOT_DETECT_ANOMALY", False))
        slot_mech_grad_debug = bool(getattr(cfg.SOLVER, "SLOT_MECH_GRAD_DEBUG", False))
        if use_mdf_amp_debug:
            trainer_cls = _MDFAMPTrainer
        elif cfg.SOLVER.AMP.ENABLED:
            trainer_cls = _SlotAnomalyAMPTrainer if (slot_detect_anomaly or slot_mech_grad_debug) else AMPTrainer
        else:
            trainer_cls = _SlotAnomalySimpleTrainer if (slot_detect_anomaly or slot_mech_grad_debug) else SimpleTrainer
        if use_mdf_amp_debug:
            logger.info("[AMP-DEBUG] Using _MDFAMPTrainer with pre-step gradient diagnostics.")
        if slot_detect_anomaly:
            logger.info("[SLOT-ANOMALY] trainer_wrapper=%s", trainer_cls.__name__)
        if slot_mech_grad_debug:
            logger.info("[SLOT-MECH-GRAD] trainer_wrapper=%s one_iter=True", trainer_cls.__name__)
        self._trainer = trainer_cls(model, data_loader, optimizer)
        self._trainer.slot_detect_anomaly = slot_detect_anomaly
        self._trainer.slot_mech_grad_debug = slot_mech_grad_debug

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        # kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model)) TODO: release ema training for large models
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self._bridge_diag_cfg = cfg.MODEL.OpenWorldSAM2.DIAG
        self._best_metric = None
        self._best_metric_key = None
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        ret = super().resume_or_load(resume=resume)
        model = _unwrap_model(self._trainer.model)
        if hasattr(model, "apply_residual_beta_override"):
            model.apply_residual_beta_override()
        return ret

    def run_step(self):
        try:
            self._trainer.run_step()
        except torch.cuda.OutOfMemoryError:
            self._handle_oom_skip()
            return
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                self._handle_oom_skip()
                return
            raise

    def _handle_oom_skip(self):
        logger = logging.getLogger("detectron2")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            logger.warning(
                "OOM skip iter=%d mem_alloc=%.1fMB mem_reserved=%.1fMB",
                self.iter,
                mem_alloc,
                mem_reserved,
            )
        else:
            logger.warning("OOM skip iter=%d (CUDA not available)", self.iter)

        if hasattr(self._trainer, "optimizer"):
            self._trainer.optimizer.zero_grad(set_to_none=True)
        get_event_storage().put_scalar("oom_skipped", 1.0)

    def _find_metric(self, results, target_key, prefix=""):
        if isinstance(results, dict):
            if target_key in results:
                return results[target_key], (prefix + target_key).lstrip(".")
            for key, value in results.items():
                found_value, found_key = self._find_metric(value, target_key, prefix + f"{key}.")
                if found_value is not None:
                    return found_value, found_key
        return None, None

    def _select_eval_metric(self, results):
        for key in ["cIoU", "mIoU", "precision@0.5"]:
            value, full_key = self._find_metric(results, key)
            if value is not None:
                return key, full_key, float(value)
        return None, None, None

    def _save_best_if_improved(self, results):
        logger = logging.getLogger("detectron2")
        _, metric_full_key, metric_value = self._select_eval_metric(results)
        if metric_value is None:
            logger.info("BestCheck: no supported metric found in eval results.")
            return

        if self._best_metric is None:
            is_best = True
        else:
            is_best = metric_value > self._best_metric

        prev_best = self._best_metric
        if is_best:
            self._best_metric = metric_value
            self._best_metric_key = metric_full_key
            if comm.is_main_process():
                self.checkpointer.save("model_best")

        logger.info(
            "BestCheck: key=%s current=%.4f best=%s updated=%s",
            metric_full_key,
            metric_value,
            "None" if prev_best is None else f"{prev_best:.4f}",
            str(is_best),
        )

    def _bridge_diag_enabled(self) -> bool:
        return bool(getattr(self._bridge_diag_cfg, "ENABLED", False))

    def _bridge_diag_model(self):
        return _unwrap_model(self.model)

    def _bridge_diag_collect_focus_stats(self) -> Dict[str, Dict[str, Any]]:
        model = self._bridge_diag_model()
        named_params = dict(model.named_parameters())
        all_names = [name for name, param in named_params.items() if param.requires_grad]
        sta_names = [
            name for name in all_names
            if (
                name.startswith("sta_adapters.")
                or name.startswith("dp_plugin.")
                or name.startswith("dense_prompt_generator.")
            )
        ]
        prompt_encoder_names = [
            name for name in all_names
            if name.startswith("evf_sam2.visual_model.sam_prompt_encoder.")
            and "pe_layer." not in name
        ]
        mask_decoder_names = [
            name for name in all_names if _param_name_matches_output_head(name)
        ]
        lora_names = [name for name in all_names if ("lora_A" in name or "lora_B" in name)]
        stats = {
            "sta": _collect_param_rows(named_params, sta_names),
            "prompt_encoder": _collect_param_rows(named_params, prompt_encoder_names),
            "mask_decoder_output": _collect_param_rows(named_params, mask_decoder_names),
            "lora_all": _collect_param_rows(named_params, lora_names),
        }
        lora_by_layer: Dict[str, Dict[str, Any]] = OrderedDict()
        for name in lora_names:
            layer_idx = _extract_lora_layer_index(name)
            key = f"layer_{layer_idx}" if layer_idx is not None else "layer_unknown"
            lora_by_layer.setdefault(key, []).append(name)
        stats["lora_by_layer"] = {
            key: _collect_param_rows(named_params, names)
            for key, names in lora_by_layer.items()
        }
        return stats

    def _bridge_diag_collect_sta_stats(self) -> List[Dict[str, Any]]:
        model = self._bridge_diag_model()
        rows = []
        adapters = getattr(model, "sta_adapters", {})
        if not hasattr(adapters, "items"):
            return rows
        for block_name, adapter in adapters.items():
            gate_val = adapter.gate.detach().float().tanh()
            rec = {
                "block": block_name,
                "gate_mean": float(gate_val.mean().item()) if gate_val.numel() > 0 else None,
                "gate_min": float(gate_val.min().item()) if gate_val.numel() > 0 else None,
                "gate_max": float(gate_val.max().item()) if gate_val.numel() > 0 else None,
            }
            last = getattr(adapter, "_bridge_diag_last", None)
            if isinstance(last, dict):
                rec["text_proj_norm_mean"] = last.get("text_proj_norm_mean")
                rec["text_proj_norm_std"] = last.get("text_proj_norm_std")
                rec["text_proj_norm_max"] = last.get("text_proj_norm_max")
            rows.append(rec)
        return rows

    def _bridge_diag_log_eval(self, when: str):
        if not self._bridge_diag_enabled() or not comm.is_main_process():
            return
        logger = logging.getLogger("detectron2")
        focus_stats = self._bridge_diag_collect_focus_stats()
        logger.info("[BRIDGE-DIAG][EVAL] when=%s begin", when)
        sta_rows = self._bridge_diag_collect_sta_stats()
        if sta_rows:
            for row in sta_rows:
                logger.info(
                    "[BRIDGE-DIAG][EVAL] when=%s sta_block=%s gate(mean=%s,min=%s,max=%s) text_proj_norm(mean=%s,std=%s,max=%s)",
                    when,
                    row["block"],
                    _format_stat(row.get("gate_mean")),
                    _format_stat(row.get("gate_min")),
                    _format_stat(row.get("gate_max")),
                    _format_stat(row.get("text_proj_norm_mean")),
                    _format_stat(row.get("text_proj_norm_std")),
                    _format_stat(row.get("text_proj_norm_max")),
                )
        else:
            logger.info("[BRIDGE-DIAG][EVAL] when=%s sta_stats=skipped", when)
        for key, label in (
            ("prompt_encoder", "prompt_encoder"),
            ("mask_decoder_output", "mask_decoder_output"),
            ("lora_all", "lora_all"),
        ):
            row = focus_stats.get(key, {})
            if not row or row.get("param_count", 0) <= 0:
                logger.info("[BRIDGE-DIAG][EVAL] when=%s %s=skipped", when, label)
                continue
            logger.info(
                "[BRIDGE-DIAG][EVAL] when=%s %s grad_norm_mean=%s grad_norm_max=%s param_count=%d",
                when,
                label,
                _format_stat(row.get("grad_norm_mean")),
                _format_stat(row.get("grad_norm_max")),
                int(row.get("param_count", 0)),
            )
        lora_by_layer = focus_stats.get("lora_by_layer", {})
        if lora_by_layer:
            for layer_name, row in lora_by_layer.items():
                logger.info(
                    "[BRIDGE-DIAG][EVAL] when=%s lora=%s grad_norm_mean=%s grad_norm_max=%s param_count=%d",
                    when,
                    layer_name,
                    _format_stat(row.get("grad_norm_mean")),
                    _format_stat(row.get("grad_norm_max")),
                    int(row.get("param_count", 0)),
                )
        else:
            logger.info("[BRIDGE-DIAG][EVAL] when=%s lora_by_layer=skipped", when)

    def _bridge_diag_log_samples_and_summary(self):
        if not self._bridge_diag_enabled() or not comm.is_main_process():
            return
        logger = logging.getLogger("detectron2")
        model = self._bridge_diag_model()
        cache = getattr(model, "_bridge_diag_cache", None)
        if not isinstance(cache, dict):
            logger.info("[BRIDGE-DIAG][SUMMARY] cache=skipped")
            return
        samples = list(cache.get("samples", []))
        if not samples:
            logger.info("[BRIDGE-DIAG][SUMMARY] eval_samples=0 skipped")
            return
        for sample in samples:
            logger.info(
                "[BRIDGE-DIAG][SAMPLE] sample_uid=%s image_id=%s prompt_idx=%s K=%s chosen_query_idx=%s chosen_mask_idx=%s chosen_mask_score=%s iou_pred_sorted=%s final_score_sorted=%s route=%s evidence=%s weak_map_fallback=%s",
                sample.get("sample_uid"),
                sample.get("image_id"),
                sample.get("prompt_index"),
                sample.get("candidate_count"),
                sample.get("chosen_query_index"),
                sample.get("chosen_mask_index"),
                _format_stat(sample.get("chosen_mask_score")),
                sample.get("iou_pred_sorted"),
                sample.get("final_score_sorted", "unavailable"),
                sample.get("route_scores", "unavailable"),
                sample.get("evidence_scores", "unavailable"),
                sample.get("weak_map_fallback"),
            )
        summary_parts: List[str] = []
        focus_stats = self._bridge_diag_collect_focus_stats()
        sta_grad = focus_stats.get("sta", {}).get("grad_norm_mean")
        lora_grad = focus_stats.get("lora_all", {}).get("grad_norm_mean")
        prompt_grad = focus_stats.get("prompt_encoder", {}).get("grad_norm_mean")
        mask_grad = focus_stats.get("mask_decoder_output", {}).get("grad_norm_mean")
        if sta_grad is not None and mask_grad is not None and sta_grad > 0 and mask_grad > 5.0 * sta_grad:
            summary_parts.append("mask-decoder gradients are much larger than STA gradients, which may imply decoder-side updates dominate")
        if sta_grad is not None and lora_grad is not None and sta_grad > 0 and lora_grad > 5.0 * sta_grad:
            summary_parts.append("LoRA gradients are much larger than STA gradients, which suggests adapter updates may be underpowered")
        sta_rows = self._bridge_diag_collect_sta_stats()
        if sta_rows:
            gate_abs_mean = float(np.mean([abs(row.get("gate_mean", 0.0) or 0.0) for row in sta_rows]))
            if gate_abs_mean < 1e-3:
                summary_parts.append("STA gates remain very close to zero across the latest eval pass, which may imply weak adapter routing")
        fallback_vals = [1.0 if sample.get("weak_map_fallback") else 0.0 for sample in samples if sample.get("weak_map_fallback") is not None]
        if fallback_vals and float(np.mean(fallback_vals)) > 0.5:
            summary_parts.append("weak-map fallback triggered frequently in sampled eval cases, which suggests map evidence is often too weak to rerank confidently")
        top_gaps = [sample.get("top1_top2_gap") for sample in samples if sample.get("top1_top2_gap") is not None]
        if top_gaps and float(np.mean(top_gaps)) < 0.05:
            summary_parts.append("candidate iou_pred separation is low in sampled eval cases, which may imply weak inter-candidate ranking confidence")
        if not summary_parts:
            summary_parts.append("current diagnostics do not show a single dominant bottleneck; compare multiple evals for stable patterns")
        for item in summary_parts:
            logger.info("[BRIDGE-DIAG][SUMMARY] %s", item)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = min(cfg.DATALOADER.NUM_WORKERS, 4)

        hooks_list = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
        ]
        if int(getattr(cfg.SOLVER, "SLOT_ONLY_WARMUP_ITERS", 0)) > 0:
            hooks_list.append(_SlotOnlyWarmupHook(int(cfg.SOLVER.SLOT_ONLY_WARMUP_ITERS)))
        if self._bridge_diag_enabled():
            hooks_list.append(_BridgeDiagTrainHook())

        def _eval_and_save():
            if self._bridge_diag_enabled():
                self._bridge_diag_log_eval("before")
                model = self._bridge_diag_model()
                if hasattr(model, "_reset_bridge_diag_cache"):
                    model._reset_bridge_diag_cache(reset_samples=True)
            results = self.test(cfg, self.model)
            if self._bridge_diag_enabled():
                self._bridge_diag_log_eval("after")
                self._bridge_diag_log_samples_and_summary()
            self._save_best_if_improved(results)
            return results

        if cfg.TEST.EVAL_PERIOD > 0:
            hooks_list.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, _eval_and_save))

        hooks_list.append(_CudaEmptyCacheHook(period=200))
        hooks_list.append(_RollingCheckpointerHook(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
        hooks_list.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        hooks_list.append(_FinalCheckpointerHook(self.checkpointer))
        return hooks_list

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        print("calling build_evaluator")
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            print("output_folder:", output_folder)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        print("evaluator_type:", evaluator_type)
        # semantic segmentation
        if evaluator_type in ["sem_seg", ]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "scannet_panoptic_seg"
        ]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))

        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))

        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))

        # RefCOCO
        if evaluator_type in ["grounding_refcoco"]:
            evaluator_list.append(GroundingEvaluator(dataset_name))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = None
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if dataset_name in ["coco_2017_val", "ade20k_instance_val"]:
            mapper = OpenWorldSAM2InstanceDatasetMapper(cfg, is_train=False)
        elif dataset_name in ["coco_2017_val_panoptic_with_sem_seg", "ade20k_panoptic_val"]:
            mapper = OpenWorldSAM2PanopticDatasetMapper(cfg, is_train=False)
        elif dataset_name in ["scannet_21_panoptic_val"]:
            mapper = ScanNetPanoDatasetMapper(cfg, is_train=False)
        elif dataset_name in ["ade20k_full_sem_seg_val", "pascal_context_459_sem_seg_val",
                              "pascal_context_59_sem_seg_val", "pascalvoc20_sem_seg_val",
                              "sunrgbd_37_val_seg", "scannet_21_val_seg", "scannet_41_val_seg"]:
            mapper = OpenWorldSAM2SemanticDatasetMapper(cfg, is_train=False)
        elif evaluator_type in ["grounding_refcoco"] or dataset_name in ["refcocog_val_umd", "refcoco_val_unc", "refcocog_val", "refcoco+_val"]:
            mapper = RefCOCODatasetMapper(cfg, is_train=False)
        if mapper is None:
            return build_detection_test_loader(cfg, dataset_name=dataset_name)
        return build_detection_test_loader(cfg, dataset_name=dataset_name, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Modify train loader to use a fixed subset of dataset.
        """
        logger = logging.getLogger("detectron2")
        logger.info("IMS_PER_BATCH before build_train_loader: %s", cfg.SOLVER.IMS_PER_BATCH)
        # Choose the appropriate dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "open_world_instance":
            mapper = OpenWorldSAM2InstanceDatasetMapper(cfg, is_train=True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "open_world_instance_all":
            mapper = OpenWorldSAM2InstanceDatasetMapperAll(cfg, is_train=True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "open_world_panoptic":
            mapper = OpenWorldSAM2PanopticDatasetMapper(cfg, is_train=True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "open_world_panoptic_all":
            mapper = OpenWorldSAM2PanopticDatasetMapperAll(cfg, is_train=True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "refcoco":
            mapper = RefCOCODatasetMapper(cfg, is_train=True)
        else:
            mapper = None
        subset_on = bool(getattr(cfg, "TRAIN_SUBSET_FILTER_ON", False))
        key_file = str(getattr(cfg, "TRAIN_SUBSET_FILTER_KEY_FILE", "") or "").strip()
        key_fields = str(getattr(cfg, "TRAIN_SUBSET_FILTER_KEY_FIELDS", "image_id,prompt_index"))
        if (not subset_on) or (key_file == ""):
            return build_detection_train_loader(cfg, mapper=mapper)

        key_map = cls._load_train_subset_key_map(key_file)
        dataset_names = list(cfg.DATASETS.TRAIN)
        dataset_dicts = []
        for name in dataset_names:
            dataset_dicts.extend(DatasetCatalog.get(name))
        filtered_dicts, stats = cls._filter_train_dataset_dicts_by_keys(
            dataset_dicts=dataset_dicts,
            key_map=key_map,
            key_fields=key_fields,
        )
        logger.info(
            "[TRAIN-SUBSET] on=True key_file=%s key_fields=%s total=%d kept=%d kept_ratio=%.4f kept_prompts=%d",
            key_file,
            key_fields,
            stats["total_images"],
            stats["kept_images"],
            stats["kept_images"] / max(stats["total_images"], 1),
            stats["kept_prompts"],
        )
        if len(filtered_dicts) == 0:
            raise ValueError(
                f"[TRAIN-SUBSET] No samples kept after filtering with key file: {key_file}"
            )
        return build_detection_train_loader(cfg, mapper=mapper, dataset=filtered_dicts)

    @staticmethod
    def _load_train_subset_key_map(key_file: str) -> Dict[int, Set[int]]:
        key_map: Dict[int, Set[int]] = {}
        with open(key_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if ":" not in s:
                    continue
                lhs, rhs = s.split(":", 1)
                try:
                    image_id = int(lhs.strip())
                    prompt_idx = int(rhs.strip())
                except Exception:
                    continue
                key_map.setdefault(image_id, set()).add(prompt_idx)
        if len(key_map) == 0:
            raise ValueError(f"[TRAIN-SUBSET] key file has no valid keys: {key_file}")
        return key_map

    @staticmethod
    def _filter_train_dataset_dicts_by_keys(
        dataset_dicts: List[Dict[str, Any]],
        key_map: Dict[int, Set[int]],
        key_fields: str = "image_id,prompt_index",
    ):
        # key_fields currently informational; effective matching key is image_id:prompt_index.
        kept = []
        kept_prompts = 0
        for d in dataset_dicts:
            image_id = d.get("image_id", None)
            if image_id is None:
                continue
            try:
                image_id_int = int(image_id)
            except Exception:
                continue
            allow = key_map.get(image_id_int, None)
            if not allow:
                continue
            d_new = copy.deepcopy(d)
            grounding = d_new.get("grounding_info", None)
            if isinstance(grounding, list):
                selected = [grounding[idx] for idx in sorted(allow) if 0 <= idx < len(grounding)]
                if len(selected) == 0:
                    continue
                d_new["grounding_info"] = selected
                kept_prompts += len(selected)
            kept.append(d_new)
        stats = {
            "total_images": int(len(dataset_dicts)),
            "kept_images": int(len(kept)),
            "kept_prompts": int(kept_prompts),
            "key_fields": key_fields,
        }
        return kept, stats


    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED
        lora_lr = float(getattr(cfg.SOLVER, "LORA_LR", 1e-5))
        slot_lr = float(getattr(cfg.SOLVER, "SLOT_LR", 1e-4))
        slot_only_warmup_iters = int(getattr(cfg.SOLVER, "SLOT_ONLY_WARMUP_ITERS", 0))
        logger = logging.getLogger("detectron2")
        compat_on = bool(getattr(cfg.MODEL.OpenWorldSAM2, "OPENWORLDSAM_COMPAT", False))
        lora_on = bool(getattr(cfg.MODEL.OpenWorldSAM2, "LORA_ON", True)) and (not compat_on)
        cand_score_head_train_only = bool(
            getattr(cfg.MODEL.OpenWorldSAM2, "CAND_SCORE_HEAD_TRAIN_ONLY", False)
        )
        allow_empty_lora_group = bool(cand_score_head_train_only)
        beit3_train_last_n_layers = 0 if compat_on else int(getattr(cfg.MODEL.OpenWorldSAM2, "BEIT3_TRAIN_LAST_N_LAYERS", 0))

        def _extract_beit3_layer_index(param_name: str):
            prefixes = (
                "evf_sam2.mm_extractor.beit3.encoder.layers.",
                "evf_sam2.mm_extractor.beit3.layers.",
                "mm_extractor.beit3.encoder.layers.",
                "mm_extractor.beit3.layers.",
            )
            for prefix in prefixes:
                if not param_name.startswith(prefix):
                    continue
                suffix = param_name[len(prefix):]
                layer_idx_str = suffix.split(".", 1)[0]
                if layer_idx_str.isdigit():
                    return int(layer_idx_str)
            return None

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        effective_sta_on = bool(getattr(cfg.MODEL.OpenWorldSAM2, "STA_ON", False)) and (not compat_on)
        effective_sta_train_on = bool(getattr(cfg.MODEL.OpenWorldSAM2, "STA_TRAIN_ON", False)) and (not compat_on)
        print(f"[BUILD-OPT] STA_ON={cfg.MODEL.OpenWorldSAM2.STA_ON} STA_TRAIN_ON={cfg.MODEL.OpenWorldSAM2.STA_TRAIN_ON}", flush=True)
        sta_adapter_only_optimizer = bool(
            effective_sta_on
            and effective_sta_train_on
        )
        print(f"[STA-OPT-DEBUG] STA_ON={getattr(cfg.MODEL.OpenWorldSAM2, 'STA_ON', 'MISSING')} STA_TRAIN_ON={getattr(cfg.MODEL.OpenWorldSAM2, 'STA_TRAIN_ON', 'MISSING')} sta_adapter_only_optimizer={sta_adapter_only_optimizer}", flush=True)
        multiscale_adapter_only_optimizer = bool(
            getattr(cfg.MODEL.OpenWorldSAM2, "MULTISCALE_DEEP_FUSION_ON", False)
            and getattr(cfg.MODEL.OpenWorldSAM2, "MULTISCALE_DEEP_FUSION_TRAIN_ON", False)
            and (not compat_on)
        )
        eftpg_adapter_only_optimizer = bool(
            getattr(cfg.MODEL.OpenWorldSAM2, "EFTPG_ON", False)
            and getattr(cfg.MODEL.OpenWorldSAM2, "EFTPG_TRAIN_ON", False)
            and (not compat_on)
        )
        pdqg_adapter_only_optimizer = bool(
            getattr(cfg.MODEL.OpenWorldSAM2, "PDQG_ON", False)
            and getattr(cfg.MODEL.OpenWorldSAM2, "PDQG_TRAIN_ON", False)
            and (not compat_on)
        )
        adapter_only_optimizer = bool(
            getattr(cfg.MODEL.OpenWorldSAM2, "DEEP_FUSION_ADAPTER_ON", False)
            and getattr(cfg.MODEL.OpenWorldSAM2, "DEEP_FUSION_ADAPTER_TRAIN_ON", False)
            and (not compat_on)
        )
        disamb_route_only_optimizer = bool(
            getattr(cfg.MODEL.OpenWorldSAM2, "DISAMB_ROUTE_ON", False)
            and getattr(cfg.MODEL.OpenWorldSAM2, "DISAMB_ROUTE_TRAIN_ON", False)
            and (not compat_on)
        )
        allowed_param_names = None
        allowed_param_ids: Set[int] | None = None
        active_tag = None
        helper_names = None
        if compat_on:
            active_tag = "OWS-COMPAT"
            named_params = dict(model.named_parameters())
            helper_names = [
                name
                for name, param in named_params.items()
                if param.requires_grad and _ows_compat_is_baseline_param_name(name)
            ]
            if len(helper_names) == 0:
                raise ValueError("OPENWORLDSAM_COMPAT=True but no baseline trainable parameters were found.")
        # Priority: EFTPG > PDQG > STA > multiscale > disamb > deep_fusion
        if (not compat_on) and eftpg_adapter_only_optimizer:
            active_tag = "EFTPG"
            if not hasattr(model, "get_eftpg_trainable_param_names"):
                raise ValueError(
                    "EFTPG adapter-only optimizer requested, but "
                    "model.get_eftpg_trainable_param_names() is missing."
                )
            helper_names = model.get_eftpg_trainable_param_names()
        elif (not compat_on) and pdqg_adapter_only_optimizer:
            active_tag = "PDQG"
            if not hasattr(model, "get_pdqg_trainable_param_names"):
                raise ValueError(
                    "PDQG adapter-only optimizer requested, but "
                    "model.get_pdqg_trainable_param_names() is missing."
                )
            helper_names = model.get_pdqg_trainable_param_names()
        elif (not compat_on) and sta_adapter_only_optimizer:
            active_tag = "STA"
            if not hasattr(model, "get_sta_trainable_param_names"):
                raise ValueError(
                    "STA adapter-only optimizer requested, but "
                    "model.get_sta_trainable_param_names() is missing."
                )
            helper_names = model.get_sta_trainable_param_names()
        elif (not compat_on) and multiscale_adapter_only_optimizer:
            active_tag = "MULTISCALE-DEEP-FUSION"
            if not hasattr(model, "get_multiscale_deep_fusion_trainable_param_names"):
                raise ValueError(
                    "MULTISCALE_DEEP_FUSION adapter-only optimizer requested, but "
                    "model.get_multiscale_deep_fusion_trainable_param_names() is missing."
                )
            helper_names = model.get_multiscale_deep_fusion_trainable_param_names()
        elif (not compat_on) and disamb_route_only_optimizer:
            active_tag = "DISAMB-ROUTE"
            if not hasattr(model, "get_disamb_route_trainable_param_names"):
                raise ValueError(
                    "DISAMB_ROUTE expert-only optimizer requested, but "
                    "model.get_disamb_route_trainable_param_names() is missing."
                )
            helper_names = model.get_disamb_route_trainable_param_names()
        elif (not compat_on) and adapter_only_optimizer:
            active_tag = "DEEP-FUSION"
            if not hasattr(model, "get_deep_fusion_trainable_param_names"):
                raise ValueError(
                    "DEEP_FUSION_ADAPTER adapter-only optimizer requested, but "
                    "model.get_deep_fusion_trainable_param_names() is missing."
                )
            helper_names = model.get_deep_fusion_trainable_param_names()

        if active_tag is not None:
            if helper_names is None or len(helper_names) == 0:
                raise ValueError(
                    f"{active_tag} adapter-only optimizer requested, but "
                    "helper returned empty parameter name list."
                )
            allowed_param_names = set(helper_names)
            named_params = dict(model.named_parameters())
            matched_names = [
                n for n in allowed_param_names if n in named_params and named_params[n].requires_grad
            ]
            if len(matched_names) == 0:
                # 诊断：打印 helper 返回的名字 vs 模型里实际存在的名字
                all_model_names = set(named_params.keys())
                helper_set = set(helper_names)
                in_model_not_trainable = [n for n in helper_set if n in named_params and not named_params[n].requires_grad]
                not_in_model = [n for n in helper_set if n not in named_params]
                logger.info(
                    "[%s] DIAGNOSTIC: helper returned %d names, 0 matched. "
                    "in_model_but_not_trainable=%s not_in_model=%s",
                    active_tag,
                    len(helper_names),
                    in_model_not_trainable[:5],
                    not_in_model[:5],
                )
                raise ValueError(
                    f"{active_tag} adapter-only optimizer requested, but no "
                    "trainable parameters matched helper names. "
                    f"helper_names={sorted(helper_names)[:5]}"
                )
            allowed_param_ids = {id(named_params[n]) for n in matched_names}
            preview = sorted(matched_names)[:20]
            if active_tag == "OWS-COMPAT":
                logger.info(
                    "[OWS-COMPAT] optimizer baseline trainable_param_count=%d preview=%s",
                    len(matched_names),
                    preview,
                )
            elif active_tag == "EFTPG":
                logger.info(
                    "[EFTPG] adapter_only_optimizer=True trainable_param_count=%d preview=%s",
                    len(matched_names),
                    preview,
                )
            elif active_tag == "PDQG":
                logger.info(
                    "[PDQG] adapter_only_optimizer=True trainable_param_count=%d preview=%s",
                    len(matched_names),
                    preview,
                )
            elif active_tag == "STA":
                logger.info(
                    "[STA] adapter_only_optimizer=True trainable_param_count=%d preview=%s",
                    len(matched_names),
                    preview,
                )
            elif active_tag == "MULTISCALE-DEEP-FUSION":
                logger.info(
                    "[MDF] multiscale_adapter_only_optimizer=True trainable_param_count=%d preview=%s",
                    len(matched_names),
                    preview,
                )
            elif active_tag == "DISAMB-ROUTE":
                logger.info(
                    "[DISAMB-ROUTE] expert_only_optimizer=True trainable_param_count=%d preview=%s",
                    len(matched_names),
                    preview,
                )
            else:
                logger.info(
                    "[DEEP-FUSION] adapter_only_optimizer=True trainable_param_count=%d preview=%s",
                    len(matched_names),
                    preview,
                )
        else:
            logger.info("[DEEP-FUSION] adapter_only_optimizer=False")

        if compat_on:
            beit3_target_layer_indices = []
        elif hasattr(model, "mm_extractor"):
            beit3_target_layer_indices = list(
                getattr(model.mm_extractor, "_beit3_full_unfreeze_layer_indices", [])
            )
        else:
            beit3_target_layer_indices = []
        beit3_target_layer_index_set = set(beit3_target_layer_indices)
        beit3_trainable_names = []
        beit3_trainable_param_count = 0
        beit3_trainable_param_ids: Set[int] = set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            layer_idx = _extract_beit3_layer_index(name)
            if layer_idx is None:
                continue
            if beit3_target_layer_index_set and layer_idx not in beit3_target_layer_index_set:
                continue
            beit3_trainable_names.append(name)
            beit3_trainable_param_count += param.numel()
            beit3_trainable_param_ids.add(id(param))
        if allowed_param_ids is not None and beit3_trainable_param_ids:
            allowed_param_ids = set(allowed_param_ids)
            allowed_param_ids.update(beit3_trainable_param_ids)
        logger.info(
            "[BEIT3-FULL-UNFREEZE] BEIT3_TRAIN_LAST_N_LAYERS=%d unfrozen_layer_indices=%s "
            "trainable_beit3_param_count=%d example_param_names=%s",
            beit3_train_last_n_layers,
            beit3_target_layer_indices,
            beit3_trainable_param_count,
            beit3_trainable_names[:20],
        )

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        grouped_params: "OrderedDict[tuple[str, float, float], Dict[str, Any]]" = OrderedDict()
        assigned_param_to_group: Dict[int, str] = {}
        model_lora_param_names = [
            n for n, _ in model.named_parameters() if "lora_A" in n or "lora_B" in n
        ]
        model_lora_param_ids = {
            id(p) for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n
        }
        model_slot_param_names = [
            n for n, p in model.named_parameters() if p.requires_grad and _is_slot_param_name(n)
        ]
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if allowed_param_ids is not None and id(value) not in allowed_param_ids:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                group_tags = [active_tag.lower() if active_tag is not None else "default"]
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    group_tags.append("backbone")
                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                    group_tags.append("no_wd_pos")
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                    group_tags.append("norm")
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                    group_tags.append("embedding")
                full_param_name = f"{module_name}.{module_param_name}" if module_name else module_param_name
                beit3_layer_idx = _extract_beit3_layer_index(full_param_name)
                is_beit3_full_unfreeze_param = (
                    beit3_train_last_n_layers > 0
                    and beit3_layer_idx is not None
                    and beit3_layer_idx in beit3_target_layer_index_set
                )
                if _is_slot_param_name(full_param_name):
                    hyperparams["lr"] = slot_lr
                    group_tags = ["slot"]
                elif "lora_A" in full_param_name or "lora_B" in full_param_name:
                    hyperparams["lr"] = lora_lr
                    group_tags = ["lora"]
                elif is_beit3_full_unfreeze_param:
                    hyperparams["lr"] = lora_lr
                    group_tags = ["beit3_full_unfreeze"]
                group_name = "+".join(group_tags)
                if id(value) in assigned_param_to_group:
                    raise ValueError(
                        f"Trainable parameter '{full_param_name}' matched multiple optimizer groups: "
                        f"{assigned_param_to_group[id(value)]} vs {group_name}"
                    )
                assigned_param_to_group[id(value)] = group_name
                group_key = (
                    group_name,
                    float(hyperparams["lr"]),
                    float(hyperparams["weight_decay"]),
                )
                if group_key not in grouped_params:
                    grouped_params[group_key] = {
                        "params": [],
                        "lr": hyperparams["lr"],
                        "weight_decay": hyperparams["weight_decay"],
                        "group_name": group_name,
                        "param_names": [],
                    }
                grouped_params[group_key]["params"].append(value)
                grouped_params[group_key]["param_names"].append(full_param_name)

        params = list(grouped_params.values())
        lora_group_param_names = [
            name
            for group in params
            if group.get("group_name") == "lora"
            for name in group.get("param_names", [])
        ]
        if lora_on and model_lora_param_names and not lora_group_param_names and not allow_empty_lora_group:
            raise ValueError(
                "Found LoRA parameters in model, but LoRA optimizer group is empty. "
                f"model_lora_param_names={model_lora_param_names[:10]}"
            )
        if lora_on and model_lora_param_names and not lora_group_param_names and allow_empty_lora_group:
            logger.info(
                "[CAND-SCORE] CAND_SCORE_HEAD_TRAIN_ONLY=True; allow empty LoRA optimizer group "
                "because LoRA params are intentionally frozen."
            )
        included_param_ids = {id(p) for group in params for p in group["params"]}
        if len(included_param_ids) != sum(len(group["params"]) for group in params):
            raise ValueError("A trainable parameter was assigned to more than one optimizer group.")
        missing_lora_param_names = [
            n for n, p in model.named_parameters()
            if id(p) in model_lora_param_ids and id(p) not in included_param_ids
        ]
        if lora_on and missing_lora_param_names and not allow_empty_lora_group:
            raise ValueError(
                "Some LoRA parameters were not assigned to any optimizer group: "
                f"{missing_lora_param_names[:10]}"
            )
        if lora_on and missing_lora_param_names and allow_empty_lora_group:
            logger.info(
                "[CAND-SCORE] CAND_SCORE_HEAD_TRAIN_ONLY=True; allow frozen LoRA params outside optimizer. "
                "missing_lora_param_count=%d example=%s",
                len(missing_lora_param_names),
                missing_lora_param_names[:10],
            )
        cand_score_head_optimizer_param_names = [
            n for n, p in model.named_parameters()
            if n.startswith("cand_score_head.") and id(p) in included_param_ids
        ]
        cand_score_head_optimizer_param_count = sum(
            p.numel()
            for n, p in model.named_parameters()
            if n.startswith("cand_score_head.") and id(p) in included_param_ids
        )
        logger.info(
            "[CAND-SCORE] optimizer includes cand_score_head params: count=%d tensor_count=%d example_param_names=%s",
            cand_score_head_optimizer_param_count,
            len(cand_score_head_optimizer_param_names),
            cand_score_head_optimizer_param_names[:20],
        )
        if cand_score_head_train_only and cand_score_head_optimizer_param_count == 0:
            raise ValueError(
                "CAND_SCORE_HEAD_TRAIN_ONLY=True but optimizer includes no cand_score_head.* parameters."
            )
        beit3_optimizer_param_names = [
            n for n, p in model.named_parameters()
            if id(p) in included_param_ids
            and _extract_beit3_layer_index(n) is not None
            and (
                not beit3_target_layer_index_set
                or _extract_beit3_layer_index(n) in beit3_target_layer_index_set
            )
        ]
        beit3_optimizer_param_count = sum(
            p.numel()
            for n, p in model.named_parameters()
            if id(p) in included_param_ids
            and _extract_beit3_layer_index(n) is not None
            and (
                not beit3_target_layer_index_set
                or _extract_beit3_layer_index(n) in beit3_target_layer_index_set
            )
        )
        if beit3_train_last_n_layers > 0 and beit3_optimizer_param_count == 0:
            raise ValueError(
                "BEIT3_TRAIN_LAST_N_LAYERS>0 but no BEiT-3 parameters were added to optimizer. "
                f"unfrozen_layer_indices={beit3_target_layer_indices} "
                f"trainable_beit3_param_names={beit3_trainable_names[:20]}"
            )
        beit3_full_unfreeze_groups = [
            group for group in params if group.get("group_name") == "beit3_full_unfreeze"
        ]
        beit3_group_lr = beit3_full_unfreeze_groups[0]["lr"] if beit3_full_unfreeze_groups else None
        beit3_group_weight_decay = (
            beit3_full_unfreeze_groups[0]["weight_decay"] if beit3_full_unfreeze_groups else None
        )
        logger.info(
            "[BEIT3-OPT] optimizer_beit3_param_count=%d example_param_names=%s group_name=%s lr=%s weight_decay=%s",
            beit3_optimizer_param_count,
            beit3_optimizer_param_names[:20],
            "beit3_full_unfreeze" if beit3_full_unfreeze_groups else None,
            beit3_group_lr,
            beit3_group_weight_decay,
        )

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)

        final_group_param_ids: Dict[int, str] = {}
        final_lora_group_count = 0
        for idx, group in enumerate(optimizer.param_groups):
            group_name = group.get("group_name", f"group_{idx}")
            param_names = list(group.get("param_names", []))
            if group_name == "lora":
                final_lora_group_count += 1
            for p in group.get("params", []):
                if id(p) in final_group_param_ids:
                    raise ValueError(
                        f"Optimizer param group duplication detected for group "
                        f"{group_name}: already seen in {final_group_param_ids[id(p)]}"
                    )
                final_group_param_ids[id(p)] = group_name
            logger.info(
                "[OPT-GROUP] group_name=%s lr=%.8g weight_decay=%.8g param_count=%d example_param_names=%s",
                group_name,
                group["lr"],
                group["weight_decay"],
                sum(p.numel() for p in group.get("params", [])),
                param_names[:10],
            )
            if bool(getattr(cfg.MODEL.OpenWorldSAM2.DIAG, "ENABLED", False)):
                logger.info(
                    "[BRIDGE-DIAG][OPT] group_name=%s lr=%.8g weight_decay=%.8g param_count=%d example_param_names=%s",
                    group_name,
                    group["lr"],
                    group["weight_decay"],
                    sum(p.numel() for p in group.get("params", [])),
                    param_names[:10],
                )

        if lora_on and model_lora_param_names and final_lora_group_count == 0 and not allow_empty_lora_group:
            raise ValueError(
                "Found LoRA parameters in model, but optimizer.param_groups has no 'lora' group."
            )
        if lora_on and model_lora_param_names and final_lora_group_count == 0 and allow_empty_lora_group:
            logger.info(
                "[CAND-SCORE] CAND_SCORE_HEAD_TRAIN_ONLY=True; optimizer.param_groups has no 'lora' group "
                "because LoRA params are intentionally frozen."
            )
        if model_lora_param_names:
            logger.info(
                "[LoRA-OPT] LORA_ON=%s lora_lr=%.8g lora_group_count=%d lora_param_count=%d example_lora_param_names=%s",
                lora_on,
                lora_lr,
                final_lora_group_count,
                len(model_lora_param_names),
                model_lora_param_names[:10],
            )
        else:
            logger.info(
                "[LoRA-OPT] LORA_ON=%s lora_lr=%.8g lora_group_count=%d lora_param_count=0 no LoRA trainable params",
                lora_on,
                lora_lr,
                final_lora_group_count,
            )
        slot_group_param_names = [
            name
            for group in optimizer.param_groups
            if group.get("group_name") == "slot"
            for name in group.get("param_names", [])
        ]
        slot_group_param_count = sum(
            p.numel()
            for group in optimizer.param_groups
            if group.get("group_name") == "slot"
            for p in group.get("params", [])
        )
        logger.info(
            "[SLOT-OPT] base_lr=%.8g lora_lr=%.8g slot_lr=%.8g slot_group_param_count=%d "
            "slot_tensor_count=%d slot_only_warmup_iters=%d enabled=%s match_keys=%s example_slot_param_names=%s",
            float(cfg.SOLVER.BASE_LR),
            lora_lr,
            slot_lr,
            int(slot_group_param_count),
            len(slot_group_param_names),
            slot_only_warmup_iters,
            bool(slot_only_warmup_iters > 0),
            SLOT_PARAM_MATCH_KEYS,
            slot_group_param_names[:10],
        )
        if model_slot_param_names and not slot_group_param_names:
            logger.warning(
                "[SLOT-OPT] Found trainable slot params but no slot optimizer group. example=%s",
                model_slot_param_names[:10],
            )
        if compat_on:
            compat_param_count = sum(
                p.numel()
                for group in optimizer.param_groups
                for p in group.get("params", [])
            )
            logger.info("[OWS-COMPAT] OPENWORLDSAM_COMPAT=True")
            logger.info("[OWS-COMPAT] BEIT3_TRAIN_LAST_N_LAYERS effective=%d", beit3_train_last_n_layers)
            logger.info("[OWS-COMPAT] trainable_param_count=%d", compat_param_count)
            for idx, group in enumerate(optimizer.param_groups):
                logger.info(
                    "[OWS-COMPAT] group_index=%d group_name=%s lr=%.8g weight_decay=%.8g param_count=%d example_param_names=%s",
                    idx,
                    group.get("group_name", f"group_{idx}"),
                    float(group.get("lr", 0.0)),
                    float(group.get("weight_decay", 0.0)),
                    sum(p.numel() for p in group.get("params", [])),
                    list(group.get("param_names", []))[:10],
                )
        return optimizer


class _SlotOnlyWarmupHook(hooks.HookBase):
    """Temporarily train only slot-specific fusion params during early iterations."""

    def __init__(self, warmup_iters: int):
        self.warmup_iters = max(int(warmup_iters), 0)
        self._original_requires_grad: Dict[str, bool] = {}
        self._active = False
        self._done = False

    def _optimizer_zero_grad(self):
        trainer_impl = getattr(self.trainer, "_trainer", None)
        optimizer = getattr(trainer_impl, "optimizer", None)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

    def _enter(self):
        if self._active or self._done or self.warmup_iters <= 0:
            return
        model = _unwrap_model(self.trainer.model)
        named_params = list(model.named_parameters())
        if not self._original_requires_grad:
            self._original_requires_grad = {name: bool(param.requires_grad) for name, param in named_params}
        slot_trainable = [
            name for name, param in named_params
            if self._original_requires_grad.get(name, False) and _is_slot_param_name(name)
        ]
        if len(slot_trainable) == 0:
            logging.getLogger("detectron2").warning(
                "[SLOT-WARMUP] requested warmup_iters=%d but found no originally-trainable slot params; skipping.",
                self.warmup_iters,
            )
            self._done = True
            return
        frozen_count = 0
        for name, param in named_params:
            keep_trainable = self._original_requires_grad.get(name, False) and _is_slot_param_name(name)
            if bool(param.requires_grad) and not keep_trainable:
                frozen_count += 1
            param.requires_grad_(keep_trainable)
        self._optimizer_zero_grad()
        self._active = True
        logging.getLogger("detectron2").info(
            "[SLOT-WARMUP] entering slot-only warmup at iter=%d until iter<%d. "
            "slot_tensor_count=%d frozen_tensor_count=%d match_keys=%s example_slot_params=%s",
            int(self.trainer.iter),
            self.warmup_iters,
            len(slot_trainable),
            frozen_count,
            SLOT_PARAM_MATCH_KEYS,
            slot_trainable[:10],
        )

    def _exit(self):
        if not self._active:
            return
        model = _unwrap_model(self.trainer.model)
        for name, param in model.named_parameters():
            if name in self._original_requires_grad:
                param.requires_grad_(self._original_requires_grad[name])
        self._optimizer_zero_grad()
        self._active = False
        self._done = True
        logging.getLogger("detectron2").info(
            "[SLOT-WARMUP] exiting slot-only warmup at iter=%d; restored original requires_grad state.",
            int(self.trainer.iter),
        )

    def before_train(self):
        if self.warmup_iters > 0 and int(self.trainer.iter) < self.warmup_iters:
            self._enter()
        else:
            self._done = True

    def before_step(self):
        if self.warmup_iters <= 0:
            return
        if int(self.trainer.iter) < self.warmup_iters:
            self._enter()
        elif self._active:
            self._exit()

    def after_step(self):
        if self._active and (int(self.trainer.iter) + 1) >= self.warmup_iters:
            self._exit()

    def after_train(self):
        self._exit()


class _BridgeDiagTrainHook(hooks.HookBase):
    def __init__(self):
        self._small_grad_streak: Dict[str, int] = {}

    def after_step(self):
        trainer = self.trainer
        if not hasattr(trainer, "_bridge_diag_enabled") or not trainer._bridge_diag_enabled():
            return
        if not comm.is_main_process():
            return
        period = int(getattr(trainer._bridge_diag_cfg, "TRAIN_LOG_PERIOD", 200))
        next_iter = trainer.iter + 1
        if period <= 0 or next_iter % period != 0:
            return

        logger = logging.getLogger("detectron2")
        small_grad_thr = float(getattr(trainer._bridge_diag_cfg, "SMALL_GRAD_THR", 1e-8))
        warn_streak = max(int(getattr(trainer._bridge_diag_cfg, "WARN_STREAK", 3)), 1)
        optimizer = trainer._trainer.optimizer
        logger.info("[BRIDGE-DIAG][TRAIN] iter=%d begin", next_iter)
        for idx, group in enumerate(optimizer.param_groups):
            group_name = group.get("group_name", f"group_{idx}")
            grad_norms: List[float] = []
            param_norms: List[float] = []
            param_count = 0
            for p in group.get("params", []):
                param_count += p.numel()
                try:
                    param_norms.append(float(p.detach().float().norm().item()))
                except Exception:
                    pass
                if p.grad is None:
                    continue
                try:
                    grad_norms.append(float(p.grad.detach().float().norm().item()))
                except Exception:
                    continue
            grad_mean = _safe_stat_mean(grad_norms)
            logger.info(
                "[BRIDGE-DIAG][TRAIN] group_name=%s lr=%.8g param_count=%d grad_norm_mean=%s grad_norm_max=%s param_norm_mean=%s param_norm_max=%s",
                group_name,
                float(group.get("lr", 0.0)),
                int(param_count),
                _format_stat(grad_mean),
                _format_stat(_safe_stat_max(grad_norms)),
                _format_stat(_safe_stat_mean(param_norms)),
                _format_stat(_safe_stat_max(param_norms)),
            )
            if grad_mean is not None and grad_mean < small_grad_thr:
                self._small_grad_streak[group_name] = self._small_grad_streak.get(group_name, 0) + 1
            else:
                self._small_grad_streak[group_name] = 0
            if self._small_grad_streak[group_name] >= warn_streak:
                logger.warning(
                    "[BRIDGE-DIAG][WARN] iter=%d group_name=%s grad_norm_mean=%s small_grad_thr=%.3e streak=%d",
                    next_iter,
                    group_name,
                    _format_stat(grad_mean),
                    small_grad_thr,
                    self._small_grad_streak[group_name],
                )

        focus_stats = trainer._bridge_diag_collect_focus_stats()
        for key, label in (
            ("sta", "STA"),
            ("prompt_encoder", "PROMPT-ENCODER"),
            ("mask_decoder_output", "MASK-DECODER-OUTPUT"),
            ("lora_all", "LORA-ALL"),
        ):
            row = focus_stats.get(key, {})
            if not row or row.get("param_count", 0) <= 0:
                logger.info("[BRIDGE-DIAG][TRAIN] focus=%s skipped", label)
                continue
            logger.info(
                "[BRIDGE-DIAG][TRAIN] focus=%s param_count=%d grad_norm_mean=%s grad_norm_max=%s param_norm_mean=%s param_norm_max=%s",
                label,
                int(row.get("param_count", 0)),
                _format_stat(row.get("grad_norm_mean")),
                _format_stat(row.get("grad_norm_max")),
                _format_stat(row.get("param_norm_mean")),
                _format_stat(row.get("param_norm_max")),
            )
        lora_by_layer = focus_stats.get("lora_by_layer", {})
        if lora_by_layer:
            for layer_name, row in lora_by_layer.items():
                logger.info(
                    "[BRIDGE-DIAG][TRAIN] focus=LORA layer=%s param_count=%d grad_norm_mean=%s grad_norm_max=%s param_norm_mean=%s param_norm_max=%s",
                    layer_name,
                    int(row.get("param_count", 0)),
                    _format_stat(row.get("grad_norm_mean")),
                    _format_stat(row.get("grad_norm_max")),
                    _format_stat(row.get("param_norm_mean")),
                    _format_stat(row.get("param_norm_max")),
                )
        model = trainer._bridge_diag_model()
        named_params = dict(model.named_parameters())
        finite_count = 0
        nonfinite_count = 0
        none_count = 0
        first_nonfinite = None
        for name in SLOT_GRAD_PROBE_NAMES:
            param = named_params.get(name, None)
            exists = param is not None
            requires_grad = bool(param.requires_grad) if exists else False
            grad = None if param is None else param.grad
            grad_is_none = grad is None
            grad_norm = None
            grad_abs_mean = None
            grad_all_finite = None
            if grad is None:
                none_count += 1
            else:
                grad_detached = grad.detach()
                try:
                    grad_norm = float(grad_detached.float().norm().item())
                except Exception:
                    grad_norm = None
                try:
                    grad_abs_mean = float(grad_detached.float().abs().mean().item())
                except Exception:
                    grad_abs_mean = None
                try:
                    grad_all_finite = bool(torch.isfinite(grad_detached).all().item())
                except Exception:
                    grad_all_finite = False
                if grad_all_finite:
                    finite_count += 1
                else:
                    nonfinite_count += 1
                    if first_nonfinite is None:
                        first_nonfinite = name
            logger.info(
                "[SLOT-GRAD-PROBE] name=%s exists=%s requires_grad=%s grad_is_none=%s grad_norm=%s grad_abs_mean=%s grad_all_finite=%s",
                name,
                exists,
                requires_grad,
                grad_is_none,
                _format_stat(grad_norm),
                _format_stat(grad_abs_mean),
                grad_all_finite,
            )
        logger.info(
            "[SLOT-GRAD-SUMMARY] first_nonfinite=%s finite_count=%d nonfinite_count=%d none_count=%d",
            first_nonfinite,
            finite_count,
            nonfinite_count,
            none_count,
        )
        fusion = getattr(model, "fusion", None)
        fusion_debug = getattr(fusion, "_last_debug", None)
        if isinstance(fusion_debug, dict):
            logger.info(
                "[SLOT-DEBUG-CACHE] slot_cosine_mean=%s slot_usage=%s slot_evidence_mean_per_slot=%s slot_evidence_std_per_slot=%s",
                _format_debug_value(fusion_debug.get("slot_cosine_mean", None)),
                _format_debug_value(fusion_debug.get("slot_usage", None)),
                _format_debug_value(fusion_debug.get("slot_evidence_mean_per_slot", None)),
                _format_debug_value(fusion_debug.get("slot_evidence_std_per_slot", None)),
            )
        else:
            logger.info("[SLOT-DEBUG-CACHE] available=False")


class _FinalCheckpointerHook(hooks.HookBase):
    def __init__(self, checkpointer):
        self.checkpointer = checkpointer

    def after_train(self):
        if comm.is_main_process():
            self.checkpointer.save("model_final")


class _CudaEmptyCacheHook(hooks.HookBase):
    def __init__(self, period=200):
        self.period = period

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self.period == 0:
            torch.cuda.empty_cache()


class _RollingCheckpointerHook(hooks.HookBase):
    def __init__(self, checkpointer, period, filename="model_last"):
        self.checkpointer = checkpointer
        self.period = period
        self.filename = filename

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self.period > 0 and next_iter % self.period == 0:
            if comm.is_main_process():
                self.checkpointer.save(self.filename)
                last_ckpt_path = os.path.join(self.checkpointer.save_dir, "last_checkpoint")
                with open(last_ckpt_path, "w") as f:
                    f.write(f"{self.filename}.pth")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    def _opts_has_key(opts, key):
        if not opts:
            return False
        for token in opts:
            if not isinstance(token, str):
                continue
            tok = token.strip("\"'")
            if tok == key or tok.startswith(key + "="):
                return True
        return False

    def _parse_route_split(value):
        if isinstance(value, (tuple, list)):
            parts = list(value)
        elif isinstance(value, str):
            s = value.strip().strip("\"'").strip("()[]")
            if "," in s:
                parts = [p for p in s.split(",") if p.strip()]
            else:
                parts = [p for p in s.split() if p.strip()]
        else:
            raise ValueError(
                "Invalid MODEL.OpenWorldSAM2.FUSION.ROUTE_SPLIT. Use: ROUTE_SPLIT: [2, 2, 2]"
            )
        if len(parts) != 3:
            raise ValueError(
                "Invalid MODEL.OpenWorldSAM2.FUSION.ROUTE_SPLIT. Use: ROUTE_SPLIT: [2, 2, 2]"
            )
        try:
            return tuple(int(p) for p in parts)
        except Exception as exc:
            raise ValueError(
                "Invalid MODEL.OpenWorldSAM2.FUSION.ROUTE_SPLIT. Use: ROUTE_SPLIT: [2, 2, 2]"
            ) from exc

    def _sanitize_route_split_opts(opts):
        if not opts:
            return opts
        key = "MODEL.OpenWorldSAM2.FUSION.ROUTE_SPLIT"
        new_opts = []
        i = 0
        while i < len(opts):
            token = opts[i]
            token_stripped = token.strip("\"'") if isinstance(token, str) else token
            if token_stripped == key and i + 1 < len(opts):
                try:
                    new_opts.append(key)
                    new_opts.append(_parse_route_split(opts[i + 1]))
                except ValueError as exc:
                    raise ValueError(
                        "Invalid MODEL.OpenWorldSAM2.FUSION.ROUTE_SPLIT. Use: ROUTE_SPLIT: [2, 2, 2]"
                    ) from exc
                i += 2
                continue
            if isinstance(token_stripped, str) and token_stripped.startswith(key + "="):
                try:
                    new_opts.append(key)
                    new_opts.append(_parse_route_split(token_stripped.split("=", 1)[1]))
                except ValueError as exc:
                    raise ValueError(
                        "Invalid MODEL.OpenWorldSAM2.FUSION.ROUTE_SPLIT. Use: ROUTE_SPLIT: [2, 2, 2]"
                    ) from exc
                i += 1
                continue
            new_opts.append(token)
            i += 1
        for j in range(0, len(new_opts) - 1, 2):
            if (
                isinstance(new_opts[j], str)
                and new_opts[j].strip("\"'") == key
                and isinstance(new_opts[j + 1], str)
            ):
                try:
                    new_opts[j + 1] = _parse_route_split(new_opts[j + 1])
                except ValueError as exc:
                    raise ValueError(
                        "Invalid MODEL.OpenWorldSAM2.FUSION.ROUTE_SPLIT. Use: ROUTE_SPLIT: [2, 2, 2]"
                    ) from exc
        return new_opts

    cfg = get_cfg()
    cfg.set_new_allowed(True)  # Add this line before merging the file
    add_open_world_sam2_config(cfg)
    # Train subset whitelist filter (default OFF / no-op)
    cfg.TRAIN_SUBSET_FILTER_ON = False
    cfg.TRAIN_SUBSET_FILTER_KEY_FILE = ""
    cfg.TRAIN_SUBSET_FILTER_KEY_FIELDS = "image_id,prompt_index"
    cfg.SOLVER.LORA_LR = 1e-5
    cfg.SOLVER.SLOT_DETECT_ANOMALY = False
    cfg.merge_from_file(args.config_file)
    sanitized_opts = _sanitize_route_split_opts(args.opts)
    if os.environ.get("OWSAM_DEBUG_ROUTE_SPLIT"):
        print("OWSAM_DEBUG_ROUTE_SPLIT opts:", sanitized_opts)
        if sanitized_opts:
            for idx in range(0, len(sanitized_opts) - 1, 2):
                if sanitized_opts[idx] == "MODEL.OpenWorldSAM2.FUSION.ROUTE_SPLIT":
                    print(
                        "OWSAM_DEBUG_ROUTE_SPLIT value type:",
                        type(sanitized_opts[idx + 1]),
                        "value:",
                        sanitized_opts[idx + 1],
                    )
    cfg.merge_from_list(sanitized_opts)
    logger = logging.getLogger("detectron2")
    logger.info("IMS_PER_BATCH after merge_from_list: %s", cfg.SOLVER.IMS_PER_BATCH)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f"run_{args.run_idx}")
    if not _opts_has_key(args.opts, "SOLVER.IMS_PER_BATCH"):
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    logger.info("IMS_PER_BATCH after batch_size override: %s", cfg.SOLVER.IMS_PER_BATCH)
    if not _opts_has_key(args.opts, "SOLVER.BASE_LR"):
        cfg.SOLVER.BASE_LR = args.lr
    logger.info("BASE_LR after override logic: %s", cfg.SOLVER.BASE_LR)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="open-world-sam2")
    return cfg

def set_seed(seed=42):
    # Set random seeds for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed()
    cfg = setup(args)
    if getattr(cfg.SOLVER, "SLOT_DETECT_ANOMALY", False):
        warnings.simplefilter("default")
    # print("Command cfg:", cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        model.metadata = MetadataCatalog.get(cfg['DATASETS']['TEST'][0])
        print(cfg.OUTPUT_DIR)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--run_idx', default=0, type=int, metavar='N',
                        help='index of the experiment')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', default=0.0001, type=float,)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
