"""Open-vocabulary change detection built on the native SAM 3 engine."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from PIL import Image
from skimage import measure
from skimage.filters import threshold_otsu

from .engine import Sam3OVSSEngine
from .fusion import logits_to_prediction, resize_map
from .loading import DEFAULT_MODEL_ID, load_sam3
from .prompts import normalize_text_prompts


def compute_fpn_similarity_map(fpn_features_t1, fpn_features_t2, target_size):
    """Cosine similarity between the last FPN levels of two timestamps."""

    def process_fpn_features(fpn_features, target_size):
        upsampled = []
        levels = fpn_features[-1:] if not torch.is_tensor(fpn_features) else [fpn_features]
        for feat in levels:
            channels = feat.shape[1]
            if channels > 320:
                part1 = F.interpolate(
                    feat[:, :320], size=target_size, mode="bilinear", align_corners=False
                )
                part2 = F.interpolate(
                    feat[:, 320:], size=target_size, mode="bilinear", align_corners=False
                )
                upsampled.append(torch.cat([part1, part2], dim=1))
            else:
                upsampled.append(
                    F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
                )
        return torch.cat(upsampled, dim=1)

    feat_t1 = process_fpn_features(fpn_features_t1, target_size)
    feat_t2 = process_fpn_features(fpn_features_t2, target_size)
    cosine_sim = F.cosine_similarity(feat_t1, feat_t2, dim=1)
    return (cosine_sim * 0.5 + 0.5).squeeze(1)


@MODELS.register_module()
class SegEarthOV3CDSeg(BaseSegmentor):
    def __init__(
        self,
        text_prompts=None,
        model_id: str = DEFAULT_MODEL_ID,
        device=None,
        prob_thd: float = 0.0,
        bg_idx: int = 0,
        slide_stride: int = 0,
        slide_crop: int = 0,
        confidence_threshold: float = 0.5,
        use_sem_seg: bool = True,
        use_presence_score: bool = True,
        use_transformer_decoder: bool = True,
        **kwargs,
    ):
        data_preprocessor = kwargs.pop("data_preprocessor", None)
        if data_preprocessor is None:
            try:
                data_preprocessor = SegDataPreProcessor()
            except Exception:
                data_preprocessor = None
        init_kwargs = {}
        if data_preprocessor is not None:
            init_kwargs["data_preprocessor"] = data_preprocessor
        try:
            super().__init__(**init_kwargs)
        except TypeError:
            super().__init__()

        if text_prompts is None:
            raise ValueError(
                "SegEarthOV3CDSeg now takes text_prompts=['background', 'building'], "
                "not a dataset classname_path txt file."
            )

        self.prompt_spec = normalize_text_prompts(text_prompts)
        self.num_cls = self.prompt_spec.num_classes
        self.num_queries = self.prompt_spec.num_queries
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, processor = load_sam3(model_id, device=resolved_device)
        self.engine = Sam3OVSSEngine(
            model,
            processor,
            confidence_threshold=confidence_threshold,
            use_sem_seg=use_sem_seg,
            use_presence_score=use_presence_score,
            use_transformer_decoder=use_transformer_decoder,
            device=resolved_device,
        )
        self.query_idx = torch.tensor(
            self.prompt_spec.query_idx, dtype=torch.int64, device=self.engine.device
        )
        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
        self.instance_iou_threshold = kwargs.get("instance_iou_threshold", 0.3)
        self.t12_min_instance_area = kwargs.get("t12_min_instance_area", 20)
        self.model_type = kwargs.get("model_type", "SAM3")

    def _get_discrete_pred(self, logits: torch.Tensor) -> torch.Tensor:
        return logits_to_prediction(logits, prob_thd=self.prob_thd, bg_idx=self.bg_idx)

    def _infer_image(self, image: Image.Image, ori_shape):
        logits = self.engine.infer_image(
            image,
            self.prompt_spec,
            slide_crop=self.slide_crop,
            slide_stride=self.slide_stride,
        )
        if tuple(logits.shape[-2:]) != tuple(ori_shape):
            logits = resize_map(logits, tuple(ori_shape))
        return logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
            ] * inputs.shape[0]

        for i, meta in enumerate(batch_img_metas):
            ori_shape = tuple(meta["ori_shape"])
            img_path_t1 = meta.get("img1_path")
            img_path_t2 = meta.get("img2_path")
            if img_path_t1 is None or img_path_t2 is None:
                raise ValueError("Change detection samples need img1_path and img2_path")

            img_t1 = Image.open(img_path_t1).convert("RGB")
            img_t2 = Image.open(img_path_t2).convert("RGB")
            if "img_path" not in meta:
                data_samples[i].set_metainfo(dict(img_path=img_path_t1))

            logits_t1 = self._infer_image(img_t1, ori_shape)
            logits_t2 = self._infer_image(img_t2, ori_shape)

            pred_inst_t1 = self._get_discrete_pred(logits_t1)
            pred_inst_t2 = self._get_discrete_pred(logits_t2)
            change_pred_inst = self._detect_instance_changes(pred_inst_t1, pred_inst_t2)

            fpn_t1, _ = self.engine.extract_fpn_features(img_t1)
            fpn_t2, _ = self.engine.extract_fpn_features(img_t2)
            sim_map = compute_fpn_similarity_map(fpn_t1, fpn_t2, ori_shape).squeeze(0)
            feature_change_map = 1.0 - sim_map

            change_pred_sem = torch.full(
                ori_shape, self.bg_idx, dtype=torch.long, device=logits_t1.device
            )
            for cls_id in range(self.num_cls):
                if cls_id == self.bg_idx:
                    continue
                prompt_indices = (self.query_idx == cls_id).nonzero(as_tuple=True)[0]
                if len(prompt_indices) == 0:
                    continue
                # After synonym aggregation, logits are already class-major.
                if logits_t1.shape[0] == self.num_cls:
                    delta_c_max = torch.abs(logits_t1[cls_id] - logits_t2[cls_id])
                    max_conf_c = torch.max(logits_t1[cls_id], logits_t2[cls_id])
                else:
                    delta_c_max = torch.abs(
                        logits_t1[prompt_indices] - logits_t2[prompt_indices]
                    ).max(dim=0)[0]
                    max_conf_c = torch.max(
                        logits_t1[prompt_indices], logits_t2[prompt_indices]
                    ).max(dim=0)[0]

                energy_map = max_conf_c * delta_c_max * feature_change_map
                energy_np = energy_map.detach().float().cpu().numpy()
                otsu_thd = threshold_otsu(energy_np) if np.ptp(energy_np) > 1e-4 else 0.0
                change_pred_sem[energy_map > max(otsu_thd, 0.01)] = cls_id

            agreed_mask = (change_pred_inst != self.bg_idx) & (
                change_pred_sem != self.bg_idx
            )
            change_pred = torch.full(
                ori_shape, self.bg_idx, dtype=torch.long, device=logits_t1.device
            )
            change_pred[agreed_mask] = change_pred_inst[agreed_mask]

            data_samples[i].set_data(
                {
                    "seg1_logits": PixelData(**{"data": logits_t1}),
                    "seg2_logits": PixelData(**{"data": logits_t2}),
                    "pred_sem_seg": PixelData(**{"data": change_pred.unsqueeze(0)}),
                }
            )
        return data_samples

    def _detect_instance_changes(self, mask_t1, mask_t2, inst_dilation_radius=2):
        mask1 = (mask_t1 > 0).float().unsqueeze(0).unsqueeze(0)
        mask2 = (mask_t2 > 0).float().unsqueeze(0).unsqueeze(0)

        if inst_dilation_radius > 0:
            kernel_size = inst_dilation_radius * 2 + 1
            mask1_dilated = F.max_pool2d(
                mask1, kernel_size, stride=1, padding=inst_dilation_radius
            )
            mask2_dilated = F.max_pool2d(
                mask2, kernel_size, stride=1, padding=inst_dilation_radius
            )
        else:
            mask1_dilated = mask1
            mask2_dilated = mask2

        mask1 = mask1.squeeze().cpu().numpy() > 0
        mask2 = mask2.squeeze().cpu().numpy() > 0
        mask1_dilated = mask1_dilated.squeeze().cpu().numpy() > 0
        mask2_dilated = mask2_dilated.squeeze().cpu().numpy() > 0

        def get_unidirectional_changes(source_mask, target_mask_dilated):
            lbl = measure.label(source_mask, connectivity=2)
            area = np.bincount(lbl.ravel())
            cover = np.bincount(
                lbl.ravel(), weights=target_mask_dilated.ravel().astype(np.float64)
            )
            ratio = cover / np.maximum(area, 1)
            changed_ids = np.where(
                (ratio < self.instance_iou_threshold)
                & (area >= self.t12_min_instance_area)
            )[0]
            changed_ids = changed_ids[changed_ids != 0]
            return np.isin(lbl, changed_ids)

        change_np = get_unidirectional_changes(mask1, mask2_dilated) | get_unidirectional_changes(
            mask2, mask1_dilated
        )
        return torch.from_numpy(change_np).long().to(mask_t1.device)

    def _forward(self, data_samples):
        return None

    def inference(self, img, batch_img_metas):
        return None

    def encode_decode(self, inputs, batch_img_metas):
        return None

    def extract_feat(self, inputs):
        return None

    def loss(self, inputs, data_samples):
        return None
