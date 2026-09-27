"""MMSegmentation wrapper around the native transformers SAM 3 pipeline."""

from __future__ import annotations

import torch
from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from PIL import Image

from .engine import Sam3OVSSEngine
from .loading import DEFAULT_MODEL_ID, load_sam3
from .prompts import normalize_text_prompts


@MODELS.register_module()
class SegEarthOV3Segmentation(BaseSegmentor):
    def __init__(
        self,
        text_prompts=None,
        model_id: str = DEFAULT_MODEL_ID,
        device=None,
        prob_thd=0.0,
        bg_idx=0,
        slide_stride=0,
        slide_crop=0,
        confidence_threshold=0.5,
        use_sem_seg=True,
        use_presence_score=True,
        use_transformer_decoder=True,
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
                "SegEarthOV3Segmentation now takes text_prompts=['class a', ...], "
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
        self.query_words = self.prompt_spec.query_words
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
        self.model_type = kwargs.get("model_type", "SAM3")

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
            image_path = meta.get("img_path")
            if image_path is None:
                raise ValueError("Each data sample must provide metainfo['img_path']")
            image = Image.open(image_path).convert("RGB")
            ori_shape = tuple(meta["ori_shape"])
            seg_logits, seg_pred = self.engine.predict_mask(
                image,
                self.prompt_spec,
                slide_crop=self.slide_crop,
                slide_stride=self.slide_stride,
                prob_thd=self.prob_thd,
                bg_idx=self.bg_idx,
                target_size=ori_shape,
            )
            data_samples[i].set_data(
                {
                    "seg_logits": PixelData(**{"data": seg_logits}),
                    "pred_sem_seg": PixelData(**{"data": seg_pred.unsqueeze(0)}),
                }
            )
        return data_samples

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
