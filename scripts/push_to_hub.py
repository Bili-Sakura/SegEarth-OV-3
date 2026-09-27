#!/usr/bin/env python3
"""Publish the Hub-standard snapshot in ./hub to a model repository.

Official Transformers flow (uploads pipeline.py into config.json custom_pipelines
and, with --with-weights, the loaded SAM 3 checkpoint):

    python scripts/push_to_hub.py --repo-id YOUR_NAMESPACE/SegEarth-OV-3 --with-weights

Code-only snapshot (pipeline.py + config.json + README, no multi-GB weights):

    python scripts/push_to_hub.py --repo-id YOUR_NAMESPACE/SegEarth-OV-3 --code-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUB_DIR = ROOT / "hub"
EXTRA_CONFIG_KEYS = (
    "custom_pipelines",
    "text_prompts",
    "confidence_threshold",
    "prob_thd",
    "bg_idx",
    "slide_crop",
    "slide_stride",
    "use_sem_seg",
    "use_presence_score",
    "use_transformer_decoder",
)


def _hub_extras() -> dict:
    raw = json.loads((HUB_DIR / "config.json").read_text())
    return {key: raw[key] for key in EXTRA_CONFIG_KEYS if key in raw}


def _import_hub_pipeline():
    import importlib.util

    path = HUB_DIR / "pipeline.py"
    spec = importlib.util.spec_from_file_location("segearth_ov3_hub_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def push_code_only(repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=private, repo_type="model")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(HUB_DIR),
        repo_type="model",
        commit_message="Add SegEarth-OV3 custom transformers pipeline",
        ignore_patterns=["__pycache__/*", "*.pyc"],
    )
    print(f"Uploaded code snapshot to https://huggingface.co/{repo_id}")


def push_with_weights(repo_id: str, model_id: str, private: bool) -> None:
    from transformers import Sam3Model, Sam3Processor, pipeline as hf_pipeline

    hub = _import_hub_pipeline()
    hub.register_pipeline()

    model = Sam3Model.from_pretrained(model_id)
    processor = Sam3Processor.from_pretrained(model_id)
    for key, value in _hub_extras().items():
        setattr(model.config, key, value)
    pipe = hf_pipeline(
        "segearth-ov3-segmentation",
        model=model,
        tokenizer=processor.tokenizer,
        image_processor=processor.image_processor,
        processor=processor,
        trust_remote_code=False,
    )
    # Copies pipeline.py next to the saved model and writes custom_pipelines
    # into config.json (see https://huggingface.co/docs/transformers/add_new_pipeline).
    pipe.push_to_hub(repo_id, private=private)
    api_readme = HUB_DIR / "README.md"
    if api_readme.exists():
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(api_readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
    print(f"Pushed pipeline + weights to https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hub repo, e.g. BiliSakura/SegEarth-OV-3")
    parser.add_argument("--model-id", default="facebook/sam3", help="Source SAM 3 checkpoint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--code-only",
        action="store_true",
        help="Upload hub/pipeline.py, config.json, and README only",
    )
    group.add_argument(
        "--with-weights",
        action="store_true",
        help="Official Pipeline.push_to_hub (includes SAM 3 weights)",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not (HUB_DIR / "pipeline.py").exists():
        sys.exit(f"Missing {HUB_DIR / 'pipeline.py'}")

    if args.code_only:
        push_code_only(args.repo_id, args.private)
    else:
        push_with_weights(args.repo_id, args.model_id, args.private)


if __name__ == "__main__":
    main()
