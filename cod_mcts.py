#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import cv2
import json
import math
import time
import argparse
import random
import numpy as np
import torch

from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from qwen_vl_utils import process_vision_info

# -----------------------
# Utils
# -----------------------

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def save_image(path: str, img: np.ndarray):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def parse_list_arg(x: str):
    # "desk,drawer,black door handle" -> ["desk","drawer","black door handle"]
    return [t.strip() for t in x.split(",") if t.strip()]

# -----------------------
# Detectron2 (RPN) setup
# -----------------------

def setup_rpn(score_thresh: float = 0.2, device: str = "cuda"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    return predictor

def generate_proposals(predictor, image: np.ndarray) -> np.ndarray:
    outputs = predictor(image)
    boxes = outputs["instances"].pred_boxes.tensor.detach().cpu().numpy()
    # boxes: (N, 4) [x1, y1, x2, y2]
    return boxes

# -----------------------
# Qwen2-VL scoring
# -----------------------

def build_vlm(qwen_model_path: str, qwen_processor_path: str, device: str = "cuda"):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        qwen_model_path,
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(qwen_processor_path)
    return model, processor

_SCORE_RE = re.compile(r"([01](?:\.\d+)?|\.\d+)")  # parse 0~1 float

def vlm_score_image(model, processor, image_bgr: np.ndarray, query: str, tmp_dir: str) -> float:
    """
    Ask the VLM to output a single confidence score in [0,1].
    Fallback: yes/no -> 1/0.
    """
    ensure_dir(tmp_dir)
    tmp_path = os.path.join(tmp_dir, f"vlm_{time.time_ns()}.jpg")
    cv2.imwrite(tmp_path, image_bgr)

    # Instruction: output only a number between 0 and 1.
    prompt = (
        f"You are a precise visual discriminator. "
        f"Given the query: '{query}', respond with a SINGLE number in [0,1] indicating your confidence. "
        f"Do not output any text besides the number."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": tmp_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=16)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    # Try parse numeric
    m = _SCORE_RE.search(out_text)
    if m:
        try:
            v = float(m.group(1))
            return float(max(0.0, min(1.0, v)))
        except:
            pass

    # Fallback yes/no
    if "yes" in out_text.lower():
        return 1.0
    if "no" in out_text.lower():
        return 0.0

    # Last resort
    return 0.0

# -----------------------
# Crop & light augmentations
# -----------------------

def crop_by_box(image: np.ndarray, box, clip=True):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    if clip:
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()

def resize_square(image: np.ndarray, S: int):
    return cv2.resize(image, (S, S), interpolation=cv2.INTER_LINEAR)

def light_augs(image: np.ndarray, n: int) -> list:
    out = []
    h, w = image.shape[:2]
    for _ in range(n):
        img = image.copy()
        # brightness
        alpha = 1.0 + np.random.uniform(-0.1, 0.1)
        beta = np.random.uniform(-12, 12)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        # slight rotation/scale
        ang = np.random.uniform(-5, 5)
        scale = 1.0 + np.random.uniform(-0.05, 0.05)
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, scale)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        out.append(img)
    return out

# -----------------------
# MCTS (flat-UCB bandit)
# -----------------------

class CandidateStat:
    __slots__ = ("N", "W")
    def __init__(self):
        self.N = 0  # visits
        self.W = 0.0  # total reward

    @property
    def Q(self):
        return 0.0 if self.N == 0 else self.W / self.N

def ucb1(Q, N_parent, N_child, c):
    if N_child == 0:
        return float("inf")
    return Q + c * math.sqrt(math.log(N_parent + 1) / (N_child))

def mcts_select(proposals, stats: dict, c_ucb: float):
    # proposals indexed 0..M-1
    N_parent = sum(s.N for s in stats.values()) + 1
    best_idx = None
    best_score = -1e9
    for idx in range(len(proposals)):
        s = stats.setdefault(idx, CandidateStat())
        score = ucb1(s.Q, N_parent, s.N, c_ucb)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx

def mcts_search_one_step(
    image_step: np.ndarray,
    di_text: str,
    proposals: np.ndarray,
    model, processor,
    S: int,
    B: int,
    c_ucb: float,
    tmp_dir: str
):
    stats = {}  # idx -> CandidateStat
    # rollouts
    for _ in range(B):
        idx = mcts_select(proposals, stats, c_ucb)
        box = proposals[idx]
        crop = crop_by_box(image_step, box)
        if crop is None or crop.size == 0:
            r = 0.0
        else:
            crop_rsz = resize_square(crop, S)
            r = vlm_score_image(model, processor, crop_rsz, di_text, tmp_dir)
        s = stats[idx]
        s.N += 1
        s.W += float(r)
    # rank by Q
    ranked = sorted(range(len(proposals)), key=lambda k: stats[k].Q if k in stats else -1, reverse=True)
    return ranked, stats

def verify_with_augs(
    image_step: np.ndarray,
    box,
    di_text: str,
    model, processor,
    S: int,
    N_aug: int,
    tau: float,
    epsilon: float,
    tmp_dir: str
):
    crop = crop_by_box(image_step, box)
    if crop is None or crop.size == 0:
        return False, 0.0, []

    crop_rsz = resize_square(crop, S)
    aug_imgs = [crop_rsz] + light_augs(crop_rsz, N_aug - 1) if N_aug > 0 else [crop_rsz]

    scores = []
    for img in aug_imgs:
        s = vlm_score_image(model, processor, img, di_text, tmp_dir)
        scores.append(s)
    mean_s = float(np.mean(scores))
    range_s = float(np.max(scores) - np.min(scores))
    ok = (mean_s >= tau) and (range_s <= epsilon)
    return ok, mean_s, scores

# -----------------------
# CoD stage runner
# -----------------------

def run_cod_mcts(
    image_path: str,
    detection_chain: list,
    qwen_model_path: str,
    qwen_processor_path: str,
    device: str,
    rpn_thresh: float,
    B: int,
    S: int,
    tau: float,
    epsilon: float,
    n_aug: int,
    c_ucb: float,
    relax_tau_factor: float,
    relax_rounds: int,
    out_dir: str,
    save_debug: bool
):
    set_seed(1234)
    ensure_dir(out_dir)
    tmp_dir = ensure_dir(os.path.join(out_dir, "_tmp"))

    # build models
    model, processor = build_vlm(qwen_model_path, qwen_processor_path, device)
    predictor = setup_rpn(rpn_thresh, device)

    image0 = load_image(image_path)
    I_i = image0.copy()
    off_x, off_y = 0, 0  # for bbox accumulation
    accepted_chain = []
    all_logs = []

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for i, di in enumerate(detection_chain, 1):
        # 1) proposals on current image
        props = generate_proposals(predictor, I_i)
        if props.shape[0] == 0:
            all_logs.append({"step": i, "di": di, "status": "no_proposals"})
            break

        # 2) MCTS bandit search
        ranked_idx, stats = mcts_search_one_step(
            I_i, di, props, model, processor, S, B, c_ucb, tmp_dir
        )

        # 3) verification & backtracking
        acc = None
        cur_tau = tau
        for round_k in range(relax_rounds):
            for idx in ranked_idx:
                box = props[idx]
                ok, mean_s, scores = verify_with_augs(
                    I_i, box, di, model, processor, S, n_aug, cur_tau, epsilon, tmp_dir
                )
                log_item = {
                    "step": i, "di": di, "idx": int(idx),
                    "box": [float(v) for v in box.tolist()],
                    "mean": mean_s, "scores": scores, "tau": cur_tau, "ok": ok
                }
                all_logs.append(log_item)

                if ok:
                    # accept
                    acc = (idx, box, mean_s)
                    # debug save
                    if save_debug:
                        crop = crop_by_box(I_i, box)
                        save_image(os.path.join(out_dir, f"{base_name}_step{i}_accept.jpg"),
                                   crop if crop is not None else I_i)
                    break

            if acc is not None:
                break
            # relax tau and retry
            cur_tau *= relax_tau_factor

        if acc is None:
            all_logs.append({"step": i, "di": di, "status": "no_accept"})
            break

        # 4) advance to next step with accepted ROI
        idx, box, mean_s = acc
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        off_x, off_y = off_x + x1, off_y + y1
        I_i_next = crop_by_box(I_i, box)
        if I_i_next is None or I_i_next.size == 0:
            all_logs.append({"step": i, "di": di, "status": "crop_failed"})
            break

        # record accepted
        accepted_chain.append({
            "step": i,
            "prompt": di,
            "bbox_global_xyxy": [x1 + (off_x - x1), y1 + (off_y - y1), x2 + (off_x - x1), y2 + (off_y - y1)],
            "score": mean_s
        })

        # optional debug
        if save_debug:
            dbg_I = I_i.copy()
            cv2.rectangle(dbg_I, (x1, y1), (x2, y2), (0, 255, 0), 2)
            save_image(os.path.join(out_dir, f"{base_name}_step{i}_bbox.jpg"), dbg_I)
            save_image(os.path.join(out_dir, f"{base_name}_step{i}_crop.jpg"), I_i_next)

        I_i = I_i_next

    # save summary
    with open(os.path.join(out_dir, f"{base_name}_mcts_logs.json"), "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, f"{base_name}_accepted.json"), "w", encoding="utf-8") as f:
        json.dump(accepted_chain, f, indent=2, ensure_ascii=False)

    return accepted_chain, all_logs

# -----------------------
# CLI
# -----------------------

def build_argparser():
    ap = argparse.ArgumentParser(
        description="CoD MCTS ROI selection with Qwen2-VL scoring (parameterized)."
    )
    ap.add_argument("--image_path", type=str, required=True, help="Path to input image.")
    ap.add_argument("--detection_chain", type=str, required=True,
                    help='Comma-separated chain, e.g. "desk,drawer,black door handle"')
    ap.add_argument("--qwen_model_path", type=str, required=True,
                    help="Path or hub id for Qwen2-VL model (e.g., /path/Qwen2-VL-7B-Instruct)")
    ap.add_argument("--qwen_processor_path", type=str, required=True,
                    help="Path or hub id for Qwen2-VL processor")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--rpn_thresh", type=float, default=0.20, help="RPN score threshold")
    ap.add_argument("--B", type=int, default=64, help="MCTS rollout budget per step")
    ap.add_argument("--S", type=int, default=224, help="Crop resize size")
    ap.add_argument("--tau", type=float, default=0.65, help="verification mean threshold")
    ap.add_argument("--epsilon", type=float, default=0.20, help="verification range (max-min)")
    ap.add_argument("--n_aug", type=int, default=5, help="#augmentations for consistency check")
    ap.add_argument("--ucb_c", type=float, default=1.4142, help="UCB exploration constant")
    ap.add_argument("--relax_tau_factor", type=float, default=0.95,
                    help="Multiply tau if no candidate passes")
    ap.add_argument("--relax_rounds", type=int, default=3,
                    help="How many relax rounds to try per step")
    ap.add_argument("--out_dir", type=str, default="./cod_mcts_out", help="Output dir")
    ap.add_argument("--save_debug", action="store_true", help="Save debug crops/boxes")
    return ap

def main():
    args = build_argparser().parse_args()

    accepted, logs = run_cod_mcts(
        image_path=args.image_path,
        detection_chain=parse_list_arg(args.detection_chain),
        qwen_model_path=args.qwen_model_path,
        qwen_processor_path=args.qwen_processor_path,
        device=args.device,
        rpn_thresh=args.rpn_thresh,
        B=args.B,
        S=args.S,
        tau=args.tau,
        epsilon=args.epsilon,
        n_aug=args.n_aug,
        c_ucb=args.ucb_c,
        relax_tau_factor=args.relax_tau_factor,
        relax_rounds=args.relax_rounds,
        out_dir=args.out_dir,
        save_debug=args.save_debug
    )

    print("[ACCEPTED STEPS]")
    for a in accepted:
        print(a)

if __name__ == "__main__":
    main()
