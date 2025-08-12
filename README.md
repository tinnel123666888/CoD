# Chain-of-Detection (CoD)

This repository contains two core components:

- **`cod_mcts.py`** — Coarse-to-fine **ROI selection with MCTS + backtracking** using an RPN (Detectron2) and a VLM (Qwen2-VL) as the ROI discriminator.
- **`chain_of_detection.py`** — A **Detection Chain Generator** that uses Qwen2-VL to produce a short sequence of prompts (e.g., `[broad object] -> [fine-grained part]`) given an image and an operation description.

> The RPN is provided via **Detectron2**, and the VLM is **Qwen2-VL**. Please install both before running.

---

## 1) Installation

### A. Detectron2 (RPN)
- GitHub: https://github.com/facebookresearch/detectron2  
- Install (Linux/macOS; pick the wheel matching your Torch/CUDA):
  ```bash
  pip install 'git+https://github.com/facebookresearch/detectron2.git'
````

> For detailed, version-matched instructions, see Detectron2’s official install page.

### B. Qwen2-VL (Vision-Language Model)

* GitHub: [https://github.com/QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)
* Basic deps:

  ```bash
  pip install transformers accelerate einops tiktoken
  ```
* Make sure the utility **`qwen_vl_utils.py`** (from Qwen2-VL examples) is importable (e.g., placed in repo root or added to `PYTHONPATH`), since both scripts call:

  ```python
  from qwen_vl_utils import process_vision_info
  ```

### C. Common

```bash
pip install opencv-python numpy
```

---

## 2) Repository Structure

```
.
├── cod_mcts.py               # CoD ROI selection with MCTS + backtracking (parameterized CLI)
├── chain_of_detection.py     # Qwen2-VL-based detection chain generator (CLI)
└── README.md
```

---

## 3) Models & Checkpoints

Prepare local or HF hub paths for Qwen2-VL:

* `--qwen_model_path`: e.g. `/path/to/Qwen2-VL-7B-Instruct`
* `--qwen_processor_path`: same directory or HF id

Detectron2 will automatically download the default Faster R-CNN R50-FPN 3x weights used as an RPN backbone in this project.

---

## 4) Running CoD MCTS Inference (`cod_mcts.py`)

**What it does:**
For each step prompt in your detection chain (e.g., `desk, drawer, black door handle`), it:

1. runs RPN to get proposals,
2. runs **MCTS (UCB1)** over proposals with the VLM as a scoring function,
3. verifies the top candidate with **light augmentations** (mean ≥ τ, range ≤ ε),
4. **backtracks** to the next-best if verification fails; optionally relaxes τ and retries,
5. crops the accepted ROI and continues to the next chain step.

**Example:**

```bash
python cod_mcts.py \
  --image_path /abs/path/to/image.png \
  --detection_chain "desk,drawer,black door handle" \
  --qwen_model_path /abs/path/to/Qwen2-VL-7B-Instruct \
  --qwen_processor_path /abs/path/to/Qwen2-VL-7B-Instruct \
  --device cuda \
  --rpn_thresh 0.20 \
  --B 64 \
  --S 224 \
  --tau 0.65 \
  --epsilon 0.20 \
  --n_aug 5 \
  --ucb_c 1.4142 \
  --relax_tau_factor 0.95 \
  --relax_rounds 3 \
  --out_dir ./cod_mcts_out \
  --save_debug
```

**Key arguments:**

* `--image_path`: Input image.
* `--detection_chain`: Comma-separated prompts for each step (coarse → fine).
* `--qwen_model_path`, `--qwen_processor_path`: Qwen2-VL model/processor paths or HF ids.
* `--B`: MCTS rollout budget per step.
* `--S`: Crop resize (square) size.
* `--tau`, `--epsilon`: Verification mean/range thresholds.
* `--n_aug`: Number of light augmentations for consistency check.
* `--ucb_c`: UCB exploration constant.
* `--relax_tau_factor`, `--relax_rounds`: How τ is relaxed and retried if verification fails.
* `--out_dir`: Outputs (accepted steps, logs, debug crops).

Outputs:

* `*_accepted.json`: accepted ROIs for each chain step.
* `*_mcts_logs.json`: per-candidate MCTS/verification logs.
* Debug images if `--save_debug` is set.

---

## 5) Generating Detection Chains (`chain_of_detection.py`)

**What it does:**
Given an image and an operation text (e.g., “open the drawer”), it prompts Qwen2-VL to produce a **two-to-three term** detection chain from broad → fine parts.

**Script content (summary):**

```python
# chain_of_detection.py
# - Loads Qwen2-VL model & processor
# - Prompts: "The robot needs to perform ... Please provide the detection steps ..."
# - Returns a short chain like: [desk], [drawer], [black door handle]
```

**Usage:**

```bash
python chain_of_detection.py \
  /abs/path/to/Qwen2-VL-7B-Instruct \
  /abs/path/to/Qwen2-VL-7B-Instruct \
  /abs/path/to/image.png \
  "Open the drawer"
```

Arguments (positional):

1. `model_path` — Qwen2-VL model path or HF id
2. `processor_path` — Qwen2-VL processor path or HF id
3. `image_path` — input image
4. `text_content` — operation description

**Notes:**

* The script uses `qwen_vl_utils.process_vision_info` (ensure `qwen_vl_utils.py` is available).
* It prints the generated chain to stdout.

---

## 6) Typical Workflow

1. **Generate a chain** (coarse → fine) with `chain_of_detection.py`.
2. **Run MCTS** with `cod_mcts.py` using the generated chain to localize each step’s ROI progressively.
3. (Optional) Use outputs to create fine-grained annotations or for downstream grasp/manipulation modules.

---


