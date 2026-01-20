# MapVerse

## Overview

The repository is designed to make it easy to run multiple models on the **MapVerse** dataset, apply controlled perturbations, and analyze results in a reproducible way.

---

## Repository Structure

```
MapVerse/
├── data/
│   ├── imgs/                   # Image assets
│   └── typed_questions.csv     # Input questions CSV
├── eval/
│   ├── Eval_Analysis.ipynb     # Eval: Boolean, Single Entity, Counting, Listing
│   └── Eval_Analysis2.ipynb    # Eval: Ranking, Reasoning
├── legacy/                     # Legacy per-model QA scripts
│   ├── ayavision_qa.py
│   ├── cogvlm_qa.py
│   ├── deepseek_qa.py
│   ├── gemini_qa.py
│   ├── idefics_qa.py
│   ├── intern_qa.py
│   ├── llama_qa.py
│   ├── molmo_qa.py
│   └── qwen_qa.py
├── models/                     # Current unified model wrappers
│   ├── __init__.py
│   ├── ayavision.py
│   ├── cogvlm.py
│   ├── deepseek.py
│   ├── gemini.py
│   ├── idefics.py
│   ├── internvl.py
│   ├── llama.py
│   ├── molmo.py
│   └── qwen.py
├── utils/
│   ├── __init__.py
│   ├── common.py               # CSV & image loading helpers
│   └── perturb.py              # Image perturbation functions
├── viz/
│   ├── heat.py                 # Heatmap visualizations
│   └── plot.py                 # UMAP / t-SNE embedding plots
├── openai_qa_with_image.ipynb  # OpenAI batch QA (with images)
├── openai_qa_without_image.ipynb # OpenAI batch QA (text-only)
├── openai_result_fetch.ipynb   # Fetch & parse OpenAI batch results
├── predict.py                  # Central runner for model wrappers
├── requirements.txt           # Dependencies for all models except DeepSeek
├── requirements_deepseek.txt  # Dependencies specific to DeepSeek models
└── README.md
```

---

## Installation & Requirements

The repository uses separate dependency files to simplify environment setup:

* **`requirements.txt`**: Required for all models **except DeepSeek**
* **`requirements_deepseek.txt`**: Additional / separate dependencies required to run **DeepSeek** models

Typical setup:

```bash
pip install -r requirements.txt
```

For DeepSeek:

```bash
pip install -r requirements_deepseek.txt
```

---

## Dataset 

* **`typed_questions.csv`**: Contains the question–answer pairs along with associated metadata (e.g., question type, identifiers, and other annotations used during evaluation).
* **`data/imgs/`**: Contains the corresponding map images referenced by the CSV.
--

## Prediction Files

### `predict.py`

`predict.py` is the main execution script. It dynamically imports a model wrapper from `models/` and runs it over an input CSV containing questions and (optionally) images.

Each model wrapper must define:

* `ask_image_question(...)`: the main inference function
* `CUSTOM_PROMPT`: the prompt template used by the model

### `models/`

Contains per-model wrappers. Each wrapper standardizes how a given model is queried, making it easy to swap models without changing the evaluation pipeline.

### `common.py`

Shared utilities for:

* Loading CSV files
* Resolving image paths
* Handling missing or optional image inputs

### `perturb.py`

Defines image perturbation functions used for robustness and stress testing (e.g., compression artifacts, occlusions).

---

## Running `predict.py`

### Usage

```bash
python predict.py <model> \
  [--input INPUT] \
  [--output OUTPUT] \
  [--image-base IMAGE_BASE] \
  [--no-image] \
  [--perturb PERTURB]
```

### Default Arguments

* `--input`: `typed_questions.csv`
* `--image-base`: `data/imgs/`
* `--output`: auto-generated under `results/` if not specified

### Examples

Run Qwen with image input:

```bash
python predict.py qwen \
  --input typed_questions.csv \
  --image-base data/imgs/
```

Run a text-only model (no image input):

```bash
python predict.py llama --no-image
```

Run with an image perturbation:

```bash
python predict.py qwen --perturb jpeg_compress
python predict.py qwen --perturb add_random_black_box
```

### Notes on Perturbations

* `--perturb` accepts the name of a function defined in `perturb.py`.
* Some perturbation functions can also support adding arguments in the cli using `name:spec` syntax (e.g., `jpeg_compress:40`) if external argument is accepted.
* If not implemented, use simple function names only.

---

## OpenAI Batch QA (Notebooks)

The notebooks directory contains workflows for running large-scale OpenAI batch jobs.

### Notebooks Overview

* **`openai_qa_with_image.ipynb`**
  Build and submit batch jobs that include base64-encoded images. Poll for completion and save results as JSONL and CSV.

* **`openai_qa_without_image.ipynb`**
  Build and submit text-only batch jobs (no images). Poll and save results.

* **`openai_result_fetch.ipynb`**
  Inspect, retrieve, and download results from previously submitted batch jobs, and parse JSONL outputs.

---

## Visualizations

### `heat.py`

* Edit the CSV path at the top of the script if needed.
* Generates heatmap visualizations.
* Output images are saved to:

```
heat_maps/
```

### `plot.py`

* Edit constants at the top of the script before running:

  * `CSV_PATH`
  * `IMAGE_ROOT`
  * `USE_UMAP`
* Supports UMAP and t-SNE visualizations.
* Outputs are saved to:

```
plots_umap/
plots_tsne/
```

---

## Evaluation & Analysis

### `Eval_Analysis.ipynb`

Evaluation for the following question types:

* Boolean
* Single Entity
* Counting
* Listing

### `Eval_Analysis2.ipynb`

Evaluation for more complex question types:

* Ranking
* Reasoning

These notebooks aggregate model outputs and compute task-specific metrics to compare performance across models and settings.

---

