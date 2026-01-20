# MapVerse/mapqa_common.py
import os
import io
import csv
import base64
from typing import Callable, Optional, Set, Tuple
from PIL import Image, ImageSequence
import pandas as pd
from tqdm import tqdm

try:
    import cairosvg
    _HAS_CAIROSVG = True
except Exception:
    _HAS_CAIROSVG = False

def find_image_recursive(base_folder: str, image_name: str) -> Optional[str]:
    """Recursively find image by filename in base_folder. Returns full path or None."""
    for root, _, files in os.walk(base_folder):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

def load_image_safely(image_path: str) -> Optional[Image.Image]:
    """Load image safely. Supports PNG, JPEG, GIF (first frame), WEBP and SVG (if cairosvg available)."""
    try:
        lower = image_path.lower()
        if lower.endswith(".svg"):
            if not _HAS_CAIROSVG:
                raise RuntimeError("cairosvg is required to load SVGs. Install `cairosvg` or skip SVGs.")
            png_bytes = cairosvg.svg2png(url=image_path)
            img = Image.open(io.BytesIO(png_bytes))
            return img.convert("RGB")
        else:
            img = Image.open(image_path)
            # For animated/sequence images (GIF), pick first frame
            if getattr(img, "is_animated", False):
                img.seek(0)
            return img.convert("RGB")
    except Exception as e:
        print(f"❌ Failed to load image {image_path}: {e}")
        return None

def load_already_answered(output_csv: str) -> Set[Tuple[str, str]]:
    """Return set of (image_name, question) pairs already present in output CSV."""
    answered = set()
    try:
        df = pd.read_csv(output_csv)
        # expected columns: image_name, question, llm_answer
        for _, row in df.iterrows():
            img = str(row.get("image_name", "")).strip()
            q = str(row.get("question", "")).strip()
            if img and q:
                answered.add((img, q))
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Warning: could not read {output_csv}: {e}")
    return answered

def append_result_row(output_csv: str, image_name: str, question: str, llm_answer: str):
    """Append a single result row to CSV (creates file with header if missing)."""
    header = ["image_name", "question", "llm_answer"]
    exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow({"image_name": image_name, "question": question, "llm_answer": llm_answer})

def process_csv(
    input_csv: str,
    output_csv: str,
    base_folder: str,
    custom_prompt: str,
    ask_fn: Callable[[str, str, str], str],
    image_field_names=None,
    question_field_names=None,
    show_progress=True,
):
    """
    Generic CSV processor:
      - input_csv: path to CSV with (image, question) rows
      - ask_fn(image_path, question, custom_prompt) -> answer
      - base_folder: root to search images (if image name only)
    """
    if image_field_names is None:
        image_field_names = ["image", "image_name", "img", "image_path", "file"]
    if question_field_names is None:
        question_field_names = ["question", "question_text", "query", "q"]

    answered = load_already_answered(output_csv)
    df = pd.read_csv(input_csv)

    rows = list(df.to_dict(orient="records"))
    iterator = tqdm(rows, desc="Processing CSV") if show_progress else rows

    for row in iterator:
        # find image and question fields
        image_name = None
        question_text = None
        for k in image_field_names:
            if k in row and not pd.isna(row[k]):
                image_name = str(row[k]).strip()
                break
        for k in question_field_names:
            if k in row and not pd.isna(row[k]):
                question_text = str(row[k]).strip()
                break
        if not image_name or not question_text:
            continue

        key = (image_name, question_text)
        if key in answered:
            continue

        # Resolve image path
        image_path = image_name
        if not os.path.isabs(image_name):
            # if it's just a filename, try to find it under base_folder
            if os.path.exists(os.path.join(base_folder, image_name)):
                image_path = os.path.join(base_folder, image_name)
            else:
                found = find_image_recursive(base_folder, image_name)
                if found:
                    image_path = found

        try:
            answer = ask_fn(image_path, question_text, custom_prompt)
        except Exception as e:
            print(f"Error asking question for {image_name}: {e}")
            answer = f"ERROR: {e}"

        append_result_row(output_csv, image_name, question_text, answer)
        answered.add(key)