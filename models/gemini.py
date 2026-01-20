# MapVerse/gemini_qa_refactored.py
import os
import csv
import time
import base64
import mimetypes
import argparse
import google.generativeai as genai
from typing import Callable, Optional
from utils.common import process_csv, load_image_safely
import utils.perturb as perturb
from PIL import Image
import io
import cairosvg

CUSTOM_PROMPT = """
You are an AI Agent with specilaised knowledge in reading and understanding map data. Analyze the following map and using information from the steps and examples given below, answer the question.

Steps to follow:

Identify Map-Related Elements in the Question
 Carefully understand the question to determine the key geographical features, locations, or spatial characteristics it refers to.


Locate the Identified Elements on the Map
 Find and observe these features or entities on the map provided. Pay attention to patterns, distributions, directions, scales, or any visual cues.


Apply Logical Reasoning
 Use spatial reasoning and contextual clues from the map to draw connections between the question and the observed map features.


Formulate a Concise Answer
 Based on your reasoning, arrive at a clear and accurate answer. Return only a word or phrase, as required—no explanation is needed. 
 If adequate data is not present, give answer as "no data". 
 If you have all the data and there is no answer to the question, give answer "none". If it is a counting problem, give answer 0. 
 If you have all the data and it is not possible to answer the question, give answer "not possible".

Assuming we are talking about a map with election results for USA. This  map contains  the voter breakdown across the United States, including the number of votes cast and the winning party in each state. Some examples of questions and there answers are as follows:


Question 1: Count the number states on the west coast where Democrats won.
Answer 1: 3


Question 2: Based on the information given in the map, who won the election, Democrats or Republicans?
Answer 2: Democrats

Question 3: Based on the information given in the map, if both Democrats and Republicans win 25 states each, do we have more blue states or red states?
Answer 3: neither

Question 4: List the top 4 states in terms of seats where the republicans won
Answer 4: Texas, Georgia, Missouri, Tenessee

Question 5: Rank these states in ascending order of seats - kansas, south carolina, nebraska, oklahoma, colorado, wisconsin
Answer 5: nebraska, kansas, oklahoma, south carolina, colorado, wisconsin

Question 6: Based on reasoning, Answer the following:
     Montana : Wyoming :: North Dakota : ?
Answer 6: South Dakota

Now, Answer the Question below based on the information, instruction and examples above.
As shown previously, your answers should be in the format Answer 1, Answer 2, etc. based on the number of questions shown below.

"""

# === CONFIG ===
MODEL_NAME = "models/gemini-1.5-flash"
INPUT_CSV = "data/typed_questions.csv"
OUTPUT_CSV = f"results/{MODEL_NAME.split('/')[-1]}_results.csv"
IMAGE_BASE_FOLDER = "data/imgs/"


API_KEYS = [
"<GEMINI_API_KEY_1>",   
"<GEMINI_API_KEY_2>",
]

CHUNK_SIZE = 1
SLEEP_TIME = 15

# runtime flags
INCLUDE_IMAGE: bool = True
PERTURB_FN: Optional[Callable[[Image.Image], Image.Image]] = None
PERTURB_NAME: Optional[str] = None

# === Gemini Auth/rotation ===
current_key_index = 0

def switch_key():
    global current_key_index
    if current_key_index >= len(API_KEYS):
        raise RuntimeError("All API keys exhausted for today.")
    genai.configure(api_key=API_KEYS[current_key_index])
    print(f"\n🔑 Using API key ending with ...{API_KEYS[current_key_index][-4:]}")
    current_key_index += 1

switch_key()

# === Helpers ===
def load_existing_pairs(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        return set((row[0], row[1]) for row in reader)

def get_image_path(image_name, image_root=IMAGE_BASE_FOLDER):
    for root, _, files in os.walk(image_root):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

def encode_image(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        raise ValueError(f"Could not detect MIME type for: {image_path}")
    ext = os.path.splitext(image_path)[-1].lower()
    if ext == ".svg":
        png_bytes = cairosvg.svg2png(url=image_path)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    else:
        with Image.open(image_path) as img:
            if img.format == 'GIF':
                img = img.convert("RGB")
            else:
                img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    image_bytes = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {"mime_type": "image/jpeg", "data": image_bytes}

def call_gemini_with_image(image_path, question):
    model = genai.GenerativeModel(MODEL_NAME)
    image_data = encode_image(image_path)
    prompt = CUSTOM_PROMPT + "\nQuestion: " + question + "\nAnswer:"
    response = model.generate_content([
        {"text": prompt},
        {"inline_data": image_data}
    ])
    return response.text

def call_gemini_text_only(question):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = CUSTOM_PROMPT + "\nQuestion: " + question + "\nAnswer:"
    response = model.generate_content([{"text": prompt}])
    return response.text

def ask_image_question(image_path: str, question: str, custom_prompt: str) -> str:
    # Use utils.common.load_image_safely to get PIL image (for applying perturbations)
    image = None
    if INCLUDE_IMAGE:
        image = load_image_safely(image_path)
        if image is None:
            print(f"⚠️ Could not load image {image_path}; falling back to text-only.")
            return call_gemini_text_only(question)
        # apply perturbation if required
        if PERTURB_FN:
            try:
                image = PERTURB_FN(image)
                # write out a temp JPEG in memory and call encode on bytes
                temp_buf = io.BytesIO()
                image.save(temp_buf, format='JPEG')
                temp_buf.seek(0)
                # save temp bytes to a temp file-like object for encode_image convenience
                tmp_path = image_path  # keep same name; encode_image expects a path, so write to temp file
                with open(tmp_path + ".tmp.jpg", "wb") as tf:
                    tf.write(temp_buf.getvalue())
                try:
                    return call_gemini_with_image(tmp_path + ".tmp.jpg", question)
                finally:
                    try:
                        os.remove(tmp_path + ".tmp.jpg")
                    except Exception:
                        pass
            except Exception as e:
                print(f"⚠️ Perturbation '{getattr(PERTURB_FN,'__name__',str(PERTURB_FN))}' failed: {e}; proceeding unperturbed.")
        # no perturb or perturb failed => call with original image path
        return call_gemini_with_image(image_path, question)
    else:
        return call_gemini_text_only(question)

def _validate_perturb(arg: str):
    name, _, spec = arg.partition(":")
    if hasattr(perturb, "get_perturb"):
        try:
            return perturb.get_perturb(name, spec if spec else None)
        except Exception as e:
            raise ValueError(f"Error building perturb wrapper: {e}")
    else:
        if name not in getattr(perturb, "__all__", []):
            raise ValueError(f"Unknown perturb '{name}'. Available: {getattr(perturb, '__all__', [])}")
        return getattr(perturb, name)

# ---- CLI / Run ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemini (refactored) with optional image/perturbation")
    parser.add_argument("--input", default=INPUT_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--image-base", default=IMAGE_BASE_FOLDER)
    parser.add_argument("--no-image", action="store_true", help="Do not send images to the model (text-only prompts).")
    parser.add_argument("--perturb", default=None, help=f"Name of perturbation from utils.perturb. Options: {', '.join(getattr(perturb, '__all__', []))}")
    args = parser.parse_args()

    INCLUDE_IMAGE = not args.no_image
    if args.perturb:
        try:
            PERTURB_FN = _validate_perturb(args.perturb)
            PERTURB_NAME = args.perturb
            print(f"✅ Using perturbation: {args.perturb}")
        except ValueError as e:
            print(e)
            raise SystemExit(1)
    if not INCLUDE_IMAGE:
        PERTURB_NAME = "no_image"

    # Compute output filename if default used
    if args.output == OUTPUT_CSV:
        model_label = MODEL_NAME.split("/")[-1]
        suffix = "no_image" if not INCLUDE_IMAGE else (PERTURB_NAME or "no_perturb")
        safe_suffix = str(suffix).replace(":", "_").replace("=", "-").replace(",", "_").replace(" ", "")
        args.output = f"results/{model_label}_{safe_suffix}_results.csv"
        print(f"🔖 Output file: {args.output}")

    process_csv(
        args.input,
        args.output,
        args.image_base,
        CUSTOM_PROMPT,
        ask_fn=ask_image_question
    )