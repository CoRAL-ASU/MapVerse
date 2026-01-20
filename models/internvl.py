import os
import argparse
from typing import Callable, Optional
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cairosvg
import csv
import io
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from utils.common import process_csv, load_image_safely
import utils.perturb as perturb


MODEL_NAME = "OpenGVLab/InternVL3-1B"
CUSTOM_PROMPT = """
You are an AI Agent with specialised knowledge in reading and understanding map data. Analyze the following map and using information from the steps and examples given below, answer the question.

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

Assuming we are talking about a map with election results for USA. This map contains the voter breakdown across the United States, including the number of votes cast and the winning party in each state. Some examples of questions and their answers are as follows:

Question: Count the number of states on the west coast where Democrats won.
Answer: 3

Question: Based on the information given in the map, who won the election, Democrats or Republicans?
Answer: Democrats

Question: Based on the information given in the map, if both Democrats and Republicans win 25 states each, do we have more blue states or red states?
Answer: neither

Question: List the top 4 states in terms of seats where the Republicans won
Answer: Texas, Georgia, Missouri, Tennessee

Question: Rank these states in ascending order of seats - Kansas, South Carolina, Nebraska, Oklahoma, Colorado, Wisconsin
Answer: Nebraska, Kansas, Oklahoma, South Carolina, Colorado, Wisconsin

Question: Based on reasoning, Answer the following:
     Montana : Wyoming :: North Dakota : ?
Answer: South Dakota

Now, answer the Question below based on the information, instruction and examples above:
"""

# runtime flags (modified in __main__)
INCLUDE_IMAGE: bool = True
PERTURB_FN: Optional[Callable[[Image.Image], Image.Image]] = None
PERTURB_NAME: Optional[str] = None

INPUT_CSV = "data/typed_questions.csv"
OUTPUT_CSV = f"results/<OUTPUT_FILE_NAME>.csv"
IMAGE_BASE_FOLDER = "data/imgs/"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---- model + processors ----
def load_model_and_processor(model_name=MODEL_NAME):
    print("Loading InternVL model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    print(f"Model loaded on: {model.device}")
    return model, tokenizer, image_processor

model, tokenizer, image_processor = load_model_and_processor()

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    # find closest
    best = min(target_ratios, key=lambda r: abs(aspect_ratio - (r[0] / r[1])))
    target_width = image_size * best[0]
    target_height = image_size * best[1]
    blocks = best[0] * best[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_to_tensor(image_path, input_size=448, max_num=12):
    image = load_image_safely(image_path)
    if image is None:
        return None
    transform = build_transform(input_size=input_size)
    imgs = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(im) for im in imgs]
    pixel_values = torch.stack(pixel_values)  # shape (n, C, H, W)
    return pixel_values

def _extract_response_text(response):
    # Various model APIs return different shapes; normalize to a string
    if isinstance(response, str):
        return response.strip()
    try:
        if hasattr(response, "text"):
            return response.text.strip()
        if isinstance(response, dict):
            for k in ("text", "answer", "response"):
                if k in response:
                    return str(response[k]).strip()
    except Exception:
        pass
    return str(response).strip()

def ask_image_question(image_path: str, question: str, custom_prompt: str) -> str:
    """
    Handles optional image / perturb application and calls model.chat (InternVL).
    Returns the string answer.
    """
    final_prompt = f"{custom_prompt}\n<image>\nQuestion:{question}"
    pixel_values = None
    if INCLUDE_IMAGE:
        # load original or perturbed image (keep in-memory)
        pil_img = load_image_safely(image_path)
        if pil_img is None:
            print(f"⚠️ Could not load image {image_path}; proceeding text-only.")
            pixel_values = None
        else:
            if PERTURB_FN:
                try:
                    pil_img = PERTURB_FN(pil_img)
                except Exception as e:
                    print(f"⚠️ Perturb '{getattr(PERTURB_FN,'__name__',str(PERTURB_FN))}' failed: {e}; using original image.")
            pixel_values = load_image_to_tensor(image_path)  # uses load_image_safely internally
            if pixel_values is None:
                print("⚠️ Failed to convert image to tensor; proceeding text-only.")
    else:
        final_prompt = f"{custom_prompt}\nQuestion:{question}"

    generation_config = dict(max_new_tokens=128, do_sample=False)
    try:
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values.to(model.device, dtype=model.dtype) if pixel_values is not None else None,
            question=final_prompt,
            generation_config=generation_config
        )
    except Exception as e:
        return f"ERROR: {e}"

    return _extract_response_text(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run aya-vision (refactored) with optional image/perturbation")
    parser.add_argument("--input", default=INPUT_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--image-base", default=IMAGE_BASE_FOLDER)
    parser.add_argument("--no-image", action="store_true", help="Do not send images to the model (text-only prompts).")
    parser.add_argument("--perturb", default=None, help=f"Name of perturbation from perturb. Options: {', '.join(getattr(perturb, '__all__', []))}")
    args = parser.parse_args()

    # configure runtime flags
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
        # ensure metadata reflects the no-image run
        PERTURB_NAME = "no_image"
        
    if args.output == OUTPUT_CSV:
        output_file_name = MODEL_NAME.split("/")[-1]
        if not INCLUDE_IMAGE:
            output_file_name += "_no_image"
        if PERTURB_NAME:
            output_file_name += f"_{PERTURB_NAME}"
        args.output = f"results/{output_file_name}.csv"

    process_csv(args.input, args.output, args.image_base, CUSTOM_PROMPT, ask_fn=ask_image_question)