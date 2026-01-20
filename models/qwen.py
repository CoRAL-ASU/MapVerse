import os
import argparse
from typing import Callable, Optional
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.common import process_csv, load_image_safely
import utils.perturb as perturb

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
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


Question: Count the number states on the west coast where Democrats won.
Answer: 3


Questions: Based on the information given in the map, who won the election, Democrats or Republicans?
Answer: Democrats

Questions: Based on the information given in the map, if both Democrats and Republicans win 25 states each, do we have more blue states or red states?
Answer: neither

Question: List the top 4 states in terms of seats where the republicans won
Answer: Texas, Georgia, Missouri, Tenessee

Question: Rank these states in ascending order of seats - kansas, south carolina, nebraska, oklahoma, colorado, wisconsin
Answer: nebraska, kansas, oklahoma, south carolina, colorado, wisconsin

Question: Based on reasoning, Answer the following:
     Montana : Wyoming :: North Dakota : ?
Answer: South Dakota

Now, Answer the Question below based on the information, instruction and examples above:

"""

# runtime flags (modified in __main__)
INCLUDE_IMAGE: bool = True
PERTURB_FN: Optional[Callable[[Image.Image], Image.Image]] = None
PERTURB_NAME: Optional[str] = None

INPUT_CSV = "data/typed_questions.csv"
OUTPUT_CSV = f"results/<OUTPUT_FILE_NAME>.csv"
IMAGE_BASE_FOLDER = "data/imgs/"

# ---- Load model & processor ----
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
print(f"Model loaded on: {model.device}\n")

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

def ask_image_question(image_path: str, question: str, custom_prompt: str) -> str:
    # Handle no-image (text-only) mode
    if not INCLUDE_IMAGE:
        messages = [
            {"role": "system", "content": custom_prompt},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[], videos=[], padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        # Trim off prompt tokens and decode
        outputs_trimmed = [out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(outputs_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text[0].strip()

    # Load and optionally perturb the image
    image = load_image_safely(image_path)
    if image is None:
        return "Error loading image"
    if PERTURB_FN:
        try:
            image = PERTURB_FN(image)
        except Exception as e:
            print(f"⚠️ Perturbation '{getattr(PERTURB_FN,'__name__',str(PERTURB_FN))}' failed: {e}; using original image.")

    # Build chat template (system + user with image)
    messages = [
        {"role": "system", "content": custom_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move to model device & dtype
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    # Generate and decode
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0].strip()


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