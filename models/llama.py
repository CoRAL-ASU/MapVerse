import os
import argparse
from typing import Callable, Optional
import torch
from transformers import MllamaProcessor, MllamaForConditionalGeneration
from utils.common import process_csv, load_image_safely
import utils.perturb as perturb
from PIL import Image


MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"   # Vision-language model
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


def load_model_and_processor(model_name=MODEL_NAME):
    print("Loading model and processor...")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    ).eval()
    processor = MllamaProcessor.from_pretrained(model_name, use_fast=True)
    print(f"Model loaded on: {model.device}")
    return model, processor

model, processor = load_model_and_processor()

def ask_image_question(image_path: str, question: str, custom_prompt: str) -> str:
    """
    Uses processor.apply_chat_template + model.generate.
    Respects global INCLUDE_IMAGE and PERTURB_FN.
    """
    image = None
    if INCLUDE_IMAGE:
        image = load_image_safely(image_path)
        if image is None:
            print(f"⚠️ Could not load image {image_path}; proceeding without image.")
        else:
            if PERTURB_FN:
                try:
                    image = PERTURB_FN(image)
                except Exception as e:
                    print(f"⚠️ Perturbation '{getattr(PERTURB_FN, '__name__', str(PERTURB_FN))}' failed: {e}; using original image.")

    # Build messages: include image only when present
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image} if image is not None else None,
            {"type": "text", "text": f"{custom_prompt}\nQuestion:{question}"},
        ]}
    ]
    # Remove None entries
    for m in messages:
        m["content"] = [c for c in m["content"] if c is not None]

    inputs = processor.apply_chat_template(
        messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    gen_tokens = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        use_cache=False
    )

    # decode model output tokens beyond the input ids
    output_text = processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return output_text.strip()

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