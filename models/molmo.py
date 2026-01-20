# MapVerse/molmo_qa_refactored.py
import os
import argparse
from typing import Callable, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, GenerationConfig
from utils.common import process_csv, load_image_safely
import utils.perturb as perturb
from PIL import Image
import io
import cairosvg

MODEL_NAME = "allenai/Molmo-7B-D-0924"
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

def load_model_and_processor(model_name=MODEL_NAME):
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    print(f"Model loaded on: {model.device}")
    return model, tokenizer, processor

model, tokenizer, processor = load_model_and_processor()

def ask_image_question(image_path: str, question: str, custom_prompt: str) -> str:
    # if images are disabled, send text-only prompt
    if not INCLUDE_IMAGE:
        try:
            inputs = processor.process(text=f"{custom_prompt}Question: {question}\nAnswer: ")
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=128, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer,
                use_cache=False
            )
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            return f"ERROR: {e}"

    # include image path / load image
    image = load_image_safely(image_path)
    if image is None:
        return "Error loading image"

    # apply perturbation if requested
    if PERTURB_FN:
        try:
            image = PERTURB_FN(image)
        except Exception as e:
            print(f"⚠️ Perturb '{getattr(PERTURB_FN, '__name__', str(PERTURB_FN))}' failed: {e}; using original image.")

    try:
        inputs = processor.process(images=[image], text=f"{custom_prompt}Question: {question}\nAnswer: ")
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=128, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
            use_cache=False
        )
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        return f"ERROR: {e}"

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