import importlib
from typing import Optional
from utils.common import process_csv
import utils.perturb as perturb

def load_model_module(name: str):
    """Dynamically import models.<name>"""
    return importlib.import_module(f"models.{name}")

def run_model(
    name: str,
    input_csv: str,
    output_csv: Optional[str] = None,
    image_base: str = "data/imgs/",
    no_image: bool = False,
    perturb: Optional[str] = None,
):
    mod = load_model_module(name)
    # determine output filename if not provided
    if output_csv is None:
        model_label = getattr(mod, "MODEL_NAME", name).split("/")[-1]
        suffix = "no_image" if no_image else (perturb or "no_perturb")
        suffix_safe = str(suffix).replace(":", "_").replace("=", "-").replace(",", "_").replace(" ", "")
        output_csv = f"results/{model_label}_{suffix_safe}_results.csv"

    # Configure module-level flags if module supports them
    if hasattr(mod, "INCLUDE_IMAGE"):
        mod.INCLUDE_IMAGE = not no_image
    if perturb:
        # use get_perturb if available
        try:
            mod.PERTURB_FN = perturb.get_perturb(*perturb.split(":", 1)) if hasattr(perturb, "get_perturb") else getattr(perturb, perturb, None)
        except Exception:
            mod.PERTURB_FN = perturb

        mod.PERTURB_NAME = perturb
    else:
        mod.PERTURB_FN = None
        mod.PERTURB_NAME = "no_perturb" if not no_image else "no_image"

    # call the generic CSV processor with the model's ask function
    process_csv(
        input_csv,
        output_csv,
        image_base,
        getattr(mod, "CUSTOM_PROMPT"),
        ask_fn=mod.ask_image_question,
    )
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Module name under models (e.g., qwen, molmo)")
    parser.add_argument("--input", default="data/typed_questions.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--image-base", default="data/imgs/")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--perturb", default=None)
    args = parser.parse_args()
    run_model(args.model, args.input, args.output, args.image_base, args.no_image, args.perturb)