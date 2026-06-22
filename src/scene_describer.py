"""
Scene Describer using Microsoft Florence-2

This module uses the Florence-2 vision-language model to generate
natural language descriptions of scenes, objects, and their relationships.
"""

import cv2
import numpy as np
from PIL import Image
import torch


class SceneDescriber:
    """Generates scene descriptions using Florence-2."""

    def __init__(self, model_name="microsoft/Florence-2-base-ft", device=None):
        """
        Initialize the Florence-2 model.

        Args:
            model_name: Hugging Face model identifier
            device: torch device (auto-detected if None)
        """
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"Loading Florence-2 model: {model_name}")
        print("(This may take a minute on first run as the model downloads)...")

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
            else:
                device = "cpu"

        self.device = device
        print(f"Using device: {device}")

        # Load model and processor
        # Use float32 on CPU for compatibility, bfloat16 on GPU
        dtype = torch.float32 if device == "cpu" else torch.bfloat16

        # Florence-2 uses custom code (trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("Florence-2 loaded successfully!")

    def describe(self, frame, task="<MORE_DETAILED_CAPTION>"):
        """
        Generate a description of the scene in the frame.

        Args:
            frame: OpenCV BGR image (numpy array)
            task: Florence-2 task prompt
                  - "<CAPTION>" - short caption
                  - "<DETAILED_CAPTION>" - detailed caption
                  - "<MORE_DETAILED_CAPTION>" - very detailed caption
                  - "<OD>" - object detection (returns labels + boxes)
                  - "<DENSE_REGION_CAPTION>" - regions with captions

        Returns:
            dict: Parsed output from Florence-2
        """
        # Convert OpenCV BGR to PIL RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Prepare inputs
        inputs = self.processor(
            text=task,
            images=pil_image,
            return_tensors="pt"
        ).to(self.device, torch.float32 if self.device == "cpu" else torch.bfloat16)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )

        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Post-process
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(pil_image.width, pil_image.height)
        )
        return parsed

    def get_caption(self, frame, detailed=True):
        """Get a simple caption for the scene."""
        task = "<MORE_DETAILED_CAPTION>" if detailed else "<CAPTION>"
        result = self.describe(frame, task=task)
        # The result is a dict like {"<MORE_DETAILED_CAPTION>": "text"}
        if task in result:
            return result[task]
        # Fallback: find any caption key
        for key, value in result.items():
            if "caption" in key.lower():
                return value
        return str(result)


# Quick test
if __name__ == "__main__":
    import sys

    print("Testing Florence-2 Scene Describer...")
    describer = SceneDescriber()

    # Test with a simple image
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"Could not load image: {sys.argv[1]}")
            sys.exit(1)
    else:
        # Create a test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Test Image", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    print("\nGenerating detailed caption...")
    caption = describer.get_caption(img, detailed=True)
    print(f"Caption: {caption}")