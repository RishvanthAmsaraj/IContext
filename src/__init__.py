"""IContext: Context-aware computer vision pipeline.

This package combines YOLOv8 object detection, MediaPipe person analysis,
and the Florence-2 vision-language model to generate rich contextual
descriptions of scenes.

Modules:
    i_context:        Main pipeline orchestrating all components.
    context_analyzer: Rule-based context inference from object lists.
    scene_describer:  Florence-2 wrapper for natural-language descriptions.
    caption_parser:   Extract object nouns from VLM-generated captions.
    detect_objects:   Simplified object detection (legacy entry point).
"""

__version__ = "0.1.0"
__author__ = "Rishvanth Amsaraj"