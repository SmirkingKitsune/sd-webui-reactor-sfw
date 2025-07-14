import gradio as gr
import importlib.util
import os
import sys

PLUGIN_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "sd-Img2img-batch-interrogator",
        "scripts",
        "sd_tag_batch.py",
    )
)


def show(is_img2img: bool):
    """Show the sd-Img2img-batch-interrogator UI inside ReActor."""
    with gr.Tab("Interrogation"):
        if not os.path.exists(PLUGIN_PATH):
            gr.Markdown(
                f"**Img2img Batch Interrogator plugin not found.**\nExpected at `{PLUGIN_PATH}`"
            )
            return []

        module_name = "sd_tag_batch"
        if module_name in sys.modules:
            sd_tag_batch = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, PLUGIN_PATH)
            sd_tag_batch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sd_tag_batch)
            sys.modules[module_name] = sd_tag_batch

        if hasattr(sd_tag_batch, "interrogation_processor"):
            return sd_tag_batch.interrogation_processor.ui(is_img2img, skip_check=True)
        gr.Markdown("**Interrogator plugin found but missing `interrogation_processor`.**")
        return []
