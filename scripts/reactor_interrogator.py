import os
import importlib.util
from scripts.reactor_logger import logger

INTERROGATOR = None

# Try to load the sd-Img2img-batch-interrogator plugin
PLUGIN_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "sd-Img2img-batch-interrogator",
    "scripts",
    "sd_tag_batch.py",
)
PLUGIN_PATH = os.path.abspath(PLUGIN_PATH)

if os.path.exists(PLUGIN_PATH):
    try:
        spec = importlib.util.spec_from_file_location("sd_tag_batch", PLUGIN_PATH)
        sd_tag_batch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sd_tag_batch)
        if hasattr(sd_tag_batch, "interrogation_processor"):
            INTERROGATOR = sd_tag_batch.interrogation_processor
            logger.info("[ReActor] Interrogator plugin loaded")
        else:
            logger.error("[ReActor] Interrogator plugin missing interrogation_processor")
    except Exception as e:
        logger.error(f"[ReActor] Failed to load interrogator plugin: {e}")
else:
    logger.info("[ReActor] Interrogator plugin not found")

def interrogate_face(p, image):
    """Run interrogation on a PIL Image and return the result string."""
    if INTERROGATOR is None:
        return ""
    try:
        result = INTERROGATOR.process_batch(
            p=p,
            tag_batch_enabled=True,
            model_selection=['CLIP (Native)'],
            debug_mode=False,
            in_front='Prepend to prompt',
            insert_target='Prompt',
            insert_index=0,
            prompt_weight_mode=False,
            prompt_weight=0.5,
            reverse_mode=False,
            exaggeration_mode=False,
            prompt_output=False,
            use_positive_filter=False,
            use_negative_filter=False,
            use_custom_filter=False,
            custom_filter='',
            use_custom_replace=False,
            custom_replace_find='',
            custom_replace_replacements='',
            clip_ext_model=[],
            clip_ext_mode='best',
            wd_ext_model=[],
            wd_threshold=0.35,
            wd_underscore_fix=True,
            wd_append_ratings=False,
            wd_ratings=0.5,
            wd_keep_tags='',
            unload_clip_models_afterwords=True,
            unload_wd_models_afterwords=True,
            no_puncuation_mode=False,
            batch_number=0,
            prompts=[p.prompt],
            seeds=[p.seed],
            subseeds=[p.subseed],
            prompt_override="",
            image_override=image,
            update_p=False,
        )
        return (result or "").strip().rstrip(',')
    except Exception as e:
        logger.error(f"[ReActor] Interrogation failed: {e}")
        return ""
