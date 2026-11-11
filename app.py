# Adult (18+) image studio on ZeroGPU, with several safety layers:
#   1) HF login required (every request belongs to an account)
#   2) text filter on the prompt + guard on the negative prompt (safety.py)
#   3) forced "adult" push in the prompt, forced "no minors" in the negative prompt (server side)
#   4) age check on the finished image (fail-closed)
#   5) logging of violations + automatic account ban after repeated hits
# MIT for more security for our children! 
import gc
import hashlib
import json
import os
import random
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import spaces
import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    EulerAncestralDiscreteScheduler,
    FluxPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
)
from huggingface_hub import CommitScheduler, hf_hub_download, scan_cache_dir, snapshot_download

import safety

# ----------------------------------------------------------------------------------------
# Configuration (Space "Settings -> Variables and secrets")
#   HF_TOKEN        secret  token of an account that accepted the FLUX.1-dev / SD3.5 licenses
#   LOG_REPO        variable e.g. "yourname/abuse-log" (private dataset is created automatically)
#   BAN_AFTER       variable strikes until an account is banned (default 3, use 1 for very strict)
#   LOG_PROMPT_TEXT variable "1" = also store the first 300 chars of BLOCKED prompts (default off)
# ----------------------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
LOG_REPO = os.environ.get("LOG_REPO")
BAN_AFTER = int(os.environ.get("BAN_AFTER", "3"))
LOG_PROMPT_TEXT = os.environ.get("LOG_PROMPT_TEXT", "0") == "1"
# REQUIRE_LOGIN=0 is a TEST switch only: without an account nobody can be banned reliably.
# Keep it at 1 (default) for a public Space.
REQUIRE_LOGIN = os.environ.get("REQUIRE_LOGIN", "1") != "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL = "SDXL - PornMaster Pro (default)"
MODELS = {
    DEFAULT_MODEL: dict(kind="sdxl", repo="votepurchase/pornmasterPro_realismILV4", steps=28, cfg=7.0),
    # Gated: accept the license on the model page with the account that owns HF_TOKEN.
    # License: FLUX.1 [dev] Non-Commercial. Fine for a private/personal Space; a PUBLIC service is
    # not covered by the non-commercial terms (needs a commercial license from Black Forest Labs).
    "FLUX.1-dev": dict(
        kind="flux", repo="black-forest-labs/FLUX.1-dev", steps=28, cfg=3.5,
        ignore=["flux1-dev.safetensors", "ae.safetensors"],  # skip the big single-file copies
    ),
    # Stable Diffusion 3.5 Medium is switched off on purpose: Stability AI's Acceptable Use Policy
    # appears to prohibit sexual-act content, so it does not fit an adult-content tool.
    # Re-enable only after checking the policy yourself: https://stability.ai/use-policy
    # "Stable Diffusion 3.5 Medium": dict(
    #     kind="sd3", repo="stabilityai/stable-diffusion-3.5-medium", steps=28, cfg=4.5,
    #     ignore=["sd3.5_medium.safetensors", "text_encoders/*"],
    # ),
    # More SDXL checkpoints work the same way, e.g.:
    # "SDXL - other model": dict(kind="sdxl", repo="user/some-sdxl-model", steps=28, cfg=7.0),
}

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1216

BLOCK_MSG = "This request can't be processed. Please keep it respectful and make sure every person is an adult."
BAN_MSG = "Your account has been suspended because of repeated violations."

# ----------------------------------------------------------------------------------------
# Logging and bans
# ----------------------------------------------------------------------------------------
STATE_DIR = Path("safety_logs")
STATE_DIR.mkdir(exist_ok=True)
EVENTS_FILE = STATE_DIR / f"events-{uuid.uuid4().hex[:8]}.jsonl"
BAN_FILE = STATE_DIR / "banned.json"
BAN_CATEGORIES = {"minor", "image_young", "negative_adult"}  # these count as strikes

scheduler = None
if LOG_REPO:
    scheduler = CommitScheduler(
        repo_id=LOG_REPO, repo_type="dataset", folder_path=STATE_DIR,
        every=5, private=True, token=HF_TOKEN,
    )
_lock = scheduler.lock if scheduler else threading.Lock()


def _load_bans():
    if LOG_REPO:
        try:
            p = hf_hub_download(LOG_REPO, "banned.json", repo_type="dataset", token=HF_TOKEN)
            BAN_FILE.write_text(Path(p).read_text())
        except Exception as e:  # first start: file does not exist yet
            print(f"[ban list] none loaded ({type(e).__name__})", flush=True)
    try:
        return set(json.loads(BAN_FILE.read_text()))
    except Exception:
        return set()


BANNED = _load_bans()
STRIKES = defaultdict(int)


def _log(event):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with _lock:
        with EVENTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _strike(user, category, prompt=None):
    event = {"user": user, "category": category}
    if prompt:
        event["prompt_sha"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        if LOG_PROMPT_TEXT:
            event["prompt"] = prompt[:300]
    _log(event)
    if category not in BAN_CATEGORIES:
        return
    STRIKES[user] += 1
    if STRIKES[user] >= BAN_AFTER and user not in BANNED:
        BANNED.add(user)
        with _lock:
            BAN_FILE.write_text(json.dumps(sorted(BANNED)))
        _log({"user": user, "category": "BANNED"})


def _identify(profile, request):
    """Account name if signed in. Without login: error, or (test mode only) a hash of the visitor address."""
    if profile is not None:
        return profile.username
    if REQUIRE_LOGIN:
        raise gr.Error("Please sign in with Hugging Face first (button at the top).")
    addr = ""
    if request is not None:
        addr = request.headers.get("x-forwarded-for") or (request.client.host if request.client else "") or ""
    return "anon-" + hashlib.sha256(addr.encode("utf-8")).hexdigest()[:10]


# ----------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------
def _make_compel(pipe):
    return Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
    )


def _build(name):
    cfg = MODELS[name]
    if cfg["kind"] == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg["repo"], torch_dtype=torch.float16, variant="fp16", use_safetensors=True, token=HF_TOKEN,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        return pipe, _make_compel(pipe)
    cls = FluxPipeline if cfg["kind"] == "flux" else StableDiffusion3Pipeline
    pipe = cls.from_pretrained(cfg["repo"], torch_dtype=torch.bfloat16, token=HF_TOKEN)
    pipe.to(device)
    return pipe, None


# The default model is loaded once at start (best for ZeroGPU). Other models are loaded on demand,
# and only ONE of them is kept at a time (memory + disk).
_default_pipe, _default_compel = _build(DEFAULT_MODEL)
_extra = {"name": None, "pipe": None, "compel": None}


def _get_pipeline(name):
    if name == DEFAULT_MODEL:
        return _default_pipe, _default_compel
    if _extra["name"] != name:
        _extra.update(name=None, pipe=None, compel=None)
        gc.collect()
        torch.cuda.empty_cache()
        pipe, compel = _build(name)
        _extra.update(name=name, pipe=pipe, compel=compel)
    return _extra["pipe"], _extra["compel"]


def _purge_cache(repo_id):
    """Free disk space of a model we no longer need."""
    try:
        info = scan_cache_dir()
        revs = [r.commit_hash for repo in info.repos if repo.repo_id == repo_id for r in repo.revisions]
        if revs:
            info.delete_revisions(*revs).execute()
    except Exception as e:
        print(f"[cache purge] {type(e).__name__}", flush=True)


def _prepare_model(name):
    """Runs on CPU BEFORE the GPU call, so downloading does not use GPU time."""
    if name == DEFAULT_MODEL or _extra["name"] == name:
        return
    previous = _extra["name"]
    if previous:
        _extra.update(name=None, pipe=None, compel=None)
        gc.collect()
        _purge_cache(MODELS[previous]["repo"])
    try:
        snapshot_download(MODELS[name]["repo"], token=HF_TOKEN, ignore_patterns=MODELS[name].get("ignore"))
    except Exception as e:
        print(f"[download failed] {name}: {type(e).__name__}", flush=True)
        raise gr.Error(
            "This model could not be loaded. (Owner: accept the license on the model page "
            "and set the HF_TOKEN secret.)"
        )


@spaces.GPU(duration=120)
def _gpu_generate(name, prompt, negative, seed, width, height, guidance, steps):
    pipe, compel = _get_pipeline(name)
    generator = torch.Generator(device=device).manual_seed(seed)
    kind = MODELS[name]["kind"]
    common = dict(
        guidance_scale=guidance, num_inference_steps=steps, width=width, height=height, generator=generator,
    )
    try:
        if kind == "sdxl":
            embeds = None
            try:  # long prompts (> 77 tokens) via Compel
                cond, pooled = compel(prompt)
                neg, neg_pooled = compel(negative)
                cond, neg = compel.pad_conditioning_tensors_to_same_length([cond, neg])
                embeds = dict(
                    prompt_embeds=cond, pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=neg, negative_pooled_prompt_embeds=neg_pooled,
                )
            except Exception as e:
                print(f"[compel failed] {type(e).__name__}, using standard prompt", flush=True)
            if embeds:
                return pipe(**embeds, **common).images[0]
            return pipe(prompt=prompt, negative_prompt=negative, **common).images[0]
        if kind == "flux":  # FLUX has no negative prompt
            return pipe(prompt=prompt, max_sequence_length=512, **common).images[0]
        return pipe(prompt=prompt, negative_prompt=negative, **common).images[0]  # SD3.5
    except RuntimeError as e:
        print(f"[generation error] {e}", flush=True)
        raise gr.Error("Generation failed (out of memory or GPU timeout). Try a smaller size or fewer steps.")


# ----------------------------------------------------------------------------------------
# Main request handler (runs on CPU; only the diffusion itself uses the GPU)
# ----------------------------------------------------------------------------------------
def generate(prompt, negative_prompt, model_name, seed, randomize_seed, width, height,
             guidance_scale, num_inference_steps,
             profile: Optional[gr.OAuthProfile] = None, request: gr.Request = None):
    user = _identify(profile, request)
    if user in BANNED:
        raise gr.Error(BAN_MSG)
    if model_name not in MODELS:
        raise gr.Error("Unknown model.")

    prompt = (prompt or "").strip()
    negative_prompt = (negative_prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    # Layer 2: text filter + negative prompt guard
    try:
        safety.ensure_safe(prompt)
        safety.ensure_safe_negative(negative_prompt)
    except safety.Blocked as e:
        _strike(user, str(e), prompt)
        raise gr.Error(BAN_MSG if user in BANNED else BLOCK_MSG)

    # Layer 3: forced hardening (added server side, the user cannot remove it)
    prompt, negative_prompt = safety.harden_prompts(prompt, negative_prompt)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)
    width = int(min(max(width, 256), MAX_IMAGE_SIZE)) // 16 * 16
    height = int(min(max(height, 256), MAX_IMAGE_SIZE)) // 16 * 16
    steps = int(min(max(num_inference_steps, 1), 50))
    guidance = float(min(max(guidance_scale, 0.0), 20.0))

    _prepare_model(model_name)
    image = _gpu_generate(model_name, prompt, negative_prompt, seed, width, height, guidance, steps)

    # Layer 4: age check on the result (fail-closed: if the checker breaks, nothing is shown)
    if safety.image_looks_young(image):
        _strike(user, "image_young", prompt)
        raise gr.Error(BAN_MSG if user in BANNED else BLOCK_MSG)
    return image


def on_model_change(name):
    m = MODELS[name]
    return (
        gr.update(value=m["steps"]),
        gr.update(value=m["cfg"]),
        gr.update(visible=m["kind"] != "flux"),
    )


def on_load(profile: Optional[gr.OAuthProfile] = None):
    """Shows who is signed in and keeps Run disabled until sign-in (unless test mode)."""
    if profile is not None:
        return f"Signed in as **{profile.username}**", gr.update(interactive=True)
    if REQUIRE_LOGIN:
        return "Not signed in. Please sign in with Hugging Face to generate images.", gr.update(interactive=False)
    return "Test mode: not signed in (anonymous).", gr.update(interactive=True)


# Load the age checker at start so problems show up in the logs immediately.
try:
    safety._load_age_clf()
except Exception as e:
    print(f"[age checker not available] {type(e).__name__} - all images will be blocked (fail-closed)", flush=True)

# ----------------------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------------------
css = """
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            "### 18+ only\n"
            "Sign in with Hugging Face. Every depicted person must be an adult. Prompts and images are "
            "checked automatically; violations are logged and lead to a ban."
        )
        gr.LoginButton()
        status = gr.Markdown("Checking sign-in...")

        model = gr.Dropdown(label="Model", choices=list(MODELS), value=DEFAULT_MODEL)

        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                show_label=False,
                lines=2,
                max_lines=8,
                placeholder="Enter your prompt (long prompts are supported)",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):
            negative_prompt = gr.Text(
                label="Negative prompt (not used by FLUX)",
                max_lines=1,
                placeholder="Enter a negative prompt",
                value="(low quality, worst quality:1.2), very displeasing, 3d, watermark, signature, ugly, poorly drawn",
            )
            seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row():
                width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=1024)
                height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=1024)
            with gr.Row():
                guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=20.0, step=0.1, value=7)
                num_inference_steps = gr.Slider(label="Number of inference steps", minimum=1, maximum=50, step=1, value=28)

    demo.load(on_load, inputs=None, outputs=[status, run_button])
    model.change(on_model_change, inputs=model, outputs=[num_inference_steps, guidance_scale, negative_prompt])

    run_button.click(
        fn=generate,
        inputs=[prompt, negative_prompt, model, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result],
    )

# ssr_mode=False: the Node proxy of SSR mode can interfere with the OAuth login cookies.
# css goes into launch() since Gradio 6.
demo.queue(max_size=20).launch(css=css, ssr_mode=False)
