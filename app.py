"""
app.py
──────
Gradio Web Interface for the VLM UI Generator.

Tabs
────
  🎨 Generate  – Upload sketch → get polished UI design
  🏋️ Train     – Launch / monitor training with live loss curves
  🖼️ Gallery   – Browse all past generations
  ℹ️  About    – Architecture overview

Run locally:
    python app.py

Run in Google Colab (shares a public URL):
    python app.py --share
"""

import argparse
import json
import os
import random
import time
import logging
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── Optional heavy imports (graceful degradation) ────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Paths ────────────────────────────────────────────────────────────────────
GALLERY_DIR  = Path("outputs/generated")
LOG_CSV      = Path("outputs/logs/train_log.csv")
CKPT_DIR     = Path("checkpoints")
GALLERY_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Model loader (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────────

_engine = None

def get_engine(checkpoint: str):
    global _engine
    if _engine is not None:
        return _engine
    try:
        from omegaconf import OmegaConf
        from inference.inference_engine import InferenceEngine
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        _engine = InferenceEngine.from_checkpoint(
            checkpoint_path   = checkpoint,
            model_config_path = "config/model_config.yaml",
            device            = device,
        )
        logger.info(f"Model loaded on {device}")
    except Exception as e:
        logger.warning(f"Could not load model: {e}. Running in DEMO mode.")
        _engine = None
    return _engine


# ─────────────────────────────────────────────────────────────────────────────
#  Demo-mode placeholder generator
# ─────────────────────────────────────────────────────────────────────────────

def _demo_generate(sketch: Image.Image, prompt: str, seed: int) -> Image.Image:
    """Generate a stylised placeholder when no model is loaded."""
    rng = random.Random(seed)
    W, H = 512, 512

    # Background gradient
    img  = Image.new("RGB", (W, H), (15, 15, 25))
    draw = ImageDraw.Draw(img)

    # Colour palette from prompt keywords
    palettes = {
        "login"   : [(99,102,241),(139,92,246),(236,72,153)],
        "home"    : [(16,185,129),(6,182,212),(59,130,246)],
        "shop"    : [(245,158,11),(239,68,68),(236,72,153)],
        "default" : [(99,102,241),(59,130,246),(16,185,129)],
    }
    key = next((k for k in palettes if k in prompt.lower()), "default")
    cols = palettes[key]

    # Draw fake UI components
    # Status bar
    draw.rectangle([0,0,W,28], fill=(30,30,45))
    draw.text((16,7), "9:41", fill=(220,220,220))
    draw.text((W-60,7), "● WiFi", fill=(220,220,220))

    # Header
    draw.rectangle([0,28,W,90], fill=cols[0])
    title = prompt[:28] + "…" if len(prompt) > 28 else prompt
    draw.text((20,50), title, fill="white")

    # Cards
    for i in range(3):
        y = 110 + i * 120
        draw.rounded_rectangle([20, y, W-20, y+100], radius=14,
                                fill=(28, 28, 42), outline=cols[i%3], width=1)
        draw.rectangle([36, y+16, 80, y+84], fill=cols[i%3])
        for j in range(3):
            draw.rounded_rectangle([96, y+16+j*20, W-36, y+28+j*20],
                                   radius=4, fill=(50,50,70))

    # Bottom nav
    draw.rectangle([0, H-64, W, H], fill=(20,20,35))
    for i, icon in enumerate(["⌂","⊕","♡","👤"]):
        x = 40 + i * 110
        draw.text((x, H-42), icon, fill=cols[i%3] if i==0 else (120,120,140))

    # Sketch overlay (very faint) from input
    if sketch:
        sk = sketch.convert("L").resize((W, H))
        sk_arr = np.array(sk).astype(float)
        sk_arr = (sk_arr - sk_arr.min()) / (sk_arr.max() - sk_arr.min() + 1e-8)
        img_arr = np.array(img).astype(float)
        blended = img_arr * 0.92 + sk_arr[:,:,None] * 0.08 * 255
        img = Image.fromarray(blended.clip(0,255).astype(np.uint8))

    img = img.filter(ImageFilter.GaussianBlur(0.4))
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  Core generation function
# ─────────────────────────────────────────────────────────────────────────────

def generate_ui(
    sketch_img,
    prompt,
    checkpoint_path,
    num_steps,
    guidance_scale,
    controlnet_scale,
    seed,
    preprocess_sketch,
    user_id_str,
):
    if sketch_img is None:
        return None, "⚠️ Please upload a sketch first."

    sketch = Image.fromarray(sketch_img).convert("RGB") if isinstance(sketch_img, np.ndarray) else sketch_img

    # Preprocess if requested
    if preprocess_sketch:
        try:
            from inference.inference_engine import InferenceEngine
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                sketch.save(f.name)
                sketch = InferenceEngine.preprocess_sketch(f.name, output_size=512)
            os.unlink(f.name)
        except Exception:
            pass  # silently skip if inference module not available

    user_id = int(user_id_str) if user_id_str.strip().isdigit() else None
    seed_val = int(seed) if seed >= 0 else random.randint(0, 99999)

    engine = get_engine(checkpoint_path)

    if engine is None:
        # Demo mode
        status = "⚠️ No checkpoint found — showing demo output. Train a model first!"
        result = _demo_generate(sketch, prompt or "Mobile UI", seed_val)
    else:
        try:
            status = f"✅ Generating with seed={seed_val}, steps={num_steps} …"
            result = engine.generate(
                sketch_image   = sketch,
                prompt         = prompt or "Design a modern mobile UI based on this sketch.",
                user_id        = user_id,
                num_steps      = int(num_steps),
                guidance_scale = float(guidance_scale),
                seed           = seed_val,
            )
            status = "✅ Generation complete!"
        except Exception as e:
            status = f"❌ Error: {e}"
            result = _demo_generate(sketch, prompt or "Mobile UI", seed_val)

    # Save to gallery
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"gen_{ts}_seed{seed_val}.png"
    result.save(GALLERY_DIR / name)

    # Save metadata
    meta = {
        "timestamp"   : ts,
        "prompt"      : prompt,
        "seed"        : seed_val,
        "steps"       : num_steps,
        "guidance"    : guidance_scale,
        "file"        : name,
    }
    with open(GALLERY_DIR / f"gen_{ts}.json", "w") as f:
        json.dump(meta, f, indent=2)

    return result, status


# ─────────────────────────────────────────────────────────────────────────────
#  Training launcher
# ─────────────────────────────────────────────────────────────────────────────

_train_proc = None

def start_training(batch_size, num_epochs, lr, use_fp16, phase1_epochs):
    global _train_proc
    import subprocess, sys

    if _train_proc and _train_proc.poll() is None:
        return "⚠️ Training is already running."

    cmd = [
        sys.executable, "train.py",
        f"training.batch_size={int(batch_size)}",
        f"training.num_epochs={int(num_epochs)}",
        f"optimizer.lr={float(lr)}",
        f"training.mixed_precision={'fp16' if use_fp16 else 'no'}",
        f"training.phase1_epochs={int(phase1_epochs)}",
    ]
    _train_proc = subprocess.Popen(cmd)
    return f"🚀 Training started (PID {_train_proc.pid}) — watch the Loss Curves tab."


def stop_training():
    global _train_proc
    if _train_proc and _train_proc.poll() is None:
        _train_proc.terminate()
        return "🛑 Training stopped."
    return "ℹ️ No training process running."


def get_training_status():
    if _train_proc is None:
        return "⬜ Not started"
    code = _train_proc.poll()
    if code is None:
        return "🟢 Training in progress …"
    elif code == 0:
        return "✅ Training completed successfully"
    else:
        return f"❌ Training exited with code {code}"


# ─────────────────────────────────────────────────────────────────────────────
#  Loss curve reader
# ─────────────────────────────────────────────────────────────────────────────

def read_loss_curves():
    """Read train_log.csv and return Gradio-compatible plot data."""
    if not LOG_CSV.exists():
        # Return empty placeholder
        return _empty_loss_plot()

    try:
        import pandas as pd
        df = pd.read_csv(LOG_CSV)
        if df.empty:
            return _empty_loss_plot()

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor="#0f0f19")
        fig.patch.set_alpha(1)

        loss_keys = ["loss_total","loss_gen","loss_rating","loss_align"]
        titles    = ["Total Loss","Generation Loss","Rating Loss","Alignment Loss"]
        colours   = ["#6366f1","#10b981","#f59e0b","#ec4899"]

        for ax, key, title, col in zip(axes.flatten(), loss_keys, titles, colours):
            ax.set_facecolor("#1a1a2e")
            if key in df.columns:
                ax.plot(df["step"], df[key], color=col, linewidth=1.8, alpha=0.9)
                ax.fill_between(df["step"], df[key], alpha=0.15, color=col)
            ax.set_title(title, color="white", fontsize=10, pad=8)
            ax.tick_params(colors="#888888", labelsize=8)
            ax.spines[:].set_color("#333355")
            ax.grid(alpha=0.15, color="#444466")

        plt.suptitle("Training Loss Curves", color="white", fontsize=13, y=1.01)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0f0f19")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    except Exception as e:
        return _empty_loss_plot(str(e))


def _empty_loss_plot(msg="No training data yet. Start training to see curves."):
    img = Image.new("RGB", (880, 560), (15, 15, 25))
    draw = ImageDraw.Draw(img)
    draw.text((200, 260), msg, fill=(120, 120, 160))
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  Gallery loader
# ─────────────────────────────────────────────────────────────────────────────

def load_gallery():
    imgs = sorted(GALLERY_DIR.glob("*.png"), key=os.path.getmtime, reverse=True)
    result = []
    for p in imgs[:24]:
        meta_path = p.with_suffix(".json").with_stem("gen_" + p.stem.split("_seed")[0][4:])
        prompt = ""
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    prompt = json.load(f).get("prompt", "")[:60]
            except Exception:
                pass
        result.append((str(p), prompt or p.name))
    return result if result else []


# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS  (dark, sophisticated)
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
/* ── Base ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-0:   #09090f;
    --bg-1:   #111120;
    --bg-2:   #1a1a2e;
    --bg-3:   #22223a;
    --accent: #6366f1;
    --accent2: #10b981;
    --accent3: #f59e0b;
    --text:   #e2e2f0;
    --muted:  #7878a0;
    --border: #2d2d50;
    --radius: 14px;
}

body, .gradio-container {
    background: var(--bg-0) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Header ───────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0f0f1e 0%, #1a1035 50%, #0f1a1e 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px 22px;
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 0 !important;
}
.app-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    box-shadow: 0 0 24px rgba(99,102,241,0.4);
}
.app-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #a5b4fc, #6366f1, #8b5cf6);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 !important;
}
.app-subtitle { color: var(--muted); font-size: 0.82rem; margin-top: 2px; }

/* ── Tabs ─────────────────────────────────────────────── */
.tab-nav { background: var(--bg-1) !important; border-bottom: 1px solid var(--border) !important; }
.tab-nav button {
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 10px 22px !important;
    border-radius: 0 !important;
    transition: all .2s !important;
}
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* ── Panels ───────────────────────────────────────────── */
.panel-box {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
}
.gr-panel, .gr-form, .gr-box {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Inputs ───────────────────────────────────────────── */
input, textarea, select, .gr-input, .gr-textarea {
    background: var(--bg-3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
input:focus, textarea:focus { border-color: var(--accent) !important; outline: none !important; }

/* ── Sliders ─────────────────────────────────────────── */
.gr-slider input[type=range]::-webkit-slider-thumb { background: var(--accent) !important; }

/* ── Buttons ──────────────────────────────────────────── */
.btn-primary {
    background: linear-gradient(135deg, var(--accent), #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: all .2s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}
.btn-primary:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 28px rgba(99,102,241,0.5) !important; }

.btn-success {
    background: linear-gradient(135deg, var(--accent2), #059669) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important; font-weight: 700 !important;
    padding: 11px 24px !important; cursor: pointer !important;
    box-shadow: 0 4px 16px rgba(16,185,129,0.3) !important;
}
.btn-danger {
    background: linear-gradient(135deg, #ef4444, #b91c1c) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.85rem !important;
    padding: 11px 24px !important; cursor: pointer !important;
}

/* ── Upload zone ──────────────────────────────────────── */
.upload-zone {
    border: 2px dashed var(--accent) !important;
    border-radius: var(--radius) !important;
    background: rgba(99,102,241,0.04) !important;
    transition: all .2s !important;
}
.upload-zone:hover { background: rgba(99,102,241,0.09) !important; }

/* ── Image outputs ────────────────────────────────────── */
.gr-image img { border-radius: 10px !important; }

/* ── Status badge ─────────────────────────────────────── */
.status-box {
    background: var(--bg-3) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    font-size: 0.82rem !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
}

/* ── Labels ───────────────────────────────────────────── */
label, .gr-label { color: var(--muted) !important; font-size: 0.78rem !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; }

/* ── Section headers ──────────────────────────────────── */
.section-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--accent) !important;
    margin-bottom: 12px !important;
}

/* ── Gallery ──────────────────────────────────────────── */
.gallery-grid .thumbnail-item { border-radius: 10px !important; overflow: hidden; }

/* ── Accordion ────────────────────────────────────────── */
.gr-accordion { background: var(--bg-2) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Example prompts
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_PROMPTS = [
    "Design a clean mobile login screen with email, password fields and a Google sign-in button",
    "Create a modern e-commerce home screen with search bar, category chips and product grid",
    "Generate a fitness tracker dashboard showing steps, calories, heart rate and weekly progress",
    "Design a minimalist chat messaging interface with message bubbles and a media input bar",
    "Create an onboarding screen for a travel app with a hero image and swipeable cards",
    "Design a music player with album art, waveform visualizer and playback controls",
    "Generate a food delivery app order tracking screen with a live map and order status",
    "Create a social media profile page with a grid layout, story bubbles and follow button",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Build Gradio app
# ─────────────────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(css=CSS, title="VLM UI Generator", theme=gr.themes.Base()) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
            <div class="app-logo">✦</div>
            <div>
                <div class="app-title">VLM UI Generator</div>
                <div class="app-subtitle">Sketch → Polished Mobile UI · Powered by CLIP + Stable Diffusion + ControlNet</div>
            </div>
        </div>
        """)

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ════════════════════════════════════════════════════════════════════
            #  TAB 1 — GENERATE
            # ════════════════════════════════════════════════════════════════════
            with gr.Tab("🎨  Generate"):
                with gr.Row(equal_height=True):

                    # ── Left column: inputs ───────────────────────────────────
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">01 — Upload Sketch</div>')
                        sketch_input = gr.Image(
                            label="Hand-drawn UI Sketch",
                            type="pil",
                            elem_classes=["upload-zone"],
                            height=280,
                        )

                        gr.HTML('<div class="section-title" style="margin-top:18px">02 — Describe Your UI</div>')
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="e.g. Design a mobile login screen with email and password fields…",
                            lines=3,
                        )

                        # Quick prompt buttons
                        gr.HTML('<div style="margin:8px 0 4px;font-size:0.72rem;color:#7878a0;text-transform:uppercase;letter-spacing:.08em">Quick prompts</div>')
                        with gr.Row():
                            for qp in EXAMPLE_PROMPTS[:4]:
                                short = qp[:28] + "…"
                                gr.Button(short, size="sm").click(
                                    fn=lambda x=qp: x, outputs=prompt_input
                                )
                        with gr.Row():
                            for qp in EXAMPLE_PROMPTS[4:]:
                                short = qp[:28] + "…"
                                gr.Button(short, size="sm").click(
                                    fn=lambda x=qp: x, outputs=prompt_input
                                )

                        gr.HTML('<div class="section-title" style="margin-top:18px">03 — Settings</div>')
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            checkpoint_input = gr.Textbox(
                                label="Checkpoint Path",
                                value="checkpoints/best_model.pth",
                            )
                            num_steps = gr.Slider(
                                label="Denoising Steps",
                                minimum=10, maximum=100, value=50, step=5,
                            )
                            guidance_scale = gr.Slider(
                                label="Guidance Scale (CFG)",
                                minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                            )
                            controlnet_scale = gr.Slider(
                                label="ControlNet Scale",
                                minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                            )
                            seed = gr.Number(label="Seed  (-1 = random)", value=-1, precision=0)
                            preprocess = gr.Checkbox(
                                label="🖊 Pre-process sketch (enhance edges — recommended for photos of paper)",
                                value=True,
                            )
                            user_id = gr.Textbox(label="User ID (optional, for personalised output)", value="")

                        generate_btn = gr.Button(
                            "✦ Generate UI", elem_classes=["btn-primary"], size="lg"
                        )

                    # ── Right column: output ──────────────────────────────────
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">04 — Generated UI</div>')
                        output_image = gr.Image(
                            label="Generated UI Design",
                            type="pil",
                            height=440,
                            interactive=False,
                        )
                        status_box = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_classes=["status-box"],
                        )

                        with gr.Row():
                            save_btn = gr.Button("💾 Save to Gallery", size="sm")
                            gr.Button("🔀 Random Seed", size="sm").click(
                                fn=lambda: random.randint(0, 99999),
                                outputs=seed,
                            )

                generate_btn.click(
                    fn=generate_ui,
                    inputs=[
                        sketch_input, prompt_input, checkpoint_input,
                        num_steps, guidance_scale, controlnet_scale,
                        seed, preprocess, user_id,
                    ],
                    outputs=[output_image, status_box],
                )

            # ════════════════════════════════════════════════════════════════════
            #  TAB 2 — TRAIN
            # ════════════════════════════════════════════════════════════════════
            with gr.Tab("🏋️  Train"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-title">Training Configuration</div>')

                        batch_size   = gr.Slider(label="Batch Size",      minimum=1,  maximum=32, value=8,   step=1)
                        num_epochs   = gr.Slider(label="Total Epochs",    minimum=1,  maximum=100, value=30,  step=1)
                        phase1_ep    = gr.Slider(label="Phase 1 Epochs (fusion only)", minimum=1, maximum=30, value=10, step=1)
                        lr_input     = gr.Number(label="Learning Rate",   value=1e-4, precision=6)
                        use_fp16     = gr.Checkbox(label="Use Mixed Precision (fp16) — requires GPU", value=True)

                        gr.HTML("""
                        <div style="background:#1a1a2e;border:1px solid #2d2d50;border-radius:10px;
                                    padding:14px;margin:12px 0;font-size:.8rem;color:#7878a0;line-height:1.7">
                            <strong style="color:#a5b4fc">ℹ️ Two-phase training:</strong><br>
                            <b style="color:#e2e2f0">Phase 1</b> — Trains cross-attention fusion, user preference module
                            and rating head only (encoders frozen). Fast convergence.<br>
                            <b style="color:#e2e2f0">Phase 2</b> — End-to-end fine-tuning including CLIP LoRA adapters,
                            ControlNet and UNet cross-attention layers.
                        </div>
                        """)

                        with gr.Row():
                            train_btn = gr.Button("🚀 Start Training", elem_classes=["btn-success"])
                            stop_btn  = gr.Button("🛑 Stop",           elem_classes=["btn-danger"])

                        train_status = gr.Textbox(
                            label="Training Status",
                            interactive=False,
                            elem_classes=["status-box"],
                        )

                        refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")

                    with gr.Column(scale=2):
                        gr.HTML('<div class="section-title">Live Loss Curves</div>')
                        loss_plot = gr.Image(
                            label="",
                            type="pil",
                            interactive=False,
                            height=460,
                        )
                        refresh_plot_btn = gr.Button("🔄 Refresh Curves", size="sm")

                train_btn.click(
                    fn=start_training,
                    inputs=[batch_size, num_epochs, lr_input, use_fp16, phase1_ep],
                    outputs=train_status,
                )
                stop_btn.click(fn=stop_training, outputs=train_status)
                refresh_status_btn.click(fn=get_training_status, outputs=train_status)
                refresh_plot_btn.click(fn=read_loss_curves, outputs=loss_plot)

                # Auto-load plot on tab open
                demo.load(fn=read_loss_curves, outputs=loss_plot)

            # ════════════════════════════════════════════════════════════════════
            #  TAB 3 — GALLERY
            # ════════════════════════════════════════════════════════════════════
            with gr.Tab("🖼️  Gallery"):
                gr.HTML('<div class="section-title" style="padding:16px 0 4px">Past Generations</div>')

                refresh_gallery_btn = gr.Button("🔄 Refresh Gallery", size="sm")
                gallery = gr.Gallery(
                    label="",
                    columns=4,
                    rows=3,
                    height=520,
                    object_fit="cover",
                    elem_classes=["gallery-grid"],
                )

                refresh_gallery_btn.click(fn=load_gallery, outputs=gallery)
                demo.load(fn=load_gallery, outputs=gallery)

            # ════════════════════════════════════════════════════════════════════
            #  TAB 4 — ABOUT
            # ════════════════════════════════════════════════════════════════════
            with gr.Tab("ℹ️  About"):
                gr.HTML("""
                <div style="max-width:780px;margin:32px auto;font-family:'DM Sans',sans-serif;line-height:1.8;color:#c8c8e8">

                <h2 style="font-family:'Space Mono',monospace;color:#a5b4fc;font-size:1.2rem">Architecture</h2>

                <div style="background:#1a1a2e;border:1px solid #2d2d50;border-radius:14px;padding:24px;
                            font-family:'Space Mono',monospace;font-size:.78rem;color:#888;line-height:2">
Hand-drawn Sketch ──► CLIP Image Encoder ──► Patch Embeddings (B, P, d)
                                                       │
Text Prompt       ──► CLIP Text Encoder  ──► Token Embeddings (B, L, d)
                                                       │
                        Cross-Attention Fusion (Text attends to Image patches)
                                  Attention(Q,K,V) = softmax(QKᵀ/√d)V
                                                       │
                                               h_ui ∈ ℝᵈ
                                           /               \
                        User Preference Module         Fused Conditioning
                               p_u                        Injector
                                │                             │
                        r̂ = σ(p_uᵀ W_r h_ui)      Stable Diffusion UNet
                                                  + ControlNet (scribble)
                                                             │
                                                    Generated UI Image
                </div>

                <h2 style="font-family:'Space Mono',monospace;color:#a5b4fc;font-size:1.2rem;margin-top:28px">Datasets</h2>
                <table style="width:100%;border-collapse:collapse;font-size:.85rem">
                  <tr style="border-bottom:1px solid #2d2d50">
                    <td style="padding:10px;color:#6366f1;font-family:'Space Mono',monospace">mrtoy/mobile-ui-design</td>
                    <td style="padding:10px;color:#888">HuggingFace</td>
                    <td style="padding:10px">Target polished UI images</td>
                  </tr>
                  <tr style="border-bottom:1px solid #2d2d50">
                    <td style="padding:10px;color:#10b981;font-family:'Space Mono',monospace">vinothpandian/uisketch</td>
                    <td style="padding:10px;color:#888">Kaggle</td>
                    <td style="padding:10px">Real hand-drawn sketch ↔ UI pairs</td>
                  </tr>
                  <tr>
                    <td style="padding:10px;color:#f59e0b;font-family:'Space Mono',monospace">antrixsh/prompt-engineering…</td>
                    <td style="padding:10px;color:#888">Kaggle</td>
                    <td style="padding:10px">Natural language prompts</td>
                  </tr>
                </table>

                <h2 style="font-family:'Space Mono',monospace;color:#a5b4fc;font-size:1.2rem;margin-top:28px">Loss Functions</h2>
                <div style="background:#1a1a2e;border:1px solid #2d2d50;border-radius:14px;padding:20px;
                            font-family:'Space Mono',monospace;font-size:.8rem;color:#888;line-height:2.2">
L = λ₁·L_rating + λ₂·L_BPR + λ₃·L_align + λ₄·L_gen<br><br>
<span style="color:#6366f1">L_rating</span> = (1/|D|) Σ (r_ui − r̂_ui)²<br>
<span style="color:#10b981">L_BPR   </span> = −Σ log σ(r̂_ui − r̂_uj)<br>
<span style="color:#f59e0b">L_align </span> = −log[exp(sim(v,t)/τ) / Σ exp(sim(v,tⱼ)/τ)]<br>
<span style="color:#ec4899">L_gen   </span> = MSE(ε, ε̂)  [diffusion noise prediction]
                </div>

                </div>
                """)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--share",  action="store_true", help="Create a public Gradio URL (needed for Colab)")
    p.add_argument("--port",   type=int, default=7860)
    p.add_argument("--host",   default="0.0.0.0")
    p.add_argument("--debug",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_app()
    demo.launch(
        server_name = args.host,
        server_port = args.port,
        share       = args.share,
        debug       = args.debug,
        show_error  = True,
    )
