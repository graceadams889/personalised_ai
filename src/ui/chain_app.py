"""Gradio app for the parametric mastering chain (Path B).

Drag a 44.1 kHz WAV in, pick a genre profile, click Master, get the mastered
audio back with target-vs-achieved stats. Run via scripts/launch_chain_ui.py.
"""
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server-side rendering
import matplotlib.pyplot as plt

import gradio as gr
import numpy as np
import soundfile as sf

from src.chain import apply_chain, available_genres, get_profile
from src.chain.measure import lufs, crest_db, stereo_corr, BANDS

ROOT = Path(__file__).resolve().parents[2]
UI_OUT_DIR = ROOT / "outputs" / "ui_runs"
UI_OUT_DIR.mkdir(parents=True, exist_ok=True)

GENRES = available_genres()


def render_profile_panel(genre: str):
    """Build the per-genre profile summary (markdown) and the EQ-curve plot."""
    profile = get_profile(genre)

    md = (
        f"### **{genre}** profile &nbsp;·&nbsp; *n = {profile['n_tracks']} tracks*\n\n"
        f"| Target | Value |\n"
        f"| --- | --- |\n"
        f"| Integrated loudness | **{profile['master_lufs']:+.1f} LUFS** |\n"
        f"| Crest factor | **{profile['master_crest_db']:.1f} dB** |\n"
        f"| L–R correlation | **{profile['master_lr_corr']:+.2f}** |"
    )

    gains = profile["eq_gains_db"]
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=110)
    ax.semilogx(BANDS, gains, "o-", color="#4f46e5", linewidth=2.2, markersize=7,
                markerfacecolor="#4f46e5", markeredgecolor="white", markeredgewidth=1)
    ax.axhline(0, color="#888", linestyle="--", linewidth=0.8)
    ax.fill_between(BANDS, 0, gains, color="#4f46e5", alpha=0.12)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title(f"{genre} — measured mastering EQ curve", fontsize=11)
    ax.set_xticks(BANDS)
    ax.set_xticklabels(
        [str(b) if b < 1000 else f"{b//1000}k" for b in BANDS],
        fontsize=9,
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(55, 22000)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return md, fig


def master_audio(input_path: str | None, genre: str, use_width: bool):
    if input_path is None:
        return None, "**Please upload a WAV file first.**"

    try:
        audio, sr = sf.read(input_path, always_2d=True)
    except Exception as e:
        return None, f"**Could not read file:** {e}"

    if sr < 32000:
        return None, (
            f"**Sample rate too low** (this file is {sr} Hz). "
            "Please use 44.1, 48, 88.2, or 96 kHz."
        )

    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    audio = audio.astype(np.float64)

    mastered = apply_chain(audio, sr, genre, use_width=use_width)

    profile = get_profile(genre)
    a_lufs = lufs(mastered, sr)
    a_crest = crest_db(mastered)
    a_corr = stereo_corr(mastered)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    in_name = Path(input_path).stem
    width_tag = "_w" if use_width else ""
    out_path = UI_OUT_DIR / f"{in_name}_{genre}{width_tag}_{timestamp}.wav"
    sf.write(out_path, np.clip(mastered, -1.0, 1.0), sr, subtype="PCM_24")

    stats = (
        f"### Result &nbsp;·&nbsp; *profile: {genre}*\n\n"
        f"| | Target | Achieved | Δ |\n"
        f"| --- | --- | --- | --- |\n"
        f"| LUFS | {profile['master_lufs']:+.2f} | {a_lufs:+.2f} | "
        f"{a_lufs - profile['master_lufs']:+.2f} |\n"
        f"| Crest (dB) | {profile['master_crest_db']:.2f} | {a_crest:.2f} | "
        f"{a_crest - profile['master_crest_db']:+.2f} |\n"
        f"| L–R corr | {profile['master_lr_corr']:+.3f} | {a_corr:+.3f} | "
        f"{a_corr - profile['master_lr_corr']:+.3f} |\n\n"
        f"<sub>Saved to <code>outputs/ui_runs/{out_path.name}</code></sub>"
    )
    return str(out_path), stats


THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#f8fafc",
    block_background_fill="white",
    block_border_width="1px",
    block_shadow="0 1px 2px rgba(0,0,0,0.04)",
    block_radius="10px",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
)

CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
.thesis-header { padding: 12px 4px 0; }
.thesis-footer { padding: 14px 4px; font-size: 13px; color: #64748b;
                 border-top: 1px solid #e2e8f0; margin-top: 18px; }
.thesis-footer a { color: #4f46e5; text-decoration: none; }
"""


with gr.Blocks(title="Personalised Mastering Chain", theme=THEME, css=CUSTOM_CSS) as app:

    gr.Markdown(
        "# Personalised Mastering Chain\n"
        "**Grace Adams** — MSc Music and Media Technologies, Trinity College Dublin\n\n"
        "A parametric mastering chain fitted from 13 paired mix/master recordings "
        "across four genres. Drop a stereo WAV (44.1, 48, 88.2, or 96 kHz), choose "
        "the closest-genre profile, and the chain applies a 9-band peaking EQ, a "
        "soft-knee compressor, and a peak limiter — each targeting the per-genre "
        "profile's measured loudness, dynamics, and tonal balance.",
        elem_classes="thesis-header",
    )

    with gr.Row():
        profile_md = gr.Markdown()
        eq_plot = gr.Plot(label="EQ curve")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 1 · Input")
                input_audio = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Drag a WAV here  ·  44.1 / 48 / 88.2 / 96 kHz",
                )
                genre = gr.Dropdown(
                    choices=GENRES,
                    value=GENRES[0] if GENRES else None,
                    label="Genre profile",
                )
                use_width = gr.Checkbox(
                    label="Include stereo widening (experimental)",
                    value=False,
                    info="Off by default. Listening tests found the chain's "
                         "single-side-gain widener produced artefacts; the mix's "
                         "natural stereo image is usually preferable.",
                )
                master_btn = gr.Button("Master", variant="primary", size="lg")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 2 · Output")
                output_audio = gr.Audio(
                    type="filepath",
                    label="Mastered output",
                    interactive=False,
                )
                stats = gr.Markdown()

    gr.Markdown(
        f"<div class='thesis-footer'>"
        f"Profiles fitted from {sum(get_profile(g)['n_tracks'] for g in GENRES)} "
        f"paired tracks across {len(GENRES)} genres "
        f"({', '.join(GENRES)}). Chain: 9-band peaking EQ → soft-knee compressor "
        f"(bisected to target crest) → optional M/S widener → peak limiter targeting "
        f"the profile's integrated LUFS. Source code under version control in "
        f"<code>personalised_ai/</code>."
        f"</div>",
        elem_classes="thesis-footer",
    )

    genre.change(render_profile_panel, inputs=genre, outputs=[profile_md, eq_plot])
    app.load(render_profile_panel, inputs=genre, outputs=[profile_md, eq_plot])

    master_btn.click(
        master_audio,
        inputs=[input_audio, genre, use_width],
        outputs=[output_audio, stats],
    )
