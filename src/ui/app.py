import gradio as gr
import tempfile
import yaml
from pathlib import Path


def load_config(config_path: str = "configs/default.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def master_audio(input_audio, checkpoint_path):
    if input_audio is None:
        return None, "Please upload an audio file."

    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None, "Please provide a valid model checkpoint path."

    try:
        from ..inference import MasteringInference

        engine = MasteringInference(checkpoint_path)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        engine.process_file(input_audio, output_path)
        return output_path, "Mastering complete."

    except RuntimeError as e:
        if "Model not loaded" in str(e):
            return None, (
                "DeepAFx-ST model not yet integrated. "
                "See the README for integration instructions."
            )
        return None, f"Error: {e}"
    except Exception as e:
        return None, f"Error: {e}"


def create_app():
    with gr.Blocks(title="Personalised AI Mastering") as app:
        gr.Markdown("# Personalised AI Mastering")
        gr.Markdown(
            "Upload a pre-mastered WAV file to apply your learned mastering style."
        )

        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(
                    label="Pre-Master (Input)",
                    type="filepath",
                    sources=["upload"],
                )
                checkpoint = gr.Textbox(
                    label="Model Checkpoint Path",
                    value="models/best_model.pt",
                    placeholder="Path to trained model checkpoint",
                )
                process_btn = gr.Button("Master", variant="primary")

            with gr.Column():
                output_audio = gr.Audio(label="Mastered (Output)", type="filepath")
                status = gr.Textbox(label="Status", interactive=False)

        process_btn.click(
            fn=master_audio,
            inputs=[input_audio, checkpoint],
            outputs=[output_audio, status],
        )

        gr.Markdown("---")
        gr.Markdown(
            "Built with [DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST) "
            "and [Gradio](https://gradio.app)"
        )

    return app


if __name__ == "__main__":
    config = load_config()
    port = config.get("deployment", {}).get("gradio_port", 7860)
    app = create_app()
    app.launch(server_port=port)
