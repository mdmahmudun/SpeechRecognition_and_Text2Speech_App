import warnings
warnings.filterwarnings('ignore')
from transformers.utils import logging
logging.set_verbosity_error()

import gradio as gr

from asr_utils import asr_with_markdown
from text2speech_utils import text2speech_with_markdown

# Initiate app
app = gr.Blocks()
with app:
    gr.TabbedInterface(
        [asr_with_markdown,
         text2speech_with_markdown],
        ["Automatic Speech Recognition",
         "Text to Specch"],
    )

if __name__ == "__main__":
    app.launch()