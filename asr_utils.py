
import gradio as gr
from transformers import pipeline
asr = gr.Blocks()

# Get the speech recognizer model
speech_recognizer = pipeline(task = 'automatic-speech-recognition',
                             model = 'distil-whisper/distil-small.en')

# Helper functions
def transcribe(filepath):
    if filepath is None:
        gr.Warning('No audio found, please retry.')
        return ''
    output = speech_recognizer(
        filepath,
        max_new_tokens = 256,
        chunk_length_s = 30,
        batch_size = 1   # If you have more computational capability, you can increase the batch size
    )
    return output['text']

mic_transcribe = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources="microphone",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    allow_flagging="never")

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources="upload",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    allow_flagging="never",
)

with asr:
    gr.TabbedInterface(
        [mic_transcribe,
         file_transcribe],
        ["Transcribe Microphone",
         "Transcribe Audio File"],
    )



# Add Markdown content
markdown_content_asr = gr.Markdown(
    """
    <div style='text-align: center; font-family: "Times New Roman";'>
        <h1 style='color: #FF6347;'>Automatic Speech Recognition</h1>
        <h3 style='color: #4682B4;'>Model: distil-whisper/distil-small.en</h3>
        <h3 style='color: #32CD32;'>Made By: Md. Mahmudun Nabi</h3>
    </div>
    """
)

# Combine the Markdown content and the demo interface
asr_with_markdown = gr.Blocks()
with asr_with_markdown:
    markdown_content_asr.render()
    asr.render()