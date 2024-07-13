import gradio as gr
from transformers import pipeline

from phonemizer.backend.espeak.wrapper import EspeakWrapper

# get the text to speech model
narrator = pipeline(task = 'text-to-speech',
                    model = 'kakao-enterprise/vits-ljs')


EspeakWrapper.set_library('C:\Program Files\eSpeak NG\libespeak-ng.dll')

def narrate_text(text):
    # Generate the narrated audio
    narrated_text = narrator(text)
    audio = narrated_text['audio'][0]
    sampling_rate = narrated_text['sampling_rate']
    # Convert the audio to a format playable in Gradio
    return sampling_rate, audio

# Create the Gradio interface
text_to_speech_interface = gr.Interface(
    fn=narrate_text, 
    inputs=gr.Textbox(lines=5, placeholder="Enter text here...", label = "Input Text"), 
    outputs = 'audio',
    allow_flagging = 'never'
)

# Add Markdown content
markdown_content_text2speech = gr.Markdown(
    """
    <div style='text-align: center; font-family: "Times New Roman";'>
        <h1 style='color: #FF6347;'>Text to Speech</h1>
        <h3 style='color: #4682B4;'>Model: kakao-enterprise/vits-ljs</h3>
        <h3 style='color: #32CD32;'>Made By: Md. Mahmudun Nabi</h3>
    </div>
    """
)

# Combine the Markdown content and the demo interface
text2speech_with_markdown = gr.Blocks()
with text2speech_with_markdown:
    markdown_content_text2speech.render()
    text_to_speech_interface.render()