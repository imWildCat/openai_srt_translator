import gradio as gr
import os
from typing import Tuple

def translate_srt(file: gr.inputs.File, language: str ="English", batch_size: int=50, model: str="gpt-3.5-turbo", verbose=False) -> Tuple[gr.outputs.Textbox, gr.outputs.File]:
    # Save the uploaded file
    with open("temp.srt", "wb") as temp_file:
        temp_file.write(file)

    # Call the main function
    main(["temp.srt"], language, batch_size, model, verbose)

    # Read the translated file
    translated_filename = get_translated_filename("temp.srt")
    with open(translated_filename, "r") as translated_file:
        translated_content = translated_file.read()

    # Delete the temporary files
    os.remove("temp.srt")
    os.remove(translated_filename)

    # Return the translated content and the translated file
    return translated_content, translated_filename

# Define the Gradio interface
iface = gr.Interface(
    fn=translate_srt,
    inputs=[
        gr.inputs.File(label="Upload a .srt file"),
        gr.inputs.Textbox(default="English", label="Language"),
        gr.inputs.Slider(minimum=1, maximum=100, default=50, label="Batch Size"),
        gr.inputs.Textbox(default="gpt-3.5-turbo", label="Model"),
        gr.inputs.Checkbox(default=False, label="Verbose")
    ],
    outputs=[
        gr.outputs.Textbox(label="Translated Content"),
        gr.outputs.File(label="Download Translated File")
    ],
    title="SRT File Translator",
    description="Upload a .srt file to translate it."
)

# Launch the Gradio interface
iface.launch()