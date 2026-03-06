import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


CHECKPOINT_DIR = "./blip_video_model_2" 
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading BLIP model and processor... This might take a few seconds.")

# Load the fine-tuned BLIP model directly
processor = BlipProcessor.from_pretrained(CHECKPOINT_DIR)
model = BlipForConditionalGeneration.from_pretrained(CHECKPOINT_DIR)

model.to(DEVICE)
model.eval()

# Processing & Generation Function

def process_and_caption(video_path):
    if not video_path:
        return None, "Please upload a video first."

    # --- Extract 8 Frames for the UI ---
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return None, "Error: Could not read video file."

    # Sample 8 evenly spaced frames just like the project instructions requested
    frame_indices = np.linspace(0, total_frames - 1, 8, dtype=int)
    frames_rgb = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB for accurate colors
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    # Create PIL Images to display in the Gradio UI gallery
    display_images = [Image.fromarray(f) for f in frames_rgb]

    # --- Prepare for BLIP ---
    # BLIP is an image model, so we pick the most representative action frame (the middle one)
    middle_image = display_images[len(display_images) // 2]
    
    inputs = processor(images=middle_image, return_tensors="pt").to(DEVICE)

    # --- Generate Caption with BEAM SEARCH ---
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=30,
            num_beams=5,             # BEAM SEARCH: Explores 5 paths simultaneously
            early_stopping=True,     # Stops as soon as the sentence naturally ends
            no_repeat_ngram_size=2   # Prevents stuttering/repeating words
        )

    # Decode the final text
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    
    return display_images, caption


#  Build the Gradio Interface

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Fine-Tuned Video Captioning App (BLIP)")
    gr.Markdown("Upload an MP4. The app will extract 8 frames, display them, and use the middle frame to generate a high-quality caption using Beam Search.")
    
    with gr.Row():
        # Left Column: Input
        with gr.Column():
            video_input = gr.Video(label="Upload Video Clip")
            submit_btn = gr.Button("Generate Caption", variant="primary")
            
        # Right Column: Output
        with gr.Column():
            caption_output = gr.Textbox(label="Generated Caption", lines=2, text_align="center")
            gallery_output = gr.Gallery(label="8 Sampled Frames", columns=4, rows=2, object_fit="contain")
            
    # Connect the button to the function
    submit_btn.click(
        fn=process_and_caption, 
        inputs=[video_input], 
        outputs=[gallery_output, caption_output]
    )

if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(share=True)
