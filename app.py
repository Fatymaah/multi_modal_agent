import gradio as gr
from groq import Groq
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize BLIP for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize Groq client
groq_client = Groq(api_key=api_key)

# Core function
def multimodal_agent(image, user_text):
    # Step 1: Generate image caption
    img = Image.open(image).convert("RGB")
    inputs = processor(img, return_tensors="pt")
    caption_ids = model.generate(**inputs)
    image_caption = processor.decode(caption_ids[0], skip_special_tokens=True)

    # Step 2: Send to Groq LLM
    prompt = f"Image description: {image_caption}\nUser question: {user_text}\nAnswer:"
    chat_completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",  # fast Groq model
        messages=[{"role": "user", "content": prompt}],
    )

    answer = chat_completion.choices[0].message.content
    return f"🖼 Image Caption: {image_caption}\n\n🤖 Agent Answer: {answer}"

# Gradio UI
demo = gr.Interface(
    fn=multimodal_agent,
    inputs=[
        gr.Image(type="filepath", label="Upload an image"),
        gr.Textbox(label="Ask a question about the image")
    ],
    outputs=gr.Textbox(label="Agent Output", lines=12),
    title="Multimodal Agent (Groq + BLIP-2)",
    description="Upload an image + type a question. The agent will describe the image and answer your query."
)

if __name__ == "__main__":
    demo.launch()
