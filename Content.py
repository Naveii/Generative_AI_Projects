# Ensure the required libraries are installed before running this script.
# Use the following command in your terminal to install them:
# pip install transformers Pillow torch torchvision torchaudio
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# Load an image
image = Image.open("D:\Data_science\Coursera_AI_Course\Image_Captioning\Street Style Showcase on Neutral Canvas.png")
# Prepare the image
inputs = processor(image, return_tensors="pt")
# Generate captions
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0],skip_special_tokens=True)
 
print("Generated Caption:", caption)