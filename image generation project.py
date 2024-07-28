# Import the necessary libraries
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch

# Define the model IDs for different versions of the diffusion models
model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id2 = "stabilityai/stable-diffusion-xl-base-1.0"

# Your Hugging Face token
token = "YOUR_HUGGING_FACE_TOKEN"

# Load the Stable Diffusion Pipeline using the first model ID
pipe = StableDiffusionPipeline.from_pretrained(model_id1, use_auth_token=token)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Prompt the user to enter a description for the image to be generated
prompt = input("Enter the image description here: ")

# Generate the image based on the user's prompt
try:
    # Batch processing if you want to generate multiple images
    num_images = 1  # Change this number to generate more images at once
    images = pipe([prompt] * num_images).images

    # Display the generated image(s) using matplotlib
    for idx, image in enumerate(images):
        plt.imshow(image)
        plt.axis('off')  # Hide the axes
        plt.savefig(f"generated_image_{idx}.png")  # Save the plotted image as a PNG file
        plt.show()  # Show the image on the screen

except Exception as e:
    print(f"An error occurred: {e}")

