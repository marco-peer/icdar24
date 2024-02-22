import os
from PIL import Image

def scale_images(input_dir, output_dir, scale_factor):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all files in the input directory
    files = os.listdir(input_dir)
    print(f'Found {len(files)} files')

    for file in files:
        file_path = os.path.join(input_dir, file)
        
        # Check if the file is an image
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Open the image
            img = Image.open(file_path)
            
            # Get the new dimensions based on the scale factor
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            
            # Resize the image
            scaled_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save the scaled image to the output directory
            output_file_path = os.path.join(output_dir, file)
            scaled_img.save(output_file_path)

            print(f"Image '{file}' scaled and saved to '{output_file_path}'")

# Example usage:
input_directory = "/data/mpeer/grk50"
output_directory = "/data/mpeer/grk50_scaled_0p5"
scale_factor = 0.5  # Change this to your desired scale factor

scale_images(input_directory, output_directory, scale_factor)