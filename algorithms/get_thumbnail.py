import os
import openslide
from PIL import Image
from tqdm import tqdm
import numpy as np

def generate_thumbnail(wsi_path, target_mpp=10.0, default_mpp=0.25):
    """
    Generate a thumbnail from a WSI at a specified target MPP.

    Args:
        wsi_path (str): Path to the WSI file.
        output_dir (str): Directory to save the thumbnail.
        target_mpp (float): Desired microns per pixel for the thumbnail (default: 10.0).
        default_mpp (float): Default MPP to use if the slide lacks the 'openslide.mpp-x' property (default: 0.25).
    """
    try:
        # Open the slide
        slide = openslide.OpenSlide(wsi_path)
        
        # Retrieve the slide's MPP (microns per pixel)
        try:
            slide_mpp = float(slide.properties["openslide.mpp-x"])
        except KeyError:
            print(f"Warning: 'openslide.mpp-x' not found for {wsi_path}. Using default MPP: {default_mpp}")
            slide_mpp = default_mpp
        
        # Calculate the downsample factor
        downsample_factor = target_mpp / slide_mpp
        
        # Get level 0 dimensions (highest resolution)
        w_l0, h_l0 = slide.level_dimensions[0]
        
        # Calculate thumbnail dimensions
        thumb_width = int(w_l0 / downsample_factor)
        thumb_height = int(h_l0 / downsample_factor)
        
        # Generate the thumbnail
        thumbnail = slide.get_thumbnail((thumb_width, thumb_height))
        if thumbnail.mode == 'RGBA':
            thumbnail = thumbnail.convert('RGB')
        
        # # Save the thumbnail as a PNG
        # base_name = os.path.splitext(os.path.basename(wsi_path))[0]
        # thumbnail_path = os.path.join(output_dir, f"{base_name}_thumbnail_mpp_{target_mpp}.png")
        # thumbnail.save(thumbnail_path)
        # print(f"Thumbnail saved to {thumbnail_path}")
        return np.array(thumbnail)
    
    except Exception as e:
        print(f"Error processing {wsi_path}: {e}")
    finally:
        # Ensure the slide is closed
        if 'slide' in locals():
            slide.close()

def main():
    """
    Process all WSIs in the input directory and generate thumbnails.
    """
    # Define input and output directories (replace with your paths)
    wsi_dir = "/path/to/your/wsi/files"  # Directory containing .svs files
    output_dir = "/path/to/your/output/thumbnails"  # Directory to save thumbnails
    os.makedirs(output_dir, exist_ok=True)
    
    # List all .svs files in the input directory
    wsi_files = [f for f in os.listdir(wsi_dir) if f.lower().endswith('.svs')]
    
    # Process each WSI
    for wsi_file in tqdm(wsi_files, desc="Generating thumbnails"):
        wsi_path = os.path.join(wsi_dir, wsi_file)
        generate_thumbnail(wsi_path, output_dir, target_mpp=10.0)

if __name__ == "__main__":
    main()