import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

def create_simple_rgb(input_path, output_path, title, satellite):
    """
    Create simple RGB image focused on study area
    """
    try:
        with rasterio.open(input_path) as src:
            # Read RGB bands (Red=4, Green=3, Blue=2)
            red = src.read(4).astype(np.float32)
            green = src.read(3).astype(np.float32)
            blue = src.read(2).astype(np.float32)
            
            print(f"Processing {title} ({satellite})...")
            
            # Simple valid data mask (exclude zeros and very low values)
            valid_mask = (red > 0) & (green > 0) & (blue > 0)
            
            # Find study area boundaries
            if np.sum(valid_mask) > 0:
                rows, cols = np.where(valid_mask)
                row_min, row_max = rows.min(), rows.max()
                col_min, col_max = cols.min(), cols.max()
                
                # Add small padding
                padding = 20
                row_min = max(0, row_min - padding)
                row_max = min(red.shape[0], row_max + padding)
                col_min = max(0, col_min - padding)
                col_max = min(red.shape[1], col_max + padding)
                
                # Crop to study area
                red_crop = red[row_min:row_max, col_min:col_max]
                green_crop = green[row_min:row_max, col_min:col_max]
                blue_crop = blue[row_min:row_max, col_min:col_max]
                valid_crop = valid_mask[row_min:row_max, col_min:col_max]
                
                # Simple stretch using valid pixels only
                def simple_stretch(band, mask):
                    if np.sum(mask) == 0:
                        return band
                    
                    valid_pixels = band[mask]
                    p2 = np.percentile(valid_pixels, 2)
                    p98 = np.percentile(valid_pixels, 98)
                    
                    stretched = np.clip((band - p2) / (p98 - p2), 0, 1)
                    return stretched
                
                red_stretched = simple_stretch(red_crop, valid_crop)
                green_stretched = simple_stretch(green_crop, valid_crop)
                blue_stretched = simple_stretch(blue_crop, valid_crop)
                
                # Create RGB array
                rgb = np.stack([red_stretched, green_stretched, blue_stretched], axis=2)
                
                # Set invalid pixels to white
                rgb[~valid_crop] = [1, 1, 1]
                
                # Create simple plot
                plt.figure(figsize=(10, 8))
                plt.imshow(rgb)
                plt.title(f'{title}', fontsize=14, fontweight='bold')
                plt.axis('off')
                
                # Save
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"Saved: {os.path.basename(output_path)}")
                
            else:
                print(f"No valid data found in {title}")
                
    except Exception as e:
        print(f"Error processing {title}: {str(e)}")

def main():
    # Paths
    base_path = r"Downloads\New folder"
    clipped_images_path = os.path.join(base_path, "Clipped_Images")
    rgb_output_path = os.path.join(base_path, "Results", "RGB Images")
    
    # Clean output directory
    if os.path.exists(rgb_output_path):
        for file in os.listdir(rgb_output_path):
            if file.endswith('.png'):
                os.remove(os.path.join(rgb_output_path, file))
    else:
        os.makedirs(rgb_output_path)
    
    # Define datasets
    datasets = [
        {
            'input': os.path.join(clipped_images_path, "Jan 2023_clipped_NGP.tif"),
            'output': os.path.join(rgb_output_path, "Jan_2023_RGB.png"),
            'title': 'January 2023',
            'satellite': 'Landsat 8'
        },
        {
            'input': os.path.join(clipped_images_path, "Jan 2024_clipped_NGP.tif"),
            'output': os.path.join(rgb_output_path, "Jan_2024_RGB.png"),
            'title': 'January 2024',
            'satellite': 'Landsat 8'
        },
        {
            'input': os.path.join(clipped_images_path, "May 2023_clipped_NGP.tif"),
            'output': os.path.join(rgb_output_path, "May_2023_RGB.png"),
            'title': 'May 2023',
            'satellite': 'Landsat 8'
        },
        {
            'input': os.path.join(clipped_images_path, "May 2024_clipped_NGP.tif"),
            'output': os.path.join(rgb_output_path, "May_2024_RGB.png"),
            'title': 'May 2024',
            'satellite': 'Landsat 9'
        }
    ]
    
    print("Creating Simple RGB Images")
    print("=" * 40)
    
    for dataset in datasets:
        if os.path.exists(dataset['input']):
            create_simple_rgb(dataset['input'], dataset['output'], dataset['title'], dataset['satellite'])
        else:
            print(f"Input file not found: {os.path.basename(dataset['input'])}")
    
    print("\nSimple RGB generation completed!")

if __name__ == "__main__":
    main()
