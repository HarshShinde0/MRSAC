"""
Script to clip processed Landsat images using NGP shapefile mask
This will clip all 4 time period images using the NGP_Zipcode_Shapes.shp file
"""

import os
import glob
from pathlib import Path
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np

def clip_image_with_shapefile(image_path, shapefile_path, output_path):
    """
    Clip a raster image using a shapefile mask
    
    Args:
        image_path (str): Path to the input raster image
        shapefile_path (str): Path to the shapefile for clipping
        output_path (str): Path for the clipped output image
    
    Returns:
        bool: Success status
    """
    try:
        print(f"Clipping: {os.path.basename(image_path)}")
        
        # Read the shapefile
        print("  - Loading shapefile...")
        gdf = gpd.read_file(shapefile_path)
        
        # Get the geometry for masking
        shapes = [geom for geom in gdf.geometry]
        
        # Open the raster
        print("  - Loading raster image...")
        with rasterio.open(image_path) as src:
            # Check if CRS match, if not reproject shapefile
            if src.crs != gdf.crs:
                print(f"  - Reprojecting shapefile from {gdf.crs} to {src.crs}")
                gdf = gdf.to_crs(src.crs)
                shapes = [geom for geom in gdf.geometry]
            
            # Clip the raster with the shapefile
            print("  - Performing clipping operation...")
            out_image, out_transform = mask(src, shapes, crop=True)
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw"
            })
        
        # Write the clipped image
        print(f"  - Saving clipped image to: {output_path}")
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
        
        print(f"  - Successfully clipped: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"  - Error clipping {os.path.basename(image_path)}: {str(e)}")
        return False

def process_all_images():
    """
    Process all 4 time period images and clip them with the shapefile
    """
    # Define paths
    base_dir = r"Downloads\New folder"
    processed_dir = os.path.join(base_dir, "Processed")
    shapefile_path = os.path.join(base_dir, "NGP SHP", "NGP_Zipcode_Shapes.shp")
    
    # Create output directory for clipped images
    clipped_dir = os.path.join(base_dir, "Clipped_Images")
    os.makedirs(clipped_dir, exist_ok=True)
    
    print("="*70)
    print("Starting Image Clipping Process with NGP Shapefile")
    print("="*70)
    print(f"Shapefile: {shapefile_path}")
    print(f"Output directory: {clipped_dir}")
    print()
    
    # Check if shapefile exists
    if not os.path.exists(shapefile_path):
        print(f"Error: Shapefile not found at {shapefile_path}")
        return
    
    # Define time periods to process
    time_periods = [
        "cold_dec-jan_2023",
        "cold_dec-jan_2024", 
        "hot_may_2023",
        "hot_may_2024"
    ]
    
    success_count = 0
    total_count = 0
    
    # Process each time period
    for period in time_periods:
        print(f"\nProcessing period: {period}")
        print("-" * 50)
        
        # Find the processed image for this period
        period_folder = os.path.join(processed_dir, f"{period}_processed")
        
        if not os.path.exists(period_folder):
            print(f"Processed folder not found: {period_folder}")
            continue
        
        # Find the merged image file
        image_pattern = os.path.join(period_folder, f"{period}_merged_joined_all_bands.tif")
        
        if not os.path.exists(image_pattern):
            print(f"Image not found: {image_pattern}")
            continue
        
        # Define output path for clipped image
        output_filename = f"{period}_clipped_NGP.tif"
        output_path = os.path.join(clipped_dir, output_filename)
        
        # Clip the image
        total_count += 1
        if clip_image_with_shapefile(image_pattern, shapefile_path, output_path):
            success_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("CLIPPING PROCESS COMPLETED")
    print("="*70)
    print(f"Total images processed: {total_count}")
    print(f"Successfully clipped: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    if success_count > 0:
        print(f"\nClipped images saved in: {clipped_dir}")
        print("\nOutput files:")
        for period in time_periods:
            output_file = os.path.join(clipped_dir, f"{period}_clipped_NGP.tif")
            if os.path.exists(output_file):
                print(f"  {os.path.basename(output_file)}")

def main():
    """Main function"""
    try:
        process_all_images()
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()
