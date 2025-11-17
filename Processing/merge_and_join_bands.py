"""
Script to merge Landsat bands and join row images
Process: 
1. Merge all bands (B1-B10) for each row (045, 046) separately
2. Join the merged images from both rows into a single image
3. Process all time periods: cold_dec-jan_2023, cold_dec-jan_2024, hot_may_2023, hot_may_2024
"""

import os
import glob
from pathlib import Path
import rasterio
from rasterio.merge import merge
import numpy as np

def merge_bands_for_row(row_folder):
    """
    Merge all bands (B1-B10) for a single row into one multi-band TIFF
    
    Args:
        row_folder (str): Path to the row folder containing band files
    
    Returns:
        str: Path to the merged file
    """
    print(f"Processing row: {row_folder}")
    
    # Get all band files (B1-B10, excluding QA)
    band_pattern = os.path.join(row_folder, "*_B[0-9]*.TIF")
    band_files = sorted(glob.glob(band_pattern))
    
    # Filter to get only B1-B10 (exclude B10 if it's thermal and you don't want it)
    band_files = [f for f in band_files if any(f"_B{i}.TIF" in f for i in range(1, 11))]
    
    if not band_files:
        print(f"No band files found in {row_folder}")
        return None
    
    print(f"Found {len(band_files)} band files")
    for bf in band_files:
        print(f"  - {os.path.basename(bf)}")
    
    # Read all bands and stack them
    bands_data = []
    profile = None
    
    for band_file in band_files:
        with rasterio.open(band_file) as src:
            if profile is None:
                profile = src.profile.copy()
            bands_data.append(src.read(1))
    
    # Update profile for multi-band output
    profile.update(count=len(bands_data))
    
    # Create output filename
    row_name = os.path.basename(row_folder)
    output_file = os.path.join(row_folder, f"{row_name}_merged_bands.tif")
    
    # Write merged bands
    with rasterio.open(output_file, 'w', **profile) as dst:
        for i, band_data in enumerate(bands_data, 1):
            dst.write(band_data, i)
    
    print(f"Merged bands saved to: {output_file}")
    return output_file

def join_row_images(merged_files, output_folder, period_name):
    """
    Join merged images from different rows into a single mosaic
    
    Args:
        merged_files (list): List of paths to merged row images
        output_folder (str): Path to output folder
        period_name (str): Name of the time period
    
    Returns:
        str: Path to the joined image
    """
    print(f"Joining images for period: {period_name}")
    
    if len(merged_files) < 2:
        print("Need at least 2 merged files to join")
        return None
    
    # Open all merged files
    src_files_to_mosaic = []
    for file in merged_files:
        src = rasterio.open(file)
        src_files_to_mosaic.append(src)
    
    # Merge the images
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Get profile from first image and update
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw"
    })
    
    # Create output filename
    output_file = os.path.join(output_folder, f"{period_name}_merged_joined_all_bands.tif")
    
    # Write the mosaic
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Close source files
    for src in src_files_to_mosaic:
        src.close()
    
    print(f"Joined image saved to: {output_file}")
    return output_file

def process_time_period(data_folder, period_name):
    """
    Process a single time period: merge bands for each row, then join rows
    
    Args:
        data_folder (str): Path to the data folder
        period_name (str): Name of the time period folder
    """
    print(f"\n{'='*60}")
    print(f"Processing time period: {period_name}")
    print(f"{'='*60}")
    
    period_folder = os.path.join(data_folder, period_name)
    
    if not os.path.exists(period_folder):
        print(f"Period folder not found: {period_folder}")
        return
    
    # Create output folder for this period
    output_folder = os.path.join(data_folder, f"{period_name}_processed")
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all row folders (should be 045 and 046)
    row_folders = [d for d in os.listdir(period_folder) 
                   if os.path.isdir(os.path.join(period_folder, d)) and ('045' in d or '046' in d)]
    
    print(f"Found row folders: {row_folders}")
    
    merged_files = []
    
    # Process each row folder
    for row_folder_name in sorted(row_folders):
        row_folder_path = os.path.join(period_folder, row_folder_name)
        merged_file = merge_bands_for_row(row_folder_path)
        if merged_file:
            merged_files.append(merged_file)
    
    # Join the merged row images
    if len(merged_files) >= 2:
        join_row_images(merged_files, output_folder, period_name)
    else:
        print(f"Not enough merged files to join for {period_name}")

def main():
    """Main function to process all time periods"""
    # Define the base data folder
    data_folder = r"Downloads\New folder\data"
    
    # Define time periods to process
    time_periods = [
        "cold_dec-jan_2023",
        "cold_dec-jan_2024", 
        "hot_may_2023",
        "hot_may_2024"
    ]
    
    print("Starting Landsat Band Merging and Joining Process")
    print(f"Data folder: {data_folder}")
    
    # Process each time period
    for period in time_periods:
        try:
            process_time_period(data_folder, period)
        except Exception as e:
            print(f"Error processing {period}: {str(e)}")
            continue
    
    print("\n" + "="*60)
    print("Processing completed!")
    print("="*60)
    
    # Show output folders created
    print("\nOutput folders created:")
    for period in time_periods:
        output_folder = os.path.join(data_folder, f"{period}_processed")
        if os.path.exists(output_folder):
            print(f"  - {output_folder}")

if __name__ == "__main__":
    main()
