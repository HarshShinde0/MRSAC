"""
Landsat analysis with basic DN Scaling for UHI, LST, NDVI.
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pandas as pd

class SimpleLandsatAnalysis:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.clipped_dir = os.path.join(base_dir, "Clipped_Images")
        self.output_dir = os.path.join(base_dir, "Simple_Analysis_Results")
    
    def setup_directories(self):
        """Create output directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(self.clipped_dir) if f.endswith('.tif')]
        periods = [f.replace('_clipped_NGP.tif', '') for f in image_files]
        
        for period in periods:
            period_dir = os.path.join(self.output_dir, period)
            os.makedirs(period_dir, exist_ok=True)
    
    def simple_lst_calculation(self, thermal_dn):
        """
        Simple empirical LST calculation for realistic temperature mapping
        Maps thermal DN values directly to realistic temperature ranges
        """
        # Remove zero/nodata values
        thermal_clean = thermal_dn.astype(np.float64)
        thermal_clean = np.where(thermal_clean == 0, np.nan, thermal_clean)
        
        # Get valid thermal data for mapping
        valid_thermal = thermal_clean[~np.isnan(thermal_clean)]
        if len(valid_thermal) == 0:
            return thermal_clean
        
        # Map actual thermal DN ranges to realistic temperatures
        # Based on actual thermal DN ranges: ~7000-65000
        
        # More conservative percentile mapping to avoid extreme outliers
        min_dn = np.percentile(valid_thermal, 1)   # Coolest 1%
        max_dn = np.percentile(valid_thermal, 99)  # Hottest 1%
        
        # Determine temperature range based on season/thermal characteristics
        # Conservative mapping to Earth surface temperature ranges
        if np.mean(valid_thermal) < 35000:  # Cooler thermal signature
            temp_min, temp_max = 10, 45  # Winter range
        else:  # Warmer thermal signature
            temp_min, temp_max = 15, 50  # Summer range
        
        # Safe division to avoid issues
        dn_range = max_dn - min_dn
        if dn_range > 0:
            normalized = np.clip((thermal_clean - min_dn) / dn_range, 0, 1)
            lst_celsius = temp_min + (normalized * (temp_max - temp_min))
        else:
            lst_celsius = np.full_like(thermal_clean, (temp_min + temp_max) / 2)
        
        # Only remove clearly impossible values (not Earth surface temperatures)
        lst_celsius = np.where((lst_celsius < -50) | (lst_celsius > 80), np.nan, lst_celsius)
        
        return lst_celsius
    
    def simple_ndvi_calculation(self, red_dn, nir_dn):
        """Simple NDVI calculation from DN values"""
        # Convert to float and remove zeros
        red = red_dn.astype(np.float64)
        nir = nir_dn.astype(np.float64)
        
        # Set zeros to NaN
        red = np.where(red == 0, np.nan, red)
        nir = np.where(nir == 0, np.nan, nir)
        
        # Calculate NDVI
        denominator = nir + red
        ndvi = np.where(denominator > 0, (nir - red) / denominator, np.nan)
        
        # Clip to valid range
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi
    
    def simple_uhi_calculation(self, lst_data):
        """Improved UHI calculation with better rural reference"""
        valid_lst = lst_data[~np.isnan(lst_data)]
        
        if len(valid_lst) == 0:
            return np.full_like(lst_data, np.nan)
        
        # Use median as rural reference for more stable UHI calculation
        rural_temp = np.median(valid_lst)
        
        uhi = lst_data - rural_temp
        
        # Return real UHI data without any artificial clipping
        return uhi
    
    def create_simple_plot(self, data, title, output_path, cmap='viridis', units=''):
        """Create high-quality plots with proper color scaling"""
        plt.figure(figsize=(12, 8))
        
        # Mask invalid data
        plot_data = np.ma.masked_invalid(data)
        
        # Get valid data for better color scaling
        valid_data = data[~np.isnan(data)]
        
        # Set proper color ranges based on analysis type
        if 'LST' in title or 'Temperature' in title:
            # Fixed temperature range for better LST visualization
            if len(valid_data) > 0:
                vmin = max(15, np.percentile(valid_data, 2))
                vmax = min(60, np.percentile(valid_data, 98))
            else:
                vmin, vmax = 15, 50
        elif 'UHI' in title:
            # Centered range for UHI with proper scaling
            if len(valid_data) > 0:
                abs_max = max(abs(np.percentile(valid_data, 2)), abs(np.percentile(valid_data, 98)))
                vmin = -min(abs_max, 12)
                vmax = min(abs_max, 12)
            else:
                vmin, vmax = -8, 8
        else:
            # Keep original percentile-based scaling for NDVI and others
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, 5)
                vmax = np.percentile(valid_data, 95)
            else:
                vmin, vmax = None, None
        
        # Plot with better color scaling
        im = plt.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Add colorbar with units
        cbar = plt.colorbar(im, shrink=0.8, pad=0.02)
        if units:
            cbar.set_label(units, rotation=270, labelpad=15, fontsize=12)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        
        # Add statistics text with units
        if len(valid_data) > 0:
            if 'LST' in title or 'Temperature' in title:
                stats_text = f"Min: {valid_data.min():.1f}°C\\nMax: {valid_data.max():.1f}°C\\nMean: {valid_data.mean():.1f}°C"
            elif 'NDVI' in title:
                stats_text = f"Min: {valid_data.min():.3f}\\nMax: {valid_data.max():.3f}\\nMean: {valid_data.mean():.3f}"
            elif 'UHI' in title:
                stats_text = f"Min: {valid_data.min():.1f}°C\\nMax: {valid_data.max():.1f}°C\\nMean: {valid_data.mean():.1f}°C"
            else:
                stats_text = f"Min: {valid_data.min():.2f}\\nMax: {valid_data.max():.2f}\\nMean: {valid_data.mean():.2f}"
                
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                    verticalalignment='top', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    Saved: {os.path.basename(output_path)}")
    
    def save_outputs(self, data, profile, period, analysis_type):
        """Save TIF and PNG"""
        period_dir = os.path.join(self.output_dir, period)
        
        # Save TIF
        save_data = np.where(np.isnan(data), -9999, data)
        
        output_profile = profile.copy()
        output_profile.update({
            'count': 1,
            'dtype': 'float32',
            'nodata': -9999
        })
        
        tif_path = os.path.join(period_dir, f"{period}_{analysis_type}.tif")
        with rasterio.open(tif_path, 'w', **output_profile) as dst:
            dst.write(save_data.astype(np.float32), 1)
        print(f"    Saved: {os.path.basename(tif_path)}")
        
        # Save PNG with appropriate colormap
        png_path = os.path.join(period_dir, f"{period}_{analysis_type}.png")
        
        if analysis_type == "LST":
            title = f"Land Surface Temperature - {period}"
            cmap = 'viridis'  # Cool to warm temperature colormap
            units = 'Temperature (°C)'
        elif analysis_type == "UHI":
            title = f"Urban Heat Island - {period}"
            cmap = 'RdBu_r'  # Red for hot, blue for cool
            units = 'UHI Intensity (°C)'
        else:  # NDVI
            title = f"NDVI - {period}"
            cmap = 'RdYlGn'  # Keep the good NDVI colormap
            units = 'NDVI Index'
        
        self.create_simple_plot(data, title, png_path, cmap, units)
    
    def calculate_simple_stats(self, lst, ndvi, uhi):
        """Calculate basic statistics"""
        stats = {}
        
        # LST stats
        lst_valid = lst[~np.isnan(lst)]
        if len(lst_valid) > 0:
            stats['LST_min'] = np.min(lst_valid)
            stats['LST_max'] = np.max(lst_valid)
            stats['LST_mean'] = np.mean(lst_valid)
            stats['LST_std'] = np.std(lst_valid)
        else:
            stats.update({f'LST_{k}': np.nan for k in ['min', 'max', 'mean', 'std']})
        
        # NDVI stats
        ndvi_valid = ndvi[~np.isnan(ndvi)]
        if len(ndvi_valid) > 0:
            stats['NDVI_min'] = np.min(ndvi_valid)
            stats['NDVI_max'] = np.max(ndvi_valid)
            stats['NDVI_mean'] = np.mean(ndvi_valid)
            stats['NDVI_std'] = np.std(ndvi_valid)
        else:
            stats.update({f'NDVI_{k}': np.nan for k in ['min', 'max', 'mean', 'std']})
        
        # UHI stats
        uhi_valid = uhi[~np.isnan(uhi)]
        if len(uhi_valid) > 0:
            stats['UHI_min'] = np.min(uhi_valid)
            stats['UHI_max'] = np.max(uhi_valid)
            stats['UHI_mean'] = np.mean(uhi_valid)
            stats['UHI_std'] = np.std(uhi_valid)
        else:
            stats.update({f'UHI_{k}': np.nan for k in ['min', 'max', 'mean', 'std']})
        
        return stats
    
    def process_image(self, image_path, period):
        """Process single image"""
        print(f"\\nProcessing {period}...")
        
        with rasterio.open(image_path) as src:
            bands = src.read()
            profile = src.profile.copy()
            
            print(f"  Bands: {src.count}, Size: {src.width}x{src.height}")
            
            # Extract bands (0-indexed)
            red_dn = bands[3]      # Band 4 (Red)
            nir_dn = bands[4]      # Band 5 (NIR) 
            thermal_dn = bands[7]  # Band 10 (Thermal)
            
            print(f"  Red DN: {red_dn[red_dn>0].min()} - {red_dn.max()}")
            print(f"  NIR DN: {nir_dn[nir_dn>0].min()} - {nir_dn.max()}")
            print(f"  Thermal DN: {thermal_dn[thermal_dn>0].min()} - {thermal_dn.max()}")
            
            # Calculate indices
            print("  Calculating LST...")
            lst = self.simple_lst_calculation(thermal_dn)
            
            print("  Calculating NDVI...")
            ndvi = self.simple_ndvi_calculation(red_dn, nir_dn)
            
            print("  Calculating UHI...")
            uhi = self.simple_uhi_calculation(lst)
            
            # Check results
            lst_valid = lst[~np.isnan(lst)]
            ndvi_valid = ndvi[~np.isnan(ndvi)]
            uhi_valid = uhi[~np.isnan(uhi)]
            
            if len(lst_valid) > 0:
                print(f"  LST: {lst_valid.min():.1f} - {lst_valid.max():.1f}°C (avg: {lst_valid.mean():.1f}°C)")
            if len(ndvi_valid) > 0:
                print(f"  NDVI: {ndvi_valid.min():.3f} - {ndvi_valid.max():.3f} (avg: {ndvi_valid.mean():.3f})")
            if len(uhi_valid) > 0:
                print(f"  UHI: {uhi_valid.min():.1f} - {uhi_valid.max():.1f}°C (avg: {uhi_valid.mean():.1f}°C)")
            
            # Save outputs
            print("  Saving outputs...")
            self.save_outputs(lst, profile, period, "LST")
            self.save_outputs(ndvi, profile, period, "NDVI")
            self.save_outputs(uhi, profile, period, "UHI")
            
            # Calculate stats
            stats = self.calculate_simple_stats(lst, ndvi, uhi)
            stats['Period'] = period
            
            # Save stats
            stats_df = pd.DataFrame([stats])
            stats_path = os.path.join(self.output_dir, period, f"{period}_statistics.csv")
            stats_df.to_csv(stats_path, index=False, float_format='%.3f')
            print(f"    Saved: {os.path.basename(stats_path)}")
            
            return stats
    
    def run_analysis(self):
        """Main analysis function"""
        print("="*70)
        print("SIMPLE LANDSAT ANALYSIS")
        print("="*70)
        
        self.setup_directories()
        
        # Find images
        image_files = [f for f in os.listdir(self.clipped_dir) if f.endswith('.tif')]
        
        if not image_files:
            print("No images found!")
            return
        
        print(f"Found {len(image_files)} images")
        
        all_stats = {}
        
        # Process each image
        for image_file in sorted(image_files):
            period = image_file.replace('_clipped_NGP.tif', '')
            image_path = os.path.join(self.clipped_dir, image_file)
            
            try:
                stats = self.process_image(image_path, period)
                all_stats[period] = stats
            except Exception as e:
                print(f"ERROR processing {period}: {e}")
                continue
        
        # Save summary
        if all_stats:
            summary_df = pd.DataFrame(list(all_stats.values()))
            summary_path = os.path.join(self.output_dir, "Analysis_Summary.csv")
            summary_df.to_csv(summary_path, index=False, float_format='%.3f')
            print(f"\\nSummary saved: {summary_path}")
        
        print("\\n" + "="*70)
        print("ANALYSIS COMPLETED")
        print("="*70)
        print(f"Results in: {self.output_dir}")

def main():
    base_dir = r"Downloads\New folder"
    analyzer = SimpleLandsatAnalysis(base_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
