import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from rtree import index
import time

def load_polygons_from_csv(csv_path):
    """
    Load polygons from a CSV file containing x,y coordinates.
    Returns a list of Shapely Polygon objects.
    """
    print(f"Loading data from {csv_path}...")
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"CSV columns: {df.columns.tolist()}")
        print(f"First few rows: \n{df.head()}")
        
        # Check if X and Y columns exist
        if 'X' not in df.columns or 'Y' not in df.columns:
            print("ERROR: CSV is missing X or Y columns!")
            if 'X' not in df.columns and ' X' in df.columns:
                print("Renaming ' X' to 'X'")
                df.rename(columns={' X': 'X'}, inplace=True)
            if 'Y' not in df.columns and ' Y' in df.columns:
                print("Renaming ' Y' to 'Y'")
                df.rename(columns={' Y': 'Y'}, inplace=True)
        
        # Check if ROI column exists
        if 'ROI' not in df.columns:
            print("ERROR: CSV is missing ROI column!")
            if ' ROI' in df.columns:
                print("Renaming ' ROI' to 'ROI'")
                df.rename(columns={' ROI': 'ROI'}, inplace=True)
        
        # Get a list of unique ROIs
        unique_rois = df['ROI'].unique()
        print(f"Found {len(unique_rois)} unique ROIs.")
        
        # Create a list to store polygons
        polygons = []
        invalid_polygons = 0
        not_enough_points = 0
        
        # For each ROI, extract its coordinates and create a polygon
        for roi in tqdm(unique_rois, desc="Creating polygons"):
            # Get points for this ROI
            roi_points = df[df['ROI'] == roi][['X', 'Y']].values
            
            # Create a Shapely polygon (if there are at least 3 points)
            if len(roi_points) >= 3:
                try:
                    poly = Polygon(roi_points)
                    if poly.is_valid:
                        polygons.append(poly)
                    else:
                        print(f"WARNING: Invalid polygon for ROI {roi} with {len(roi_points)} points")
                        # Try to fix the polygon
                        from shapely.validation import make_valid
                        fixed_poly = make_valid(poly)
                        if fixed_poly.geom_type == 'Polygon':
                            print(f"  - Successfully fixed polygon for ROI {roi}")
                            polygons.append(fixed_poly)
                        elif fixed_poly.geom_type == 'MultiPolygon':
                            print(f"  - Fixed polygon for ROI {roi} is a MultiPolygon, using largest part")
                            # Take the largest polygon from the multipolygon
                            largest_poly = max(fixed_poly.geoms, key=lambda p: p.area)
                            if largest_poly.is_valid:
                                polygons.append(largest_poly)
                                print(f"  - Successfully added largest part of MultiPolygon for ROI {roi}")
                            else:
                                print(f"  - Largest part is still invalid for ROI {roi}")
                                invalid_polygons += 1
                        else:
                            print(f"  - Could not fix polygon for ROI {roi} (got {fixed_poly.geom_type})")
                            invalid_polygons += 1
                except Exception as e:
                    print(f"ERROR creating polygon for ROI {roi}: {e}")
                    invalid_polygons += 1
            else:
                print(f"Not enough points for ROI {roi}: {len(roi_points)} points")
                not_enough_points += 1
        
        print(f"Created {len(polygons)} valid polygons.")
        print(f"Skipped {invalid_polygons} invalid polygons and {not_enough_points} ROIs with too few points.")
        
        # Check if we have any valid polygons
        if len(polygons) == 0:
            print("WARNING: No valid polygons were created!")
            
        return polygons
        
    except Exception as e:
        print(f"ERROR loading CSV file: {e}")
        return []

def find_best_spatial_match(polygon, other_polygons):
    """
    Find the polygon in other_polygons that has the highest spatial overlap with the given polygon.
    Returns the index of the best match and the DICE score.
    """
    best_match = -1
    best_dice = 0.0
    
    for i, other_poly in enumerate(other_polygons):
        try:
            # Calculate intersection
            intersection = polygon.intersection(other_poly).area
            dice = 2 * intersection / (polygon.area + other_poly.area)
            
            # Update best match if this is better
            if dice > best_dice:
                best_dice = dice
                best_match = i
        except Exception as e:
            # Skip any problematic polygons
            continue
    
    return best_match, best_dice

def create_spatial_index(polygons):
    """
    Create an R-tree spatial index for faster spatial querying.
    """
    idx = index.Index()
    for i, poly in enumerate(polygons):
        idx.insert(i, poly.bounds)
    return idx

def find_overlapping_pairs(polygons1, polygons2, threshold=0.0):
    """
    Find all pairs of polygons that overlap between the two sets.
    Uses spatial indexing for efficiency.
    Returns a list of tuples (index1, index2, dice_score).
    """
    print("Creating spatial index for second polygon set...")
    spatial_idx = create_spatial_index(polygons2)
    
    pairs = []
    print("Finding overlapping polygon pairs...")
    for i, poly1 in tqdm(enumerate(polygons1), total=len(polygons1), desc="Comparing polygons"):
        # Query the spatial index to find potential matches
        potential_matches = list(spatial_idx.intersection(poly1.bounds))
        
        # Check each potential match
        for j in potential_matches:
            poly2 = polygons2[j]
            
            try:
                # Calculate DICE score
                intersection = poly1.intersection(poly2).area
                union = poly1.area + poly2.area
                
                if union > 0:
                    dice = 2 * intersection / union
                    
                    # Add to pairs if above threshold
                    if dice > threshold:
                        pairs.append((i, j, dice))
            except Exception as e:
                continue
    
    print(f"Found {len(pairs)} overlapping polygon pairs with DICE > {threshold}")
    return pairs

def calculate_global_dice(polygons1, polygons2):
    """
    Calculate a global DICE score treating all polygons in each set as a single geometry.
    
    Returns a dictionary with different variations of global DICE scores.
    """
    global_metrics = {}
    
    try:
        # Method 1: Combine all polygons in each set into a single geometry
        print("Calculating global DICE score (Method 1: Union of all polygons)...")
        from shapely.ops import unary_union
        
        combined1 = unary_union(polygons1)
        combined2 = unary_union(polygons2)
        
        # Calculate global DICE score using the union method
        intersection_area = combined1.intersection(combined2).area
        union_area = combined1.area + combined2.area
        
        if union_area > 0:
            dice_union = 2 * intersection_area / union_area
            global_metrics['global_dice_union'] = dice_union
            print(f"Global DICE (Union method): {dice_union:.4f}")
        else:
            print("Warning: Sum of areas is zero, cannot calculate union-based DICE score")
            global_metrics['global_dice_union'] = 0.0
    
    except Exception as e:
        print(f"Error calculating global DICE using union method: {e}")
        global_metrics['global_dice_union'] = np.nan
    
    try:
        # Method 2: Sum of all pairwise intersections
        print("Calculating global DICE score (Method 2: Sum of all pairwise intersections)...")
        total_intersection = 0
        total_area1 = sum(p.area for p in polygons1)
        total_area2 = sum(p.area for p in polygons2)
        
        # Create spatial index for faster intersection queries
        from rtree import index
        idx = index.Index()
        for i, poly in enumerate(polygons2):
            idx.insert(i, poly.bounds)
        
        # Calculate sum of all intersections
        for i, poly1 in enumerate(polygons1):
            potential_matches = list(idx.intersection(poly1.bounds))
            for j in potential_matches:
                poly2 = polygons2[j]
                if poly1.intersects(poly2):
                    total_intersection += poly1.intersection(poly2).area
        
        # Calculate DICE using sum of intersections
        if (total_area1 + total_area2) > 0:
            dice_sum = 2 * total_intersection / (total_area1 + total_area2)
            global_metrics['global_dice_sum'] = dice_sum
            print(f"Global DICE (Sum method): {dice_sum:.4f}")
        else:
            print("Warning: Sum of areas is zero, cannot calculate sum-based DICE score")
            global_metrics['global_dice_sum'] = 0.0
    
    except Exception as e:
        print(f"Error calculating global DICE using sum method: {e}")
        global_metrics['global_dice_sum'] = np.nan
    
    # Return all metrics
    return global_metrics

def visualize_best_matches(polygons1, polygons2, pairs, output_path, sample_size=5):
    """
    Visualize a sample of the best matching polygon pairs.
    """
    # Sort pairs by DICE score in descending order
    sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    
    # Limit to sample size
    sample_pairs = sorted_pairs[:sample_size]
    
    # Create figure
    fig, axes = plt.subplots(len(sample_pairs), 3, figsize=(15, 5*len(sample_pairs)))
    
    # Handle case with only one pair
    if len(sample_pairs) == 1:
        axes = np.array([axes])
    
    for i, (idx1, idx2, dice) in enumerate(sample_pairs):
        poly1 = polygons1[idx1]
        poly2 = polygons2[idx2]
        
        # Plot first polygon
        ax1 = axes[i, 0]
        x1, y1 = poly1.exterior.xy
        ax1.plot(x1, y1, 'b-')
        ax1.set_title(f"Set 1 - Polygon {idx1}")
        ax1.axis('equal')
        
        # Plot second polygon
        ax2 = axes[i, 1]
        x2, y2 = poly2.exterior.xy
        ax2.plot(x2, y2, 'r-')
        ax2.set_title(f"Set 2 - Polygon {idx2}")
        ax2.axis('equal')
        
        # Plot overlap
        ax3 = axes[i, 2]
        ax3.plot(x1, y1, 'b-', alpha=0.5)
        ax3.plot(x2, y2, 'r-', alpha=0.5)
        
        # Plot intersection if it exists
        try:
            intersection = poly1.intersection(poly2)
            if intersection.area > 0:
                if intersection.geom_type == 'Polygon':
                    x_int, y_int = intersection.exterior.xy
                    ax3.fill(x_int, y_int, 'g', alpha=0.3)
                elif intersection.geom_type == 'MultiPolygon':
                    for geom in intersection.geoms:
                        x_int, y_int = geom.exterior.xy
                        ax3.fill(x_int, y_int, 'g', alpha=0.3)
        except Exception as e:
            pass
            
        ax3.set_title(f"Overlap - DICE: {dice:.4f}")
        ax3.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")

def main(file1_path, file2_path, output_dir="results"):
    """
    Main function to compare polygons from two CSV files based on spatial overlap.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at: {output_dir}")
    
    # Load polygons from both files
    print("Processing first file...")
    polygons1 = load_polygons_from_csv(file1_path)
    
    if len(polygons1) == 0:
        print("No valid polygons found in the first file. Cannot continue.")
        return None, None, None
    
    print("\nProcessing second file...")
    polygons2 = load_polygons_from_csv(file2_path)
    
    if len(polygons2) == 0:
        print("No valid polygons found in the second file. Cannot continue.")
        return None, None, None
    
    # Calculate global DICE scores first
    print("\nCalculating global DICE scores...")
    try:
        global_metrics = calculate_global_dice(polygons1, polygons2)
        
        # Save global metrics to CSV
        metrics_df = pd.DataFrame([
            {'Metric': 'Global DICE (Union Method)', 'Value': global_metrics.get('global_dice_union', np.nan)},
            {'Metric': 'Global DICE (Sum Method)', 'Value': global_metrics.get('global_dice_sum', np.nan)}
        ])
        metrics_df.to_csv(os.path.join(output_dir, "global_metrics.csv"), index=False)
        print(f"Global metrics saved to {os.path.join(output_dir, 'global_metrics.csv')}")
        
        # Create a bar chart for global metrics
        plt.figure(figsize=(10, 6))
        metrics = list(metrics_df['Metric'])
        values = list(metrics_df['Value'])
        
        # Check if we have valid values
        valid_metrics = []
        valid_values = []
        for m, v in zip(metrics, values):
            if not np.isnan(v):
                valid_metrics.append(m)
                valid_values.append(v)
        
        if valid_values:
            bars = plt.bar(valid_metrics, valid_values, color=['blue', 'green'])
            plt.ylim(0, 1)  # DICE scores range from 0 to 1
            plt.ylabel('DICE Score')
            plt.title('Global DICE Scores')
            
            # Add value labels on the bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            global_metrics_path = os.path.join(output_dir, "global_metrics.png")
            plt.savefig(global_metrics_path)
            plt.close()
            print(f"Global metrics visualization saved to {global_metrics_path}")
    except Exception as e:
        print(f"Error processing global DICE scores: {e}")
    
    # Find all overlapping polygon pairs
    print("\nFinding overlapping polygon pairs...")
    start_time = time.time()
    overlap_pairs = find_overlapping_pairs(polygons1, polygons2, threshold=0.01)
    end_time = time.time()
    print(f"Pair finding completed in {end_time - start_time:.2f} seconds")
    
    if len(overlap_pairs) == 0:
        print("No overlapping polygons found between the two files.")
        return None, None, None
    
    # Calculate overall statistics
    dice_scores = [score for _, _, score in overlap_pairs]
    mean_dice = np.mean(dice_scores)
    median_dice = np.median(dice_scores)
    std_dice = np.std(dice_scores)
    
    print(f"\nPairwise DICE Score Summary:")
    print(f"Number of overlapping pairs: {len(overlap_pairs)}")
    print(f"Mean DICE: {mean_dice:.4f}")
    print(f"Median DICE: {median_dice:.4f}")
    print(f"Standard Deviation: {std_dice:.4f}")
    
    # Save overlap pairs to CSV
    pairs_df = pd.DataFrame([
        {'Polygon1_Index': idx1, 'Polygon2_Index': idx2, 'DICE': score}
        for idx1, idx2, score in overlap_pairs
    ])
    pairs_output_path = os.path.join(output_dir, "overlap_pairs.csv")
    pairs_df.to_csv(pairs_output_path, index=False)
    print(f"Overlap pairs saved to {pairs_output_path}")
    
    # Create histogram of DICE scores
    plt.figure(figsize=(10, 6))
    plt.hist(dice_scores, bins=20, alpha=0.7)
    plt.axvline(mean_dice, color='r', linestyle='--', label=f'Mean: {mean_dice:.4f}')
    plt.axvline(median_dice, color='g', linestyle='--', label=f'Median: {median_dice:.4f}')
    plt.title("Distribution of Pairwise DICE Scores")
    plt.xlabel("DICE Score")
    plt.ylabel("Frequency")
    plt.legend()
    hist_output_path = os.path.join(output_dir, "dice_histogram.png")
    plt.savefig(hist_output_path)
    plt.close()
    print(f"Histogram saved to {hist_output_path}")
    
    # Create summary report with all metrics
    summary_df = pd.DataFrame([
        {'Metric': 'Global DICE (Union Method)', 'Value': global_metrics.get('global_dice_union', np.nan)},
        {'Metric': 'Global DICE (Sum Method)', 'Value': global_metrics.get('global_dice_sum', np.nan)},
        {'Metric': 'Mean Pairwise DICE', 'Value': mean_dice},
        {'Metric': 'Median Pairwise DICE', 'Value': median_dice},
        {'Metric': 'Pairwise DICE Std Dev', 'Value': std_dice},
        {'Metric': 'Number of Overlapping Pairs', 'Value': len(overlap_pairs)},
        {'Metric': 'Total Polygons in Set 1', 'Value': len(polygons1)},
        {'Metric': 'Total Polygons in Set 2', 'Value': len(polygons2)}
    ])
    summary_path = os.path.join(output_dir, "dice_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary report saved to {summary_path}")
    
    # Visualize best matches
    vis_output_path = os.path.join(output_dir, "best_matches.png")
    visualize_best_matches(polygons1, polygons2, overlap_pairs, vis_output_path)
    
    return overlap_pairs, mean_dice, median_dice

if __name__ == "__main__":
    # Get the current directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script is running from: {script_dir}")
    
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Try different filenames that might match your files
    possible_file1_names = [
        "66H-Cx-neuropil1resultsMG.csv",
        "66H-Cx-neuropil-1resultsMG.csv",
        "66H-Cx-neuropil1-resultsMG.csv",
        "66HCxneuropil1resultsMG.csv",
        "66H-Cx-neuropil 1resultsMG.csv"
    ]
    
    possible_file2_names = [
        "66H-Cx-neuropil-1-XY-results.csv",
        "66H-Cx-neuropil1-XY-results.csv",
        "66H-Cx-neuropil-1-XY-results.csv",
        "66HCxneuropil1XYresults.csv",
        "66H-Cx-neuropil 1-XY-results.csv"
    ]
    
    # Find the actual files
    file1_path = None
    file2_path = None
    
    print("Searching for file 1...")
    for filename in possible_file1_names:
        potential_path = os.path.join(script_dir, filename)
        print(f"  Checking: {potential_path}")
        if os.path.exists(potential_path):
            file1_path = potential_path
            print(f"  FOUND: {file1_path}")
            break
    
    print("Searching for file 2...")
    for filename in possible_file2_names:
        potential_path = os.path.join(script_dir, filename)
        print(f"  Checking: {potential_path}")
        if os.path.exists(potential_path):
            file2_path = potential_path
            print(f"  FOUND: {file2_path}")
            break
    
    # If files weren't found in the script directory, look in the current working directory
    if file1_path is None or file2_path is None:
        print("Files not found in script directory, checking current working directory...")
        cwd = os.getcwd()
        
        # Try to list files in the current directory to see what's available
        try:
            files_in_dir = os.listdir(cwd)
            print(f"Files in current directory:")
            for file in files_in_dir:
                if file.endswith('.csv'):
                    print(f"  {file}")
                    
            # Check for CSV files with similar names
            if file1_path is None:
                for file in files_in_dir:
                    if 'neuropil' in file.lower() and 'result' in file.lower() and not 'xy' in file.lower() and file.endswith('.csv'):
                        file1_path = os.path.join(cwd, file)
                        print(f"  Potential match for file 1: {file1_path}")
                        break
                        
            if file2_path is None:
                for file in files_in_dir:
                    if 'neuropil' in file.lower() and 'xy' in file.lower() and file.endswith('.csv'):
                        file2_path = os.path.join(cwd, file)
                        print(f"  Potential match for file 2: {file2_path}")
                        break
        except Exception as e:
            print(f"Error listing directory contents: {e}")
    
    # If we still can't find the files, prompt the user
    if file1_path is None:
        print("\nERROR: Could not find file 1. Please enter the full path to the file:")
        file1_path = input("> ").strip()
    
    if file2_path is None:
        print("\nERROR: Could not find file 2. Please enter the full path to the file:")
        file2_path = input("> ").strip()
    
    # Create an output directory in the same folder as the script
    output_dir = os.path.join(script_dir, "results")
    
    # Run main function if files exist
    if os.path.exists(file1_path) and os.path.exists(file2_path):
        print(f"\nAnalyzing files:")
        print(f"  File 1: {file1_path}")
        print(f"  File 2: {file2_path}")
        print(f"  Output directory: {output_dir}")
        
        main(file1_path, file2_path, output_dir)
    else:
        print("Cannot run analysis because one or both files were not found.")