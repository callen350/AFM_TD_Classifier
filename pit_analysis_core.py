# Core analysis functions for AFM pit segmentation and classification

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from config import FEATURE_EXTRACTION_WINDOW


def extract_pit_features_single(afm_img, mask, window=FEATURE_EXTRACTION_WINDOW):
    """
    Extract features for a single image/mask pair.
    
    Args:
        afm_img: 2D numpy array of AFM image
        mask: 2D binary mask of detected pits
        window: window size for depth calculation
        
    Returns:
        List of dictionaries containing pit features
    """
    features = []
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    for region in regions:
        y, x = region.centroid
        area = region.area
        diameter = 2 * np.sqrt(area / np.pi)
        
        # Depth calculation
        x_int, y_int = int(round(x)), int(round(y))
        half = window // 2
        h, w = afm_img.shape
        x1, x2 = max(0, x_int-half), min(w, x_int+half)
        y1, y2 = max(0, y_int-half), min(h, y_int+half)
        patch = afm_img[y1:y2, x1:x2]
        
        if patch.size > 0:
            depth = float(np.mean(patch) - np.min(patch))
        else:
            depth = 0.0
        
        volume = area * depth
        
        features.append({
            'centroid': (y, x),
            'coords': region.coords,
            'diameter': diameter,
            'depth': depth,
            'volume': volume,
            'area': area
        })
    return features


def calculate_pit_features(afm_img, x, y, window=FEATURE_EXTRACTION_WINDOW):
    """
    Calculate features for a single pit at given coordinates.
    
    Args:
        afm_img: 2D numpy array of AFM image
        x, y: coordinates of pit center
        window: window size for depth calculation
        
    Returns:
        Dictionary containing pit features (depth, diameter, volume, area)
    """
    h, w = afm_img.shape
    y_int, x_int = int(round(y)), int(round(x))
    
    half = window // 2
    x1, x2 = max(0, x_int-half), min(w, x_int+half)
    y1, y2 = max(0, y_int-half), min(h, y_int+half)
    patch = afm_img[y1:y2, x1:x2]
    
    if patch.size > 0:
        depth = float(np.mean(patch) - np.min(patch))
    else:
        from config import DEFAULT_DEPTH
        depth = DEFAULT_DEPTH
    
    from config import DEFAULT_PIT_DIAMETER
    diameter = DEFAULT_PIT_DIAMETER
    area = np.pi * (diameter / 2)**2
    volume = area * depth
    
    return {
        'diameter': diameter,
        'depth': depth,
        'volume': volume,
        'area': area
    }


def find_closest_pit(results_df, x, y, max_distance=None):
    """
    Find the closest pit to the given coordinates.
    
    Args:
        results_df: DataFrame containing pit information
        x, y: click coordinates
        max_distance: maximum distance to consider (default from config)
        
    Returns:
        Index of closest pit, or None if no pit within max_distance
    """
    if max_distance is None:
        from config import PIT_CLICK_RADIUS
        max_distance = PIT_CLICK_RADIUS
    
    min_dist = float('inf')
    closest_idx = None
    
    for idx, row in results_df.iterrows():
        cy, cx = row['centroid']
        dist = np.sqrt((cx - x)**2 + (cy - y)**2)
        if dist < min_dist and dist < max_distance:
            min_dist = dist
            closest_idx = idx
    
    return closest_idx


def create_corrected_dataframe(results_df, corrections):
    """
    Create a corrected DataFrame applying all corrections.
    
    Args:
        results_df: Original results DataFrame
        corrections: Dictionary containing all corrections
        
    Returns:
        Corrected DataFrame
    """
    updated_results = results_df.copy()
    
    # Remove false positives
    updated_results = updated_results[~updated_results.index.isin(corrections['false_positives'])]
    
    # Update modified classes
    for idx, new_class in corrections['modified_pits']:
        if idx in updated_results.index:
            updated_results.loc[idx, 'class_label'] = new_class
            updated_results.loc[idx, 'class_name'] = 'Small' if new_class == 1 else 'Large'
    
    # Add new pits
    for pit in corrections['added_pits']:
        pit_copy = pit.copy()
        pit_copy.pop('coords', None)  # Remove coords if it exists
        new_idx = updated_results.index.max() + 1 if len(updated_results) > 0 else 0
        updated_results.loc[new_idx] = pit_copy
    
    return updated_results