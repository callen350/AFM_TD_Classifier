# Correction mode handlers for AFM pit analysis GUI

import numpy as np
from config import (
    DEFAULT_PIT_DIAMETER, DEFAULT_DEPTH,
    FEATURE_EXTRACTION_WINDOW
)
from pit_analysis_core import calculate_pit_features, find_closest_pit


class CorrectionHandler:
    """Handles all correction operations for the pit analysis GUI."""
    
    def __init__(self, app):
        """
        Initialize correction handler.
        
        Args:
            app: Reference to main PitAnalysisApp instance
        """
        self.app = app
    
    def mark_false_positive(self, x, y):
        """Mark a pit as false positive."""
        closest_idx = find_closest_pit(self.app.results_df, x, y)
        
        if closest_idx is not None:
            if closest_idx not in self.app.corrections['false_positives']:
                self.app.corrections['false_positives'].append(closest_idx)
                self.app.correction_history.append(('false_positive_add', closest_idx))
                self.app.status_label.config(
                    text=f"Marked pit {closest_idx} as false positive. Total FP: {len(self.app.corrections['false_positives'])}"
                )
                self.app.redraw_results()
            else:
                # Undo if clicking again
                self.app.corrections['false_positives'].remove(closest_idx)
                self.app.correction_history.append(('false_positive_remove', closest_idx))
                self.app.status_label.config(
                    text=f"Unmarked pit {closest_idx}. Total FP: {len(self.app.corrections['false_positives'])}"
                )
                self.app.redraw_results()
    
    def add_pit(self, x, y, mode):
        """Add a new pit at the clicked location."""
        h, w = self.app.current_image.shape
        y_int, x_int = int(round(y)), int(round(x))
        
        if 0 <= x_int < w and 0 <= y_int < h:
            # Calculate features for the new pit
            features = calculate_pit_features(self.app.current_image, x, y)
            
            new_class = 1 if mode == 'add_small' else 2
            new_class_name = 'Small' if mode == 'add_small' else 'Large'
            
            new_pit = {
                'centroid': (y_int, x_int),
                'diameter': features['diameter'],
                'depth': features['depth'],
                'volume': features['volume'],
                'area': features['area'],
                'class_label': new_class,
                'class_name': new_class_name
            }
            
            self.app.corrections['added_pits'].append(new_pit)
            self.app.correction_history.append(('added_pit', len(self.app.corrections['added_pits']) - 1))
            self.app.status_label.config(
                text=f"Added {new_class_name} pit at ({x_int}, {y_int}). Total added: {len(self.app.corrections['added_pits'])}"
            )
            self.app.redraw_results()
    
    def modify_pit_class(self, x, y):
        """Change the class of an existing pit."""
        closest_idx = find_closest_pit(self.app.results_df, x, y)
        
        if closest_idx is not None:
            current_class = self.app.results_df.loc[closest_idx, 'class_label']
            new_class = 2 if current_class == 1 else 1
            
            # Check if already modified
            existing_mod = None
            for i, (idx, _) in enumerate(self.app.corrections['modified_pits']):
                if idx == closest_idx:
                    existing_mod = i
                    break
            
            if existing_mod is not None:
                # Remove existing modification
                old_entry = self.app.corrections['modified_pits'].pop(existing_mod)
                self.app.correction_history.append(('modified_pit_remove', old_entry))
            else:
                # Add new modification
                self.app.corrections['modified_pits'].append((closest_idx, new_class))
                self.app.correction_history.append(('modified_pit_add', (closest_idx, new_class)))
            
            self.app.status_label.config(
                text=f"Changed pit {closest_idx} to class {new_class}. Total modified: {len(self.app.corrections['modified_pits'])}"
            )
            self.app.redraw_results()
    
    def undo_last_correction(self):
        """Undo the last correction made."""
        if not self.app.correction_history:
            self.app.status_label.config(text="No corrections to undo.")
            return
        
        last_action = self.app.correction_history.pop()
        action_type, data = last_action
        
        if action_type == 'false_positive_add':
            self.app.corrections['false_positives'].remove(data)
            self.app.status_label.config(text=f"Undid false positive marking for pit {data}.")
        
        elif action_type == 'false_positive_remove':
            self.app.corrections['false_positives'].append(data)
            self.app.status_label.config(text=f"Undid false positive unmarking for pit {data}.")
        
        elif action_type == 'added_pit':
            self.app.corrections['added_pits'].pop(data)
            self.app.status_label.config(text=f"Undid pit addition.")
        
        elif action_type == 'modified_pit_add':
            idx, _ = data
            for i, (mod_idx, _) in enumerate(self.app.corrections['modified_pits']):
                if mod_idx == idx:
                    self.app.corrections['modified_pits'].pop(i)
                    break
            self.app.status_label.config(text=f"Undid class modification for pit {idx}.")
        
        elif action_type == 'modified_pit_remove':
            self.app.corrections['modified_pits'].append(data)
            self.app.status_label.config(text=f"Undid class modification removal for pit {data[0]}.")
        
        self.app.redraw_results()