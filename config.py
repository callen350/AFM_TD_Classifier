# Configuration file for AFM Pit Analysis GUI

import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- File Paths ---
AFM_FOLDER_PATH = resource_path("afm_data")
CSV_FOLDER_PATH = resource_path("results_csv")

# --- Model Paths ---
UNET_MODEL_PATH = "segmentation_model.keras"
LGR_MODEL_PATH = "pit_classifier_lr_unnormalised.joblib"

# Get absolute paths
UNET_MODEL_PATH_ABS = resource_path(UNET_MODEL_PATH)
LGR_MODEL_PATH_ABS = resource_path(LGR_MODEL_PATH)

# --- Model Settings ---
NORMALISED_MODEL = False  # Set to True if model expects normalized input
CIRCLE_PITS = True  # Set to True if pits should be circled instead of filled

# --- Zoom Settings ---
ZOOM_WINDOW_SIZE = 30  # 30x30 pixel region
ZOOM_WINDOW_HALF = ZOOM_WINDOW_SIZE // 2

# --- Pit Detection Settings ---
PIT_CLICK_RADIUS = 20  # Maximum distance in pixels to click on a pit
DEFAULT_PIT_DIAMETER = 10.0
DEFAULT_DEPTH = 10.0
FEATURE_EXTRACTION_WINDOW = 15

# --- Visualization Settings ---
COLORMAP = 'afmhot'
SMALL_PIT_COLOR = 'blue'
LARGE_PIT_COLOR = 'red'
FALSE_POSITIVE_COLOR = 'gray'
ADDED_PIT_COLOR = 'green'