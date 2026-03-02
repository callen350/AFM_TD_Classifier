# AFM Pit Segmenter & Classifier — GUI

Overview
--------
This GUI provides a lightweight interface for running a U-Net segmentation and a logistic-regression classifier on AFM (atomic force microscope) images, reviewing detected pits, and applying manual corrections (mark false positives, add pits, change classes, and resolve combined spots).

Requirements
------------
- Python 3.10+
- Packages: numpy, pandas, scikit-image, matplotlib, joblib, tensorflow (or keras), pySPM. Tkinter is used for the GUI (usually included with Python on Windows).

Quick install
-------------
1. Create and activate a virtual environment (recommended):

   python -m venv venv
   venv\Scripts\activate

2. Install required packages (example):

   pip install numpy pandas scikit-image matplotlib joblib tensorflow pyspm

Running the GUI
---------------
From the repository root run:

   python processing_gui_active_v2.py

This opens the main application window.

Models and data
---------------
- Place AFM images in the `afm_data/` folder (provided in the repo). Supported formats: `.npy` and `.spm`.

- The GUI expects two model files by default (see `config.py`):
  - `segmentation_model.keras` (U-Net for segmentation)
  - `pit_classifier_lr_unnormalised.joblib` (logistic regression classifier)

These paths are resolved via the `resource_path()` helper in `config.py`; when packaging with PyInstaller the function preserves bundled resource paths.

Main UI overview
----------------
- Load Image: Select a single `.npy` or `.spm` image.
- Random Image: Load a random image from `afm_data/`.
- Run Analysis: Runs segmentation (U-Net) then classification (LR) and displays results.
- Save Corrections: Saves any manual corrections (enabled once corrections exist).
- Hide/Show Labels: Toggle pit label visibility on visualizations.

Correction tools
----------------
Once an analysis has been run, enable the correction tools to inspect and edit detections:

- Mark False Positive: Click a detected pit on the image (or zoom window) to mark/unmark it as a false positive.
- Add Small Pit / Add Large Pit: Click on the image (or zoom window) to add a pit at that position. Feature values (depth, diameter, area, volume) are estimated using a small local window.
- Change Class: Toggle the class label between `Small` and `Large` for a selected pit.
- 🔍 Zoom: Opens a zoomed view around a clicked point. Use the zoom window to make precise corrections.
- Resolve Combined: Marks the clicked detection as a false positive and opens a zoomed view to add multiple pits in its place.
- Undo: Reverts the last correction action.

Zoom window behavior
--------------------
- The zoom window shows a fixed-size region (configured in `config.py` via `ZOOM_WINDOW_SIZE`).
- In resolve-combined mode you can repeatedly add multiple pits before closing the window.

Saving and exporting
--------------------
- Save Corrections will write corrected results to a CSV file. The CSV filename is based on the current image filename.
- The `save_data()` helper will drop internal `coords` column before writing.

Configurable settings
---------------------
Open `config.py` to tweak behavior and defaults such as:
- `AFM_FOLDER_PATH` — folder containing AFM images
- `UNET_MODEL_PATH` / `LGR_MODEL_PATH` — model filenames
- `NORMALISED_MODEL` — whether the segmentation/classifier expects normalized input
- Visualization colors and `CIRCLE_PITS` toggle
- `PIT_CLICK_RADIUS`, `DEFAULT_PIT_DIAMETER`, `DEFAULT_DEPTH`, `FEATURE_EXTRACTION_WINDOW` for feature extraction and clicking behavior

Troubleshooting
---------------
- If models fail to load: verify paths in `config.py` and that the model files exist in the repository root or packaged resources.
- If `.spm` support is required but missing, install `pyspm` and the file reader dependencies.
- If the GUI appears unresponsive during model loading or segmentation, run the script from a terminal to view printed debug messages.

Developer notes
---------------
- Main files:
  - `processing_gui_active_v2.py` — GUI and main app logic
  - `pit_analysis_core.py` — feature extraction and utilities
  - `correction_handlers.py` — correction operations (add/remove/modify/undo)
  - `zoom_window.py` — zoom and resolve-combined helpers
  - `config.py` — central configuration and resource helper

License
-------
This repository contains research code. Check with the project owner for redistribution or reuse terms.

--
Generated README for the AFM Pit GUI. If you want, I can also generate a `requirements.txt` or an example `models/` folder structure.
