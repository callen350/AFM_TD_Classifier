import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
from tensorflow import keras
import os
import glob
import random
import copy
import pySPM

# Import custom modules
from config import *
from pit_analysis_core import extract_pit_features_single, create_corrected_dataframe
from correction_handlers import CorrectionHandler
from zoom_window import ZoomWindowManager

# Import custom loss function
try:
    from unet_functions import LOSS_FUNCTIONS
    dice_focal_loss = LOSS_FUNCTIONS["dice_focal"]
except ImportError:
    def dice_focal_loss(y_true, y_pred): return 0


class PitAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AFM Pit Segmenter & Classifier")
        self.root.geometry("1200x800")
        
        # Model paths
        self.unet_path = UNET_MODEL_PATH_ABS
        self.lr_path = LGR_MODEL_PATH_ABS
        
        # Models
        self.unet_model = None
        self.lr_model = None
        
        # Data
        self.current_image = None
        self.current_filename = None
        self.results_df = None
        self.binary_mask = None
        
        # Correction tracking
        self.corrections = {
            'false_positives': [],
            'added_pits': [],
            'modified_pits': []
        }
        self.correction_history = []
        self.correction_mode = None
        self.show_labels = True
        
        # Visualization
        self.fig_ax = None
        self.canvas_widget = None
        
        # Initialize handlers
        self.correction_handler = CorrectionHandler(self)
        self.zoom_manager = ZoomWindowManager(self)
        
        # Build GUI
        self._build_gui()
        
        # Load models last
        self.load_models()
    
    def _build_gui(self):
        """Build the GUI layout."""
        # Top Control Panel - Main Container
        control_container = tk.Frame(self.root, padx=10, pady=5, bg="#e1e1e1")
        control_container.pack(fill=tk.X)
        
        # Row 1: Main operation buttons
        row1_frame = tk.Frame(control_container, bg="#e1e1e1")
        row1_frame.pack(fill=tk.X, pady=(0, 5))
        
        btn_load = tk.Button(row1_frame, text="Load Image", command=self.load_image, bg="white", height=2)
        btn_load.pack(side=tk.LEFT, padx=5)
        
        btn_random = tk.Button(row1_frame, text="Random Image", command=self.load_random_image, bg="#f0f0f0", height=2)
        btn_random.pack(side=tk.LEFT, padx=5)
        
        btn_run = tk.Button(row1_frame, text="Run Analysis", command=self.run_analysis, bg="#d1ffbd", height=2)
        btn_run.pack(side=tk.LEFT, padx=5)
        
        self.btn_save_corrections = tk.Button(row1_frame, text="Save Corrections", 
                                               command=self.save_corrections, 
                                               bg="#bdffd1", height=2, state=tk.DISABLED)
        self.btn_save_corrections.pack(side=tk.LEFT, padx=5)
        
        self.btn_toggle_labels = tk.Button(row1_frame, text="Hide Labels", 
                                            command=self.toggle_labels, 
                                            bg="#e1d1ff", height=2, state=tk.DISABLED)
        self.btn_toggle_labels.pack(side=tk.LEFT, padx=5)
        
        # Status label on right side of row 1
        self.status_label = tk.Label(row1_frame, text="Status: Ready", bg="#e1e1e1")
        self.status_label.pack(side=tk.RIGHT, padx=20)
        
        # Row 2: Correction tools
        row2_frame = tk.Frame(control_container, bg="#e1e1e1")
        row2_frame.pack(fill=tk.X)
        
        self.btn_remove = tk.Button(row2_frame, text="Mark False Positive", 
                                     command=lambda: self.set_correction_mode('remove'), 
                                     bg="#ffcccc", height=2, state=tk.DISABLED)
        self.btn_remove.pack(side=tk.LEFT, padx=5)
        
        self.btn_add_small = tk.Button(row2_frame, text="Add Small Pit", 
                                        command=lambda: self.set_correction_mode('add_small'), 
                                        bg="#ccccff", height=2, state=tk.DISABLED)
        self.btn_add_small.pack(side=tk.LEFT, padx=5)
        
        self.btn_add_large = tk.Button(row2_frame, text="Add Large Pit", 
                                        command=lambda: self.set_correction_mode('add_large'), 
                                        bg="#ffcccc", height=2, state=tk.DISABLED)
        self.btn_add_large.pack(side=tk.LEFT, padx=5)
        
        self.btn_modify = tk.Button(row2_frame, text="Change Class", 
                                     command=lambda: self.set_correction_mode('modify'), 
                                     bg="#ffffcc", height=2, state=tk.DISABLED)
        self.btn_modify.pack(side=tk.LEFT, padx=5)
        
        self.btn_zoom = tk.Button(row2_frame, text="🔍 Zoom", 
                                   command=lambda: self.set_correction_mode('zoom'), 
                                   bg="#e0e0e0", height=2, state=tk.DISABLED)
        self.btn_zoom.pack(side=tk.LEFT, padx=5)
        
        self.btn_resolve_combined = tk.Button(row2_frame, text="Resolve Combined", 
                                               command=lambda: self.set_correction_mode('resolve_combined'), 
                                               bg="#ffd4b3", height=2, state=tk.DISABLED)
        self.btn_resolve_combined.pack(side=tk.LEFT, padx=5)
        
        self.btn_undo = tk.Button(row2_frame, text="Undo", 
                                   command=self.correction_handler.undo_last_correction, 
                                   bg="#ffd1bd", height=2, state=tk.DISABLED)
        self.btn_undo.pack(side=tk.LEFT, padx=5)
        
        # Remove the old btn_correct_frame reference since we're not using grid layout anymore
        self.btn_correct_frame = None
        
        # Main Content Area
        self.content_frame = tk.Frame(self.root)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Image Display
        self.canvas_frame = tk.Frame(self.content_frame, bg="white", width=600)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right: Stats Display
        self.stats_frame = tk.Frame(self.content_frame, bg="white", width=400)
        self.stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def load_models(self):
        """Load U-Net and Logistic Regression models."""
        print("--- DEBUG: Starting Model Load ---")
        
        # Check U-Net
        print(f"Checking U-Net path: {self.unet_path}")
        if not os.path.exists(self.unet_path):
            messagebox.showerror("Error", f"U-Net file missing: {self.unet_path}")
            return
        
        try:
            print("Attempting to load U-Net model...")
            self.unet_model = keras.models.load_model(self.unet_path, compile=False, 
                                                     custom_objects={"dice_focal_loss": dice_focal_loss})
            print("SUCCESS: U-Net loaded.")
        except Exception as e:
            print(f"FAIL: U-Net crashed. Error: {e}")
            messagebox.showerror("Model Error", f"Could not load U-Net.\n\n{str(e)}")
            return
        
        # Check LR
        print(f"Checking LR path: {self.lr_path}")
        if not os.path.exists(self.lr_path):
            messagebox.showerror("Error", f"LR file missing: {self.lr_path}")
            return
        
        try:
            print("Attempting to load Logistic Regression model...")
            self.lr_model = joblib.load(self.lr_path)
            print("SUCCESS: LR loaded.")
        except Exception as e:
            print(f"FAIL: LR crashed. Error: {e}")
            messagebox.showerror("Model Error", f"Could not load LR model.\n\n{str(e)}")
            return
        
        self.status_label.config(text="Models Loaded Successfully.")
        print("--- DEBUG: All Models Loaded ---")
    
    def load_image(self):
        """Load a .npy or .spm image file."""
        file_path = filedialog.askopenfilename(filetypes=[
            ("AFM files", "*.npy *.spm"),
            ("Numpy files", "*.npy"),
            ("SPM files", "*.spm")
        ])
        if file_path:
            self._load_image_from_path(file_path)
    
    def load_random_image(self):
        """Load a random .npy or .spm image from afm_data folder, excluding files with existing corrections."""
        try:
            # Get all AFM files from afm_data folder
            afm_folder = AFM_FOLDER_PATH
            if not os.path.exists(afm_folder):
                messagebox.showerror("Error", f"AFM data folder '{afm_folder}' not found.")
                return
            
            npy_files = glob.glob(os.path.join(afm_folder, "*.npy"))
            spm_files = glob.glob(os.path.join(afm_folder, "*.spm"))
            afm_files = npy_files + spm_files
            
            if not afm_files:
                messagebox.showwarning("Warning", f"No .npy or .spm files found in '{afm_folder}' folder.")
                return
            
            # Filter out files that already have corrections
            available_files = []
            for afm_file in afm_files:
                basename = os.path.splitext(os.path.basename(afm_file))[0]
                corrections_file = os.path.join(CSV_FOLDER_PATH, f"{basename}_corrections.csv")
                if not os.path.exists(corrections_file):
                    available_files.append(afm_file)
            
            if not available_files:
                messagebox.showinfo("Info", "All AFM files in the afm_data folder already have corrections. Please add more data or clear existing corrections.")
                return
            
            # Select random file
            random_file = random.choice(available_files)
            self.status_label.config(text=f"Loading random file: {os.path.basename(random_file)}")
            self._load_image_from_path(random_file)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load random image: {str(e)}")
            self.status_label.config(text="Error loading random image")
    
    def _process_spm_file(self, file_path):
        """Process .spm file using pySPM with the provided correction pipeline."""
        try:
            ScanB = pySPM.Bruker(file_path)
            
            # Get the Height Sensor channel
            data_channel = ScanB.get_channel("Height Sensor")
            
            # Apply the correction pipeline
            topo2 = copy.deepcopy(data_channel)
            topo2.correct_median_diff()
            topo2.correct_median_diff()
            
            topo3 = topo2.filter_scars_removal(.7, inline=False)
            
            # Correct the plane and apply filtering
            topoD = topo3.corr_fit2d(inline=False)
            
            return topoD.pixels
            
        except Exception as e:
            raise Exception(f"Failed to process SPM file: {str(e)}")
    
    def _load_image_from_path(self, file_path):
        """Common image loading logic used by both load_image and load_random_image."""
        try:
            self.current_filename = os.path.basename(file_path).split('.')[0]
            
            # Determine file type and load accordingly
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.spm':
                self.status_label.config(text="Processing SPM file...")
                self.root.update()
                img = self._process_spm_file(file_path)
            elif file_ext == '.npy':
                img = np.load(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            img = np.squeeze(img)
            
            if img.ndim != 2:
                raise ValueError(f"Expected 2D image, got shape {img.shape}")
            
            # Check if image is 512x512 pixels (only supported resolution)
            if img.shape != (512, 512):
                raise ValueError(f"Only 512x512 pixel images are supported. Got {img.shape[0]}x{img.shape[1]} pixels.")
            
            if NORMALISED_MODEL:
                img = (img - img.min()) / (img.max() - img.min())
            
            self.current_image = img
            
            # Clear previous results
            self.results_df = None
            self.binary_mask = None
            self.corrections = {
                'false_positives': [],
                'added_pits': [],
                'modified_pits': []
            }
            self.correction_history = []
            
            # Display preview
            self.plot_preview(img)
            
            # Reset correction buttons to disabled state
            for btn in [self.btn_remove, self.btn_add_small, self.btn_add_large, 
                       self.btn_modify, self.btn_zoom, self.btn_resolve_combined]:
                btn.config(state=tk.DISABLED)
            self.btn_save_corrections.config(state=tk.DISABLED)
            self.btn_undo.config(state=tk.DISABLED)
            self.btn_toggle_labels.config(state=tk.DISABLED)
            
            self.status_label.config(text=f"Loaded: {self.current_filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
            self.status_label.config(text="Error loading image")
    
    def run_analysis(self):
        """Run segmentation and classification analysis."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        if self.unet_model is None or self.lr_model is None:
            messagebox.showerror("Error", "Models are not loaded correctly.")
            return
        
        self.status_label.config(text="Segmenting...")
        self.root.update()
        
        try:
            # Preprocess and segment
            img_tensor = self.current_image.astype(np.float32)[np.newaxis, ..., np.newaxis]
            pred_mask = self.unet_model.predict(img_tensor, verbose=0)
            binary_mask = (pred_mask[0, ..., 0] > 0.5).astype(np.uint8)
            
            # Extract features
            self.status_label.config(text="Extracting Features...")
            features = extract_pit_features_single(self.current_image, binary_mask)
            
            if not features:
                messagebox.showinfo("Result", "No pits detected.")
                self.status_label.config(text="Done. No pits found.")
                return
            
            # Classify
            X_lr = np.array([[f['diameter'], f['depth'], f['volume']] for f in features])
            predictions = self.lr_model.predict(X_lr)
            
            # Create results DataFrame
            df = pd.DataFrame(features)
            df['class_label'] = predictions
            df['class_name'] = df['class_label'].map({1: 'Small', 2: 'Large'})
            
            self.results_df = df.copy()
            self.binary_mask = binary_mask
            
            # Reset corrections
            self.corrections = {
                'false_positives': [],
                'added_pits': [],
                'modified_pits': []
            }
            self.correction_history = []
            
            # Display and save
            self.display_results(df, binary_mask)
            self.enable_correction_mode()
            
            self.status_label.config(text="Analysis Complete. Click correction buttons to refine results.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            print(e)
    
    def display_results(self, df, binary_mask):
        """Display analysis results with visualization."""
        # Clear previous
        for widget in self.canvas_frame.winfo_children(): 
            widget.destroy()
        for widget in self.stats_frame.winfo_children(): 
            widget.destroy()
        
        # Create visualization
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.imshow(self.current_image, cmap=COLORMAP)
        
        if CIRCLE_PITS:
            for _, row in df.iterrows():
                y, x = row['centroid']
                radius = row['diameter'] / 2 + 3
                color = SMALL_PIT_COLOR if row['class_label'] == 1 else LARGE_PIT_COLOR
                circle = Circle((x, y), radius, color=color, fill=False, linewidth=1)
                ax1.add_patch(circle)
        
        ax1.set_title("Result: Blue=Small, Red=Large")
        ax1.axis('off')
        
        canvas1 = FigureCanvasTkAgg(fig1, master=self.canvas_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig_ax = (fig1, ax1)
        self.canvas_widget = canvas1
        fig1.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
        # Stats visualization
        fig2 = plt.figure(figsize=(4, 8))
        
        ax2a = plt.subplot(2, 1, 1)
        counts = df['class_name'].value_counts()
        colors = [SMALL_PIT_COLOR if x == 'Small' else LARGE_PIT_COLOR for x in counts.index]
        counts.plot(kind='bar', color=colors, ax=ax2a)
        ax2a.set_title("Class Distribution")
        ax2a.set_ylabel("Count")
        
        ax2b = plt.subplot(2, 1, 2)
        ax2b.hist(self.current_image.flatten(), bins=50, color='steelblue', edgecolor='black')
        ax2b.set_title("Image Value Histogram")
        ax2b.set_xlabel("Height Sensor (nm)")
        ax2b.set_ylabel("Frequency")
        
        plt.tight_layout()
        
        canvas2 = FigureCanvasTkAgg(fig2, master=self.stats_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_preview(self, img):
        """Plot image preview."""
        for widget in self.canvas_frame.winfo_children(): 
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, cmap=COLORMAP)
        ax.set_title("Input Image")
        ax.axis('off')
        
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def save_data(self, df):
        """Save results to CSV."""
        csv_name = f"{self.current_filename}_results.csv"
        df_save = df.drop(columns=['coords'])
        df_save.to_csv(csv_name, index=False)
        print(f"Saved data to {csv_name}")
    
    def enable_correction_mode(self):
        """Enable correction buttons."""
        self.btn_remove.config(state=tk.NORMAL)
        self.btn_add_small.config(state=tk.NORMAL)
        self.btn_add_large.config(state=tk.NORMAL)
        self.btn_modify.config(state=tk.NORMAL)
        self.btn_zoom.config(state=tk.NORMAL)
        self.btn_resolve_combined.config(state=tk.NORMAL)
        self.btn_save_corrections.config(state=tk.NORMAL)
        self.btn_undo.config(state=tk.NORMAL)
        self.btn_toggle_labels.config(state=tk.NORMAL)
    
    def set_correction_mode(self, mode):
        """Set the active correction mode."""
        if self.correction_mode == mode:
            self.correction_mode = None
            self.status_label.config(text="Correction mode deactivated.")
            self._reset_button_states()
            if self.canvas_widget:
                self.canvas_widget.get_tk_widget().config(cursor="")
        else:
            self.correction_mode = mode
            self._update_button_states(mode)
            
            mode_messages = {
                'remove': "Click on a pit to mark it as false positive.",
                'add_small': "Click on the image to add a small pit.",
                'add_large': "Click on the image to add a large pit.",
                'modify': "Click on a pit to change its class.",
                'zoom': "Click on the image to open a zoomed view.",
                'resolve_combined': "Click on a combined spot to split it into multiple pits."
            }
            self.status_label.config(text=mode_messages.get(mode, ""))
            
            if self.canvas_widget:
                cursor = "crosshair" if mode in ['zoom', 'resolve_combined'] else ""
                self.canvas_widget.get_tk_widget().config(cursor=cursor)
    
    def _reset_button_states(self):
        """Reset all button states."""
        for btn in [self.btn_remove, self.btn_add_small, self.btn_add_large, 
                    self.btn_modify, self.btn_zoom, self.btn_resolve_combined]:
            btn.config(relief=tk.RAISED)
    
    def _update_button_states(self, mode):
        """Update button states to show active mode."""
        self.btn_remove.config(relief=tk.SUNKEN if mode == 'remove' else tk.RAISED)
        self.btn_add_small.config(relief=tk.SUNKEN if mode == 'add_small' else tk.RAISED)
        self.btn_add_large.config(relief=tk.SUNKEN if mode == 'add_large' else tk.RAISED)
        self.btn_modify.config(relief=tk.SUNKEN if mode == 'modify' else tk.RAISED)
        self.btn_zoom.config(relief=tk.SUNKEN if mode == 'zoom' else tk.RAISED)
        self.btn_resolve_combined.config(relief=tk.SUNKEN if mode == 'resolve_combined' else tk.RAISED)
    
    def on_canvas_click(self, event):
        """Handle canvas click events."""
        if event.inaxes is None or self.correction_mode is None:
            return
        
        click_x, click_y = event.xdata, event.ydata
        
        if self.correction_mode == 'remove':
            self.correction_handler.mark_false_positive(click_x, click_y)
        elif self.correction_mode in ['add_small', 'add_large']:
            self.correction_handler.add_pit(click_x, click_y, self.correction_mode)
        elif self.correction_mode == 'modify':
            self.correction_handler.modify_pit_class(click_x, click_y)
        elif self.correction_mode == 'zoom':
            self.zoom_manager.open_zoom_window(click_x, click_y)
        elif self.correction_mode == 'resolve_combined':
            self.zoom_manager.open_resolve_combined_window(click_x, click_y)
    
    def redraw_results(self):
        """Redraw visualization with corrections."""
        if self.fig_ax is None or self.results_df is None:
            return
        
        fig1, ax1 = self.fig_ax
        ax1.clear()
        ax1.imshow(self.current_image, cmap=COLORMAP)
        
        if self.show_labels and CIRCLE_PITS:
            # Draw original pits
            for idx, row in self.results_df.iterrows():
                y, x = row['centroid']
                radius = row['diameter'] / 2 + 3
                cls = row['class_label']
                
                # Check modifications
                modified_class = None
                for mod_idx, new_cls in self.corrections['modified_pits']:
                    if mod_idx == idx:
                        modified_class = new_cls
                        break
                
                if idx in self.corrections['false_positives']:
                    color = FALSE_POSITIVE_COLOR
                    linestyle = '--'
                    linewidth = 1
                elif modified_class is not None:
                    color = SMALL_PIT_COLOR if modified_class == 1 else LARGE_PIT_COLOR
                    linestyle = '-'
                    linewidth = 2
                else:
                    color = SMALL_PIT_COLOR if cls == 1 else LARGE_PIT_COLOR
                    linestyle = '-'
                    linewidth = 1
                
                circle = Circle((x, y), radius, color=color, fill=False,
                              linewidth=linewidth, linestyle=linestyle)
                ax1.add_patch(circle)
            
            # Draw added pits
            for pit in self.corrections['added_pits']:
                y, x = pit['centroid']
                radius = pit['diameter'] / 2 # + 3
                color = SMALL_PIT_COLOR if pit['class_label'] == 1 else LARGE_PIT_COLOR
                circle = Circle((x, y), radius, color=color, fill=False,
                              linewidth=2, linestyle='-')
                ax1.add_patch(circle)
        
        title = "Result: Blue=Small, Red=Large\nGray=FP, Thick=Modified/Added"
        if not self.show_labels:
            title = "Labels Hidden - Raw Image View"
        ax1.set_title(title)
        ax1.axis('off')
        
        self.canvas_widget.draw()
    
    def toggle_labels(self):
        """Toggle label visibility."""
        self.show_labels = not self.show_labels
        self.btn_toggle_labels.config(text="Hide Labels" if self.show_labels else "Show Labels")
        self.status_label.config(text="Labels visible" if self.show_labels else "Labels hidden")
        self.redraw_results()
    
    def save_corrections(self):
        """Save corrections to CSV files."""
        if self.results_df is None:
            messagebox.showwarning("Warning", "No results to save corrections for.")
            return
        
        corrections_data = []
        
        # Add false positives
        for idx in self.corrections['false_positives']:
            row = self.results_df.loc[idx]
            y, x = row['centroid']
            corrections_data.append({
                'filename': self.current_filename,
                'correction_type': 'false_positive',
                'x': x, 'y': y,
                'original_class': row['class_label'],
                'corrected_class': 0,
                'diameter': row['diameter'],
                'depth': row['depth'],
                'volume': row['volume']
            })
        
        # Add new pits
        for pit in self.corrections['added_pits']:
            y, x = pit['centroid']
            corrections_data.append({
                'filename': self.current_filename,
                'correction_type': 'added_pit',
                'x': x, 'y': y,
                'original_class': 0,
                'corrected_class': pit['class_label'],
                'diameter': pit['diameter'],
                'depth': pit['depth'],
                'volume': pit['volume']
            })
        
        # Add modified pits
        for idx, new_class in self.corrections['modified_pits']:
            row = self.results_df.loc[idx]
            y, x = row['centroid']
            corrections_data.append({
                'filename': self.current_filename,
                'correction_type': 'modified_class',
                'x': x, 'y': y,
                'original_class': row['class_label'],
                'corrected_class': new_class,
                'diameter': row['diameter'],
                'depth': row['depth'],
                'volume': row['volume']
            })
        
        if not corrections_data:
            messagebox.showinfo("Info", "No corrections to save.")
            return
        
        # Ensure CSV folder exists
        os.makedirs(CSV_FOLDER_PATH, exist_ok=True)
        
        # Save corrections
        corrections_df = pd.DataFrame(corrections_data)
        corrections_filename = os.path.join(CSV_FOLDER_PATH, f"{self.current_filename}_corrections.csv")
        corrections_df.to_csv(corrections_filename, index=False)
        
        # Save updated results
        updated_results = create_corrected_dataframe(self.results_df, self.corrections)
        updated_filename = os.path.join(CSV_FOLDER_PATH, f"{self.current_filename}_results_corrected.csv")
        updated_results_save = updated_results.drop(columns=['coords'], errors='ignore')
        updated_results_save.to_csv(updated_filename, index=False)
        
        messagebox.showinfo("Success", 
                          f"Saved {len(corrections_data)} corrections to:\n{corrections_filename}\n\n" +
                          f"Updated results saved to:\n{updated_filename}")
        
        self.status_label.config(text=f"Corrections saved! {len(corrections_data)} changes recorded.")


if __name__ == "__main__":
    root = tk.Tk()
    app = PitAnalysisApp(root)
    root.mainloop()