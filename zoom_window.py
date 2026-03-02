# Zoom window functionality for AFM pit analysis GUI

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
from config import (
    ZOOM_WINDOW_SIZE, ZOOM_WINDOW_HALF,
    COLORMAP, SMALL_PIT_COLOR, LARGE_PIT_COLOR,
    CIRCLE_PITS
)


class ZoomWindowManager:
    """Manages zoom window and resolve combined spots functionality."""
    
    def __init__(self, app):
        """
        Initialize zoom window manager.
        
        Args:
            app: Reference to main PitAnalysisApp instance
        """
        self.app = app
        self.zoom_window = None
        self.zoom_click_pos = None
        self.zoom_canvas_data = None
        self.zoom_mode = None
        self.resolve_combined_mode = False
    
    def open_zoom_window(self, x, y):
        """Open a popup window with a zoomed view around the clicked point."""
        if self.app.current_image is None:
            return
        
        self._close_existing_window()
        self.zoom_click_pos = (int(round(x)), int(round(y)))
        
        # Create popup window
        self.zoom_window = tk.Toplevel(self.app.root)
        self.zoom_window.title("Zoomed View")
        self.zoom_window.geometry("600x700")
        
        # Setup zoom view
        self._setup_zoom_view(self.zoom_window, x, y, "Zoomed View")
        
        # Add control buttons
        self._add_zoom_buttons(self.zoom_window, resolve_mode=False)
        
        # Connect click event
        self.zoom_canvas_data['fig'].canvas.mpl_connect('button_press_event', self.on_zoom_canvas_click)
    
    def open_resolve_combined_window(self, x, y):
        """Open a zoom window for resolving combined spots."""
        if self.app.current_image is None:
            return
        
        # First, mark the clicked spot as false positive
        self.app.correction_handler.mark_false_positive(x, y)
        self.resolve_combined_mode = True
        
        self._close_existing_window()
        self.zoom_click_pos = (int(round(x)), int(round(y)))
        
        # Create popup window
        self.zoom_window = tk.Toplevel(self.app.root)
        self.zoom_window.title("Resolve Combined Spots - Add Multiple Pits")
        self.zoom_window.geometry("600x700")
        
        # Setup zoom view
        self._setup_zoom_view(self.zoom_window, x, y, "Resolve Combined Spots")
        
        # Add control buttons for resolve mode
        self._add_zoom_buttons(self.zoom_window, resolve_mode=True)
        
        # Connect click event
        self.zoom_canvas_data['fig'].canvas.mpl_connect('button_press_event', self.on_resolve_combined_canvas_click)
    
    def _close_existing_window(self):
        """Close any existing zoom window."""
        if self.zoom_window is not None:
            try:
                self.zoom_window.destroy()
            except:
                pass
            self.zoom_window = None
            self.zoom_canvas_data = None
    
    def _setup_zoom_view(self, window, x, y, title):
        """Setup the zoom view visualization."""
        h, w = self.app.current_image.shape
        cx, cy = self.zoom_click_pos
        
        # Calculate bounds
        x1 = max(0, cx - ZOOM_WINDOW_HALF)
        x2 = min(w, cx + ZOOM_WINDOW_HALF)
        y1 = max(0, cy - ZOOM_WINDOW_HALF)
        y2 = min(h, cy + ZOOM_WINDOW_HALF)
        
        # Extract region
        zoom_region = self.app.current_image[y1:y2, x1:x2].copy()
        
        # Create figure
        zoom_fig = plt.figure(figsize=(6, 6))
        zoom_ax = zoom_fig.add_subplot(111)
        zoom_ax.imshow(zoom_region, cmap=COLORMAP, interpolation='nearest')
        
        # Draw pit markers
        self._draw_zoom_markers(zoom_ax, x1, y1, x2, y2)
        
        zoom_ax.axis('off')
        
        # Create canvas
        zoom_canvas_frame = tk.Frame(window)
        zoom_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        zoom_canvas = FigureCanvasTkAgg(zoom_fig, master=zoom_canvas_frame)
        zoom_canvas.draw()
        zoom_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store data
        self.zoom_canvas_data = {
            'canvas': zoom_canvas,
            'fig': zoom_fig,
            'ax': zoom_ax,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
    
    def _draw_zoom_markers(self, ax, x1, y1, x2, y2):
        """Draw pit markers in the zoom view."""
        if not self.app.show_labels or self.app.results_df is None:
            return
        
        if not CIRCLE_PITS:
            return
        
        # Draw existing pits
        for idx, row in self.app.results_df.iterrows():
            pit_y, pit_x = row['centroid']
            
            if x1 <= pit_x < x2 and y1 <= pit_y < y2:
                zoom_x = pit_x - x1
                zoom_y = pit_y - y1
                
                radius = row['diameter'] / 2 # + 3
                cls = row['class_label']
                
                # Check if modified
                modified_class = None
                for mod_idx, new_cls in self.app.corrections['modified_pits']:
                    if mod_idx == idx:
                        modified_class = new_cls
                        break
                
                # Skip if marked as false positive
                if idx in self.app.corrections['false_positives']:
                    if not self.resolve_combined_mode:  # Show in regular zoom
                        color = 'gray'
                        linestyle = '--'
                        linewidth = 1
                    else:
                        continue  # Don't show in resolve mode
                elif modified_class is not None:
                    color = SMALL_PIT_COLOR if modified_class == 1 else LARGE_PIT_COLOR
                    linestyle = '-'
                    linewidth = 2
                else:
                    color = SMALL_PIT_COLOR if cls == 1 else LARGE_PIT_COLOR
                    linestyle = '-'
                    linewidth = 1
                
                circle = Circle((zoom_x, zoom_y), radius, color=color, fill=False,
                              linewidth=linewidth, linestyle=linestyle)
                ax.add_patch(circle)
        
        # Draw added pits
        for pit in self.app.corrections['added_pits']:
            pit_y, pit_x = pit['centroid']
            
            if x1 <= pit_x < x2 and y1 <= pit_y < y2:
                zoom_x = pit_x - x1
                zoom_y = pit_y - y1
                
                radius = pit['diameter'] / 2 # + 3
                color = SMALL_PIT_COLOR if pit['class_label'] == 1 else LARGE_PIT_COLOR
                circle = Circle((zoom_x, zoom_y), radius, color=color, fill=False,
                              linewidth=2, linestyle='-')
                ax.add_patch(circle)
    
    def _add_zoom_buttons(self, window, resolve_mode=False):
        """Add control buttons to zoom window."""
        button_frame = tk.Frame(window, bg="#e1e1e1", pady=10)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        if resolve_mode:
            # Resolve combined mode buttons
            btn_remove = tk.Button(button_frame, text="Remove Another Spot",
                                   command=lambda: self.set_zoom_correction_mode('remove'),
                                   bg="#ffcccc", height=2)
            btn_remove.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            
            btn_add_small = tk.Button(button_frame, text="Add Small Pit",
                                      command=lambda: self.set_zoom_correction_mode('add_small'),
                                      bg="#ccccff", height=2)
            btn_add_small.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            
            btn_add_large = tk.Button(button_frame, text="Add Large Pit",
                                      command=lambda: self.set_zoom_correction_mode('add_large'),
                                      bg="#ffcccc", height=2)
            btn_add_large.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            
            btn_close = tk.Button(button_frame, text="Done - Close",
                                  command=self.close_zoom_window,
                                  bg="#bdffd1", height=2, width=15)
            btn_close.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        else:
            # Regular zoom mode buttons
            btn_remove = tk.Button(button_frame, text="Mark False Positive",
                                   command=lambda: self.set_zoom_correction_mode('remove'),
                                   bg="#ffcccc", height=2)
            btn_remove.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            
            btn_add_small = tk.Button(button_frame, text="Add Small Pit",
                                      command=lambda: self.set_zoom_correction_mode('add_small'),
                                      bg="#ccccff", height=2)
            btn_add_small.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            
            btn_add_large = tk.Button(button_frame, text="Add Large Pit",
                                      command=lambda: self.set_zoom_correction_mode('add_large'),
                                      bg="#ffcccc", height=2)
            btn_add_large.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            
            btn_modify = tk.Button(button_frame, text="Change Class",
                                   command=lambda: self.set_zoom_correction_mode('modify'),
                                   bg="#ffffcc", height=2)
            btn_modify.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            
            btn_close = tk.Button(button_frame, text="Close (No Changes)",
                                  command=self.close_zoom_window,
                                  bg="#d1d1d1", height=2)
            btn_close.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
    
    def set_zoom_correction_mode(self, mode):
        """Set the correction mode for zoom window."""
        self.zoom_mode = mode
        
        if self.zoom_window:
            title = f"{'Resolve Combined - ' if self.resolve_combined_mode else 'Zoomed View - '}{mode.replace('_', ' ').title()} Mode"
            self.zoom_window.title(title)
    
    def on_zoom_canvas_click(self, event):
        """Handle click events on the regular zoom canvas."""
        if event.inaxes is None or self.zoom_mode is None or self.zoom_canvas_data is None:
            return
        
        orig_x, orig_y = self._transform_zoom_coords(event.xdata, event.ydata)
        
        # Apply correction
        if self.zoom_mode == 'remove':
            self.app.correction_handler.mark_false_positive(orig_x, orig_y)
        elif self.zoom_mode in ['add_small', 'add_large']:
            self.app.correction_handler.add_pit(orig_x, orig_y, self.zoom_mode)
        elif self.zoom_mode == 'modify':
            self.app.correction_handler.modify_pit_class(orig_x, orig_y)
        
        # Close window after action
        self.close_zoom_window()
    
    def on_resolve_combined_canvas_click(self, event):
        """Handle click events on the resolve combined canvas - keeps window open."""
        if event.inaxes is None or self.zoom_mode is None or self.zoom_canvas_data is None:
            return
        
        orig_x, orig_y = self._transform_zoom_coords(event.xdata, event.ydata)
        
        # Apply correction
        if self.zoom_mode == 'remove':
            self.app.correction_handler.mark_false_positive(orig_x, orig_y)
            self.refresh_zoom_view()
        elif self.zoom_mode in ['add_small', 'add_large']:
            self.app.correction_handler.add_pit(orig_x, orig_y, self.zoom_mode)
            self.refresh_zoom_view()
        
        # Don't close window - allow multiple additions
    
    def _transform_zoom_coords(self, zoom_x, zoom_y):
        """Transform zoom coordinates back to original image coordinates."""
        x1 = self.zoom_canvas_data['x1']
        y1 = self.zoom_canvas_data['y1']
        return zoom_x + x1, zoom_y + y1
    
    def refresh_zoom_view(self):
        """Refresh the zoom window to show latest corrections."""
        if self.zoom_canvas_data is None:
            return
        
        zoom_ax = self.zoom_canvas_data['ax']
        zoom_canvas = self.zoom_canvas_data['canvas']
        x1, y1 = self.zoom_canvas_data['x1'], self.zoom_canvas_data['y1']
        x2, y2 = self.zoom_canvas_data['x2'], self.zoom_canvas_data['y2']
        
        # Clear and redraw
        zoom_ax.clear()
        zoom_region = self.app.current_image[y1:y2, x1:x2].copy()
        zoom_ax.imshow(zoom_region, cmap=COLORMAP, interpolation='nearest')
        self._draw_zoom_markers(zoom_ax, x1, y1, x2, y2)
        zoom_ax.axis('off')
        zoom_canvas.draw()
    
    def close_zoom_window(self):
        """Close the zoom window."""
        self._close_existing_window()
        self.zoom_mode = None
        self.resolve_combined_mode = False