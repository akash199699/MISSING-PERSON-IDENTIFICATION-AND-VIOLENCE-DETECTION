import sys
import tkinter as tk
from tkinter import filedialog

def select_files(title, filetypes):
    """Open file dialog to select files"""
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    return files

def load_video_files():
    """Load video files to search for the missing person"""
    print("Select video files to analyze:")
    video_files = select_files("Select Video Files", 
                             [("Video files", "*.mp4 *.avi *.mov")])
    
    if not video_files:
        sys.exit("No video files selected.")
    return video_files