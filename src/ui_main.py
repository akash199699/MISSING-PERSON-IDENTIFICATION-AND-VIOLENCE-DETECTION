import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import torch
from PIL import Image, ImageTk
import numpy as np
import time

# Import functions from the new modules
from missing_person_detection import setup_missing_person_detection, process_video
from violence_detection import load_violence_detection_model, detect_violence_in_video
from report_generation import export_to_pdf, export_violence_report

class MissingPersonDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Missing Person & Violence Detection")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f5f5f5")
        
        # Variables
        self.ref_files = []
        self.video_files = []
        self.detection_threshold = tk.DoubleVar(value=0.65)
        self.frame_interval = tk.IntVar(value=60)
        self.use_gpu = tk.BooleanVar(value=torch.cuda.is_available())
        self.running = False
        self.status_text = tk.StringVar(value="Ready")
        self.start_time = None  # To track the start time of detection
        self.num_detections = 0  # To count the number of detections
        
        # Create UI
        self.create_header()
        self.create_main_frame()
        self.create_status_bar()
        
    def create_header(self):
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            header_frame, 
            text="Missing Person & Violence Detection", 
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#2c3e50",
            pady=10
        )
        title_label.pack()
        
    def create_main_frame(self):
        main_frame = tk.Frame(self.root, bg="#f5f5f5", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (settings)
        settings_frame = tk.LabelFrame(main_frame, text="Settings", bg="#f5f5f5", padx=10, pady=10)
        settings_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel (file selection and preview)
        files_frame = tk.LabelFrame(main_frame, text="Files", bg="#f5f5f5", padx=10, pady=10)
        files_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Settings components
        self.create_settings_components(settings_frame)
        
        # Files components
        self.create_files_components(files_frame)
        
    def create_settings_components(self, parent):
        # Detection type
        detection_frame = tk.Frame(parent, bg="#f5f5f5")
        detection_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(detection_frame, text="Detection Mode:", bg="#f5f5f5").pack(side=tk.LEFT)
        
        self.detection_mode = tk.StringVar(value="Full Pipeline")
        modes = ["Full Pipeline", "Missing Person Only", "Violence Only"]
        
        mode_combo = ttk.Combobox(detection_frame, textvariable=self.detection_mode, values=modes, state="readonly", width=15)
        mode_combo.pack(side=tk.LEFT, padx=5)
        
        # GPU checkbox
        gpu_frame = tk.Frame(parent, bg="#f5f5f5")
        gpu_frame.pack(fill=tk.X, pady=5)
        
        gpu_check = tk.Checkbutton(
            gpu_frame, 
            text=f"Use GPU {'(Available)' if torch.cuda.is_available() else '(Not Available)'}", 
            variable=self.use_gpu,
            bg="#f5f5f5",
            state=tk.NORMAL if torch.cuda.is_available() else tk.DISABLED
        )
        gpu_check.pack(side=tk.LEFT)
         
        # # Add a checkbox for real-time video display
        # self.display_video = tk.BooleanVar(value=True)
        # display_check = tk.Checkbutton(
        #     gpu_frame, 
        #     text="Display Real-Time Video", 
        #     variable=self.display_video,
        #     bg="#f5f5f5"
        # )
        # display_check.pack(side=tk.RIGHT, padx=5)

        # Threshold slider 
        threshold_frame = tk.Frame(parent, bg="#f5f5f5")
        threshold_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(threshold_frame, text="Detection Threshold:", bg="#f5f5f5").pack(anchor=tk.W)
        
        threshold_slider = tk.Scale(
            threshold_frame,
            variable=self.detection_threshold,
            from_=0.5,
            to=0.95,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            bg="#f5f5f5"
        )
        threshold_slider.pack(fill=tk.X)

        # Frame interval slider
        interval_frame = tk.Frame(parent, bg="#f5f5f5")
        interval_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(interval_frame, text="Frame Interval:", bg="#f5f5f5").pack(anchor=tk.W)
        
        interval_slider = tk.Scale(
            interval_frame,
            variable=self.frame_interval,
            from_=1,
            to=300,
            resolution=1,
            orient=tk.HORIZONTAL,
            bg="#f5f5f5"
        )
        interval_slider.pack(fill=tk.X)
        
        # Action buttons
        action_frame = tk.Frame(parent, bg="#f5f5f5", pady=10)
        action_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.run_button = tk.Button(
            action_frame,
            text="Run Detection",
            command=self.run_detection,
            bg="#27ae60",
            fg="white",
            font=("Arial", 12, "bold"),
            pady=10
        )
        self.run_button.pack(fill=tk.X, pady=5)
        
        exit_button = tk.Button(
            action_frame,
            text="Exit",
            command=self.root.destroy,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 12),
            pady=5
        )
        exit_button.pack(fill=tk.X, pady=5)
        
    def create_files_components(self, parent):
        # Reference images 
        ref_frame = tk.Frame(parent, bg="#f5f5f5")
        ref_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(ref_frame, text="Reference Images:", bg="#f5f5f5").pack(anchor=tk.W)
        
        self.ref_label = tk.Label(
            ref_frame,
            text="No images selected",
            bg="white", 
            fg="#555",
            relief=tk.SUNKEN,
            height=2,
            anchor=tk.W,
            padx=5
        )
        self.ref_label.pack(fill=tk.X, pady=2)
        
        ref_button = tk.Button(
            ref_frame,
            text="Select Reference Images",
            command=self.select_reference_images,
            bg="#3498db",
            fg="white"
        )
        ref_button.pack(anchor=tk.W)
        
        # Video files
        video_frame = tk.Frame(parent, bg="#f5f5f5")
        video_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(video_frame, text="Video Files:", bg="#f5f5f5").pack(anchor=tk.W)
        
        self.video_label = tk.Label(
            video_frame,
            text="No videos selected",
            bg="white", 
            fg="#555",
            relief=tk.SUNKEN,
            height=2,
            anchor=tk.W,
            padx=5
        )
        self.video_label.pack(fill=tk.X, pady=2)
        
        video_button = tk.Button(
            video_frame,
            text="Select Video Files",
            command=self.select_video_files,
            bg="#3498db",
            fg="white"
        )
        video_button.pack(anchor=tk.W)
        
        # Preview area
        preview_frame = tk.LabelFrame(parent, text="Preview", bg="#f5f5f5")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.preview_label = tk.Label(
            preview_frame,
            text="Select files to see preview",
            bg="#eee",
            height=10
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
    def create_status_bar(self):
        status_frame = tk.Frame(self.root, bg="#ecf0f1", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.progress_bar = ttk.Progressbar(
            status_frame,
            orient=tk.HORIZONTAL,
            mode='indeterminate'
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)
        
        status_label = tk.Label(
            status_frame,
            textvariable=self.status_text,
            bg="#ecf0f1",
            padx=10
        )
        status_label.pack(side=tk.RIGHT, pady=5)
        
    def select_reference_images(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png")]
        files = filedialog.askopenfilenames(title="Select Reference Images", filetypes=filetypes)
        
        if files:
            self.ref_files = files
            if len(files) == 1:
                self.ref_label.config(text=os.path.basename(files[0]))
            else:
                self.ref_label.config(text=f"{len(files)} images selected")
            
            # Show preview of first image
            try:
                img = Image.open(files[0])
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)
                self.preview_label.config(image=photo, text="")
                self.preview_label.image = photo  # Keep a reference
            except Exception as e:
                print(f"Error loading preview: {e}")
    
    def select_video_files(self):
        filetypes = [("Video files", "*.mp4 *.avi *.mov")]
        files = filedialog.askopenfilenames(title="Select Video Files", filetypes=filetypes)
        
        if files:
            self.video_files = files
            if len(files) == 1:
                self.video_label.config(text=os.path.basename(files[0]))
            else:
                self.video_label.config(text=f"{len(files)} videos selected")
    
    def run_detection(self):
        # Validation
        mode = self.detection_mode.get()
        
        if mode in ["Full Pipeline", "Missing Person Only"] and not self.ref_files:
            messagebox.showerror("Error", "Please select reference images for missing person detection.")
            return
            
        if not self.video_files:
            messagebox.showerror("Error", "Please select video files to analyze.")
            return
        
        if self.running:
            messagebox.showinfo("Info", "Detection is already running.")
            return

        # Save the current settings to persist across mode changes
        self.last_used_ref_files = self.ref_files
        self.last_used_video_files = self.video_files

        # Start detection in a separate thread
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.progress_bar.start()
        self.status_text.set("Processing...")
        
        detection_thread = threading.Thread(target=self.execute_detection)
        detection_thread.daemon = True
        detection_thread.start()
    
    def execute_detection(self):
        try:
            self.start_time = time.time()  # Record the start time
            self.num_detections = 0  # Reset the detection counter

            mode = self.detection_mode.get()
            threshold = self.detection_threshold.get()
            frame_interval = self.frame_interval.get()
            use_gpu = self.use_gpu.get()
        
            # Set device based on GPU selection
            device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
            # Initialize detection models first
            device, mtcnn, resnet = setup_missing_person_detection()
        
            # Update status
            self.root.after(0, lambda: self.status_text.set("Loading models..."))
        
            if mode in ["Full Pipeline", "Missing Person Only"]:
                # Instead of calling load_reference_images(), process the selected files directly
                ref_embeddings = []
            
                # Update status
                self.root.after(0, lambda: self.status_text.set("Processing reference images..."))
            
                # Process reference images (adapted from your load_reference_images function)
                for ref_filename in self.ref_files:
                    ref_img = Image.open(ref_filename).convert("RGB")
                    faces, probs = mtcnn(ref_img, return_prob=True)
                    if faces is None or (hasattr(faces, '__len__') and len(faces) == 0):
                        continue
                    
                    # Use highest probability face if multiple are detected
                    ref_face = faces[int(np.argmax(probs))] if faces.ndim == 4 else faces
                    with torch.no_grad():
                        emb = resnet(
                            ref_face.unsqueeze(0).to(device).half() if device.type=='cuda'
                            else ref_face.unsqueeze(0).to(device)
                        )
                    ref_embeddings.append(emb)
            
                if not ref_embeddings:
                    raise Exception("No valid faces detected in the reference images.")
                
                # Convert reference embeddings to half precision if on GPU
                if device.type == 'cuda':
                    ref_embeddings = [emb.half() for emb in ref_embeddings]
                
                # Process each video
                all_detections = []
                for video_file in self.video_files:
                    self.root.after(0, lambda v=video_file: self.status_text.set(f"Processing video: {os.path.basename(v)}..."))
                
                    # Call your process_video function with our parameters
                    detections = process_video(
                        video_file, 
                        mtcnn, 
                        resnet, 
                        device, 
                        ref_embeddings, 
                        frame_interval=frame_interval,
                        detection_threshold=threshold
                        # display_video=self.display_video.get()  # Pass the flag
                    )
                    all_detections.extend(detections)
                    self.num_detections += len(detections)  # Update the detection count
                
                # Export results    
                if all_detections:
                    self.root.after(0, lambda: self.status_text.set("Generating report..."))
                    all_detections.sort(key=lambda x: x['similarity'], reverse=True)
                    export_to_pdf(all_detections, ref_filenames=self.ref_files)
                
            if mode in ["Full Pipeline", "Violence Only"]:
                # Violence detection part
                self.root.after(0, lambda: self.status_text.set("Detecting violence..."))
            
                # Load violence model
                model = load_violence_detection_model(device)
            
                # Process each video
                for video_file in self.video_files:
                    self.root.after(0, lambda v=video_file: self.status_text.set(f"Checking violence in: {os.path.basename(v)}..."))
                    violence_detections = detect_violence_in_video(
                        video_file, 
                        model, 
                        device, 
                        threshold=threshold
                        # display_video=self.display_video.get()  # Pass the flag
                    )
                    self.num_detections += len(violence_detections)  # Update the detection count
                
                    if violence_detections:
                        export_violence_report(violence_detections, video_file)
                
            self.root.after(0, self.detection_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.detection_error(str(e)))
    
    def detection_complete(self):
        self.progress_bar.stop()
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.status_text.set("Detection completed")

         # Calculate the time taken
        time_taken = time.time() - self.start_time
        time_taken_str = f"{time_taken:.2f} seconds"
    
        # Check if PDF files were generated
        pdf_files = []
        output_dir = "Output"
        for file in os.listdir(output_dir):
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(output_dir, file))
    
        if pdf_files:
            # Create a dialog with buttons to open the PDFs
            result = messagebox.askquestion("Detection Complete", 
                                        f"Detection process completed successfully.\nTime taken: {time_taken_str}\nNumber of detections: {self.num_detections}\nWould you like to view the results?")
            if result == "yes":
                self.show_pdf_viewer(pdf_files)
        else:
            messagebox.showinfo("Complete", f"Detection process completed successfully.\nTime taken: {time_taken_str}\nNumber of detections: {self.num_detections}")

    def show_pdf_viewer(self, pdf_files):
        """Display a window with buttons to open available PDF reports"""
        pdf_window = tk.Toplevel(self.root)
        pdf_window.title("Detection Reports")
        pdf_window.geometry("400x300")
        pdf_window.configure(bg="#f5f5f5")
    
        # Header
        tk.Label(
            pdf_window, 
            text="Available Detection Reports",
            font=("Arial", 14, "bold"),
            bg="#f5f5f5",
            pady=10
        ).pack(fill=tk.X)
    
        # Create a frame for the PDF list
        list_frame = tk.Frame(pdf_window, bg="#f5f5f5", padx=20, pady=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
    
        # Add a button for each PDF file
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
        
            # Determine file type for icon/color
            if "violence" in filename.lower():
                bg_color = "#e74c3c"  # Red for violence
                prefix = "🔍 "
            else:
                bg_color = "#3498db"  # Blue for missing person
                prefix = "👤 "
            
            button_frame = tk.Frame(list_frame, bg="#f5f5f5", pady=5)
            button_frame.pack(fill=tk.X)
        
            # Button to open the PDF
            open_button = tk.Button(
                button_frame,
                text=f"{prefix} Open {filename}",
                command=lambda f=pdf_file: self.open_pdf(f),
                bg=bg_color,
                fg="white",
                font=("Arial", 11),
                pady=8
            )
            open_button.pack(fill=tk.X)
    
        # Close button
        tk.Button(
            pdf_window,
            text="Close",
            command=pdf_window.destroy,
            bg="#7f8c8d",
            fg="white",
            font=("Arial", 11),
            pady=5
        ).pack(fill=tk.X, padx=20, pady=10)

    def open_pdf(self, pdf_path):
        """Open the PDF file with the default viewer"""
        try:
            import platform
            import subprocess
        
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', pdf_path))
            elif platform.system() == 'Windows':  # Windows
                os.startfile(pdf_path)
            else:  # Linux
                subprocess.call(('xdg-open', pdf_path))
        except Exception as e:
            messagebox.showerror("Error", f"Could not open PDF: {e}")
    
    def detection_error(self, error_msg):
        self.progress_bar.stop()
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.status_text.set("Error")
        messagebox.showerror("Error", f"An error occurred during detection:\n{error_msg}")

# Main application
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("Output", exist_ok=True)
    
    # Initialize the app
    root = tk.Tk()
    app = MissingPersonDetectionApp(root)
    
    # Set window icon (if available)
    try:
        root.iconbitmap("icon.ico")  # Replace with your icon path
    except:
        pass
        
    root.mainloop()