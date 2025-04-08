import os, time, tempfile
from PIL import Image, ImageDraw
from fpdf import FPDF
import platform
import subprocess
import matplotlib.pyplot as plt
from config import config

# Missing Person Detection PDF Report
def export_to_pdf(detections, pdf_filename="Output/detections.pdf", ref_filenames=None):
    """
    Export detection detections to a PDF report with improved formatting.
    Each detection includes the video filename, detection time, similarity score,
    and an image preview with bounding box and dominant color.
    """
    # Create PDF with appropriate settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Set up metadata
    pdf.set_title("Missing Person Detection Report")
    pdf.set_author("Missing Person Detection System")

    # Add cover page
    pdf.add_page()
    pdf.set_font("Arial", "B", size=27)
    pdf.cell(0, 80, "Missing Person Detection", 0, 1, 'C')
    pdf.cell(0, 20, "Report", 0, 1, 'C')

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, f"Total Detections: {len(detections)}", 0, 1, 'C')

    # Add timestamp
    pdf.set_font("Arial", "I", size=10)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Generated: {current_time}", 0, 1, 'C')

    # Add reference images section
    if ref_filenames:
        pdf.add_page()
        pdf.set_font("Arial", "B", size=16)
        pdf.cell(0, 10, "Reference Images Used", 0, 1, 'C')

        # Add description
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, "The following reference images were used to identify the missing person:", 0, 1, 'L')

        # Calculate layout for reference images
        images_per_row = 2
        margin = 10
        image_width = (pdf.w - 2 * margin) / images_per_row - 10
        image_height = image_width * 0.75  # Aspect ratio

        # Add each reference image
        x_pos = margin
        y_pos = pdf.get_y() + 5

        for i, ref_filename in enumerate(ref_filenames):
            # Position images in a grid
            if i > 0 and i % images_per_row == 0:
                x_pos = margin
                y_pos += image_height + 20

            # Add the image
            pdf.image(ref_filename, x=x_pos, y=y_pos, w=image_width, h=image_height)

            # Add filename caption
            pdf.set_xy(x_pos, y_pos + image_height + 2)
            pdf.set_font("Arial", "I", size=8)
            display_name = os.path.basename(ref_filename)
            if len(display_name) > 25:
                display_name = display_name[:22] + "..."
            pdf.cell(image_width, 10, display_name, 0, 1, 'C')

            x_pos += image_width + 10

    # Add detection summary table
    if detections:
        pdf.add_page()
        pdf.set_font("Arial", "B", size=16)
        pdf.cell(0, 10, "Detections Summary", 0, 1, 'C')

        # Create header row
        pdf.set_font("Arial", "B", size=10)
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(70, 8, "Video", 1, 0, 'C', True)
        pdf.cell(30, 8, "Time (s)", 1, 0, 'C', True)
        pdf.cell(30, 8, "Similarity", 1, 0, 'C', True)
        pdf.cell(60, 8, "Dominant Color", 1, 1, 'C', True)

        # Fill data rows
        pdf.set_font("Arial", size=9)
        for det in detections:
            # Truncate filename if too long
            filename = det['video_filename']
            if len(filename) > 30:
                filename = filename[:27] + "..."

            pdf.cell(70, 8, filename, 1, 0, 'L')
            pdf.cell(30, 8, f"{det['time']:.2f}", 1, 0, 'C')

            # Color the similarity cell based on confidence level
            similarity = det['similarity']
            if similarity > 0.8:
                pdf.set_fill_color(150, 255, 150)  # Green for high confidence
            elif similarity > 0.7:
                pdf.set_fill_color(255, 255, 150)  # Yellow for medium confidence
            else:
                pdf.set_fill_color(255, 200, 200)  # Light red for lower confidence

            pdf.cell(30, 8, f"{similarity:.2f}", 1, 0, 'C', True)

            # Reset fill color
            pdf.set_fill_color(255, 255, 255)

            # Get color values
            r, g, b = det['dominant_color']

            # Add color box and text
            pdf.cell(40, 8, f"RGB: {r},{g},{b}", 1, 0, 'L')
            pdf.set_fill_color(r, g, b)
            pdf.cell(20, 8, "", 1, 1, 'C', True)

            # Reset fill color
            pdf.set_fill_color(255, 255, 255)

    # # Create timeline of detections if we have time data
    # t=1
    # if any('time' in r for r in detections):
    #     times = [r.get('time', 0) for r in detections]
    #     similarities = [r.get('similarity', 0) for r in detections]
            
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(times, similarities, 'b-', linewidth=2)
    #     plt.axhline(y=config.FACE_THRESH, color='black', linestyle='--', alpha=0.7)
    #     plt.xlabel("Time (seconds)")
    #     plt.ylabel("Similarity Score")
    #     plt.title("Missing Person Detection Timeline")
    #     plt.grid(True, alpha=0.3)
            
    #     # Save timeline plot
    #     timeline_path = os.path.join(config.OUTPUT_DIR, f"missing_timeline_{t}.png")
    #     plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    #     plt.close()
            
    #     # Add to PDF
    #     pdf.image(timeline_path, w=190)
    #     pdf.ln(5)
        
    #     t+=1

    #     # Clean up
    #     if os.path.exists(timeline_path):
    #         os.remove(timeline_path)

    # Add detailed detections
    images_per_page = 4  # Reduced from 6 to allow more space
    current_image = 0

    for idx, det in enumerate(detections):
        # Start a new page for first image or when page is full
        if current_image == 0:
            pdf.add_page()
            pdf.set_font("Arial", "B", size=14)
            pdf.cell(0, 10, f"Detection Details", 0, 1, 'C')
            current_image = 0

        # Calculate position with more spacing
        row = current_image // 2
        col = current_image % 2

        # Base positions with more space between items
        x_start = 10 + col * 95
        y_start = 30 + row * 120

        # Create a box around the entire detection
        pdf.set_draw_color(100, 100, 100)
        pdf.rect(x_start, y_start, 90, 110)

        # Process and add the image
        img = Image.fromarray(det['frame_img'])
        draw = ImageDraw.Draw(img)
        draw.rectangle(det['box'], outline="red", width=3)
        img_resized = img.resize((400, 300))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_filename = tmp.name
            img_resized.save(tmp_filename)

        # Add the image with proper positioning
        pdf.image(tmp_filename, x_start + 5, y_start + 5, w=80, h=60)
        os.remove(tmp_filename)

        # Add detection information below the image
        y_text = y_start + 70
        pdf.set_xy(x_start + 5, y_text)

        # Add video filename and timestamp with proper styling
        pdf.set_font("Arial", "B", size=10)
        pdf.cell(30, 5, "Video:", 0, 0, 'L')
        pdf.set_font("Arial", size=10)
        pdf.cell(50, 5, f"{det['video_filename']}", 0, 2, 'R')

        y_text = y_start + 70 + 10
        pdf.set_xy(x_start + 5, y_text)

        pdf.set_font("Arial", "B", size=9)
        pdf.cell(20, 10, "Timestamp:", 0, 0, 'L')
        pdf.set_font("Arial", size=9)
        pdf.cell(30, 10, f"{det['time']:.2f}s", 0, 0, 'L')

        pdf.set_font("Arial", "B", size=9)
        pdf.cell(20, 10, "Similarity:", 0, 0, 'L')
        pdf.set_font("Arial", size=9)
        pdf.cell(10, 10, f"{det['similarity']:.2f}", 0, 1, 'R')

        y_text = y_start + 70 + 20
        pdf.set_xy(x_start + 5, y_text)

        # Add dominant color information
        pdf.set_font("Arial", "B", size=9)
        pdf.cell(50, 10, "Clothing Color:", 0, 0, 'L')

        # Add color swatch
        r, g, b = det['dominant_color']
        pdf.set_fill_color(r, g, b)
        pdf.rect(x_start + 35, y_text + 2.5 , 50, 5, style='F')

        # Increment counter for positioning
        current_image += 1

        # Reset after filling a page
        if current_image >= images_per_page:
            current_image = 0

    # Add information footer
    pdf.set_y(-25)
    pdf.set_font("Arial", "I", size=8)
    pdf.cell(0, 10, "Missing Person Detection System - Confidential Report", 0, 0, 'C')
    pdf.cell(0, 10, f"Page {pdf.page_no()}", 0, 0, 'R')

    # Save the PDF
    pdf.output(pdf_filename)
    print(f"PDF saved as {pdf_filename}")
    # Open the PDF with the default PDF viewer
    try:
        import platform
        import subprocess
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', pdf_filename))
        elif platform.system() == 'Windows':    # Windows
            os.startfile(pdf_filename)
        else:                                   # Linux
            subprocess.call(('xdg-open', pdf_filename))
    except:
        print("Could not open PDF automatically. Please open it manually.")

# Violence Detection PDF Report  
def export_violence_report(detections, video_filename, pdf_filename="Output/violence_detections.pdf"):
    """Create a PDF report for violence detections with improved formatting"""
    if not detections:
        print(f"No violence detected in {video_filename}")
        return

    # Create PDF with larger margins to avoid overlap
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add title
    pdf.set_font("Arial", "B", size=16)
    pdf.cell(0, 10, "Violence Detection Report", 0, 1, 'C')
    pdf.ln(5)

    # Add video information
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 8, f"Video File:", 0, 1)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"{os.path.basename(video_filename)}", 0, 1)

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 8, f"Total Detections:", 0, 1)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"{len(detections)}", 0, 1)

    pdf.ln(10)

    # Add each detection with proper spacing
    for idx, det in enumerate(detections):
        # Create a new page for each detection except the first one
        if idx > 0:
            pdf.add_page()

        # Detection header with background color
        pdf.set_fill_color(220, 220, 220)  # Light gray background
        pdf.set_font("Arial", "B", size=14)
        pdf.cell(0, 10, f"Detection #{idx+1}", 1, 1, 'L', fill=True)
        pdf.ln(5)

        # Save thumbnail to temp file
        thumbnail = Image.fromarray(det['thumbnail'])
        thumbnail_resized = thumbnail.resize((320, 240))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_filename = tmp.name
            thumbnail_resized.save(tmp_filename)

        # Add thumbnail to PDF - centered
        image_width = 120
        margin_left = (210 - image_width) / 2  # A4 width is 210mm
        pdf.image(tmp_filename, x=margin_left, y=pdf.get_y(), w=image_width)
        os.remove(tmp_filename)

        # Move cursor below the image
        pdf.ln(120)  # Adjust based on your image height

        # Add detection details in a table-like format
        pdf.set_font("Arial", "B", size=11)
        pdf.cell(40, 8, "Time:", 1, 0)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, f"{det['time']:.2f} seconds", 1, 1)

        pdf.set_font("Arial", "B", size=11)
        pdf.cell(40, 8, "Probability:", 1, 0)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, f"{det['probability']:.4f}", 1, 1)

        pdf.set_font("Arial", "B", size=11)
        pdf.cell(40, 8, "Frame Index:", 1, 0)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, f"{det['frame_idx']}", 1, 1)

        # Add severity indicator based on probability
        pdf.ln(10)
        pdf.set_font("Arial", "B", size=12)
        pdf.cell(0, 8, "Severity Level:", 0, 1)

        # Determine severity level based on probability
        if det['probability'] > 0.9:
            severity = "HIGH"
            r, g, b = 255, 0, 0  # Red
        elif det['probability'] > 0.8:
            severity = "MEDIUM"
            r, g, b = 255, 165, 0  # Orange
        else:
            severity = "LOW"
            r, g, b = 255, 255, 0  # Yellow

        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(0 if severity == "LOW" else 255)
        pdf.cell(60, 10, f" {severity} ", 1, 1, 'C', fill=True)
        pdf.set_text_color(0)  # Reset text color to black

    # Add footer with timestamp
    pdf.set_y(-30)
    pdf.set_font("Arial", "I", size=8)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Report generated on {current_time}", 0, 0, 'C')

    pdf.output(pdf_filename)
    print(f"Violence detection report saved as {pdf_filename}")
    
    # Open the PDF with the default PDF viewer
    try:
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', pdf_filename))
        elif platform.system() == 'Windows':    # Windows
            os.startfile(pdf_filename)
        else:                                   # Linux
            subprocess.call(('xdg-open', pdf_filename))
    except:
        print("Could not open PDF automatically. Please open it manually.")
