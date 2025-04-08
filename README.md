# 🧠 CNN-Based Framework for Missing Person Identification and Violence Detection Using Deep Learning

## 📌 Project Title
CNN-Based Framework for Missing Person Identification and Violence Detection Using Deep Learning

## 🧠 Introduction / Overview
This project aims to leverage Convolutional Neural Networks (CNNs) and computer vision to automate two critical tasks in surveillance systems:
- **Missing person identification** from video feeds or static frames.
- **Violence detection** to alert authorities in real-time for swift action.

The system is designed to enhance public safety through AI-powered automation, especially in crowded or high-risk environments.

## ⚙️ Working / Functionality
- **Missing Person Identification**:
  - Detects and identifies faces using CNN-based classifiers.
  - Matches detected faces against a known database of missing persons.
- **Violence Detection**:
  - Analyzes frames from video streams using motion & object features.
  - Flags frames showing physical aggression using trained models.
- **UI Module**:
  - User-friendly interface to upload input videos/images and view results.
- **Report Generation**:
  - Auto-generates summary reports with findings.

## 🏗️ Architecture / Flow
1. Load the video stream or image.
2. Perform frame-wise analysis:
   - Face detection → Match with missing person DB.
   - Movement/gesture analysis → Detect violence.
3. Generate alert or log results.
4. Visualize outcomes via the UI.
5. Optional: Export findings to a structured report.

```
[Input Video] → [Preprocessing] → [CNN Inference] → [Classification] → [Alert/Report]
```

## 🔧 Technologies Used
- Python
- OpenCV
- TensorFlow / Keras
- CNN Architectures
- PyQt5 (for UI)
- Pandas & NumPy

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.7 or higher

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the UI
```bash
python src/ui_main.py
```

### Run core detection script (headless mode)
```bash
python src/main.py
```

## 💻 Usage

| Script                     | Function                            |
|---------------------------|-------------------------------------|
| `ui_main.py`              | Launch UI                           |
| `main.py`                 | Execute detection pipelines         |
| `violence_detection.py`   | Violence-specific model             |
| `missing_person_detection.py` | Face detection & matching    |
| `report_generation.py`    | Create PDF reports                  |
| `config.py`               | Set paths, model params             |

- **Input**: Videos or Images  
- **Output**: UI-based alerts, logs, screenshots, reports

## 📁 Project Structure

```
Project/
├── src/
│   ├── main.py
│   ├── ui_main.py
│   ├── violence_detection.py
│   ├── missing_person_detection.py
│   ├── report_generation.py
│   ├── utils.py
│   └── config.py
├── requirements.txt
└── README.md
```

## 🙌 Credits

**Project Developer**  
Akash Krishna – B.Tech AI & ML, KTU – 6th Semester  
📧 Email: akash199699@gmail.com  
🔗 GitHub: [@akash199699](https://github.com/akash199699)

This project was developed as part of the mini project under the university curriculum.  
**Special thanks** to our mentors for their guidance, and teammates **Anandhu S Kumar** and **Jewel Saji** for their collaboration.