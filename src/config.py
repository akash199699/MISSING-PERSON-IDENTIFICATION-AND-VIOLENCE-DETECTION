import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_GPUS = torch.cuda.device_count()
    FP16 = True if DEVICE == "cuda" else False
    
    # Violence Detection
    VIOLENCE_MODEL = "i3d"  # "slowfast", "r3d", or "i3d"
    VIOLENCE_THRESH = 0.68
    CLIP_LENGTH = 32
    CLIP_STRIDE = 16
    
    # Missing Person
    FACE_THRESH = 0.72
    FRAME_INTERVAL = 15
    BATCH_SIZE = 16
    
    # Spark
    SPARK_CONF = {
        "app": "DetectionSystem",
        "executor.memory": "8g",
        "driver.memory": "4g",
        "executor.cores": "2"
    }
    
    # Paths
    MODEL_CACHE = os.getenv("MODEL_CACHE", "./models")
    OUTPUT_DIR = "./Output"
    
    def __post_init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.MODEL_CACHE, exist_ok=True)
    
config = Config()