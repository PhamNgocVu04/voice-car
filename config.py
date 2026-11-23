SAMPLE_RATE = 16000
DURATION = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)

LABELS = ["up", "down", "left", "right"]
LABEL_SHOW = {"up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT"}

DATA_DIR = "data"
MODEL_DIR = "models"

N_MFCC = 13
N_FFT = 400
HOP = 160
CONF_THRESHOLD = 0.80
MAX_PER_CLASS = 1500