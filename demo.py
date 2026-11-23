import os, math, numpy as np, pygame
import tensorflow as tf
import sounddevice as sd
from config import SAMPLE_RATE, N_SAMPLES, LABELS, LABEL_SHOW, MODEL_DIR, CONF_THRESHOLD
from features import fix_length, mfcc_sequence

model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm.keras"))
nz = np.load(os.path.join(MODEL_DIR, "norm.npz"))
MEAN, STD, T = nz["mean"], nz["std"], int(nz["T"])

def predict():
    a = sd.rec(N_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    seq = mfcc_sequence(fix_length(a.flatten()))
    X = np.zeros((1, T, seq.shape[1]), np.float32)
    n = min(seq.shape[0], T)
    X[0, :n] = seq[:n]
    X = (X - MEAN) / STD
    p = model.predict(X, verbose=0)[0]
    i = int(p.argmax())
    return LABELS[i], float(p[i])

pygame.init()
W, H = 800, 600
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Voice Car - bam SPACE de noi")
font = pygame.font.SysFont("arial", 32)
small = pygame.font.SysFont("arial", 22)
clock = pygame.time.Clock()
x, y, angle = W / 2, H / 2, 90.0
status, conf = "San sang", 0.0
trail = [(int(x), int(y))]

def move(cmd):
    global x, y, angle
    step = 40
    if cmd == "up":
        x += step * math.cos(math.radians(angle)); y -= step * math.sin(math.radians(angle))
    elif cmd == "down":
        x -= step * math.cos(math.radians(angle)); y += step * math.sin(math.radians(angle))
    elif cmd == "left":
        angle = (angle + 90) % 360
    elif cmd == "right":
        angle = (angle - 90) % 360
    x = max(30, min(W - 30, x)); y = max(30, min(H - 30, y))
    trail.append((int(x), int(y)))

def draw():
    screen.fill((25, 25, 35))
    for gx in range(0, W, 40): pygame.draw.line(screen, (40, 40, 55), (gx, 0), (gx, H))
    for gy in range(0, H, 40): pygame.draw.line(screen, (40, 40, 55), (0, gy), (W, gy))
    if len(trail) > 1: pygame.draw.lines(screen, (90, 90, 140), False, trail, 3)
    car = pygame.Surface((60, 36), pygame.SRCALPHA); car.fill((80, 200, 120))
    pygame.draw.polygon(car, (255, 255, 255), [(60, 18), (44, 6), (44, 30)])
    rot = pygame.transform.rotate(car, angle)
    screen.blit(rot, rot.get_rect(center=(int(x), int(y))))
    screen.blit(font.render(f"{status}  ({conf*100:.0f}%)", True, (255, 220, 0)), (20, 20))
    screen.blit(small.render("SPACE: noi lenh | Mui ten: du phong | ESC: thoat", True, (160, 160, 160)), (20, H - 35))
    pygame.display.flip()

KEYS = {pygame.K_UP: "up", pygame.K_DOWN: "down", pygame.K_LEFT: "left", pygame.K_RIGHT: "right"}
running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            elif e.key == pygame.K_SPACE:
                status, conf = "Dang nghe...", 0.0
                draw()
                cmd, conf = predict()
                if conf >= CONF_THRESHOLD:
                    status = LABEL_SHOW[cmd]; move(cmd)
                else:
                    status = "Khong chac: " + LABEL_SHOW[cmd]
            elif e.key in KEYS:
                status, conf = "Ban phim: " + LABEL_SHOW[KEYS[e.key]], 1.0
                move(KEYS[e.key])
    draw()
    clock.tick(30)
pygame.quit()