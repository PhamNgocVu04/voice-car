import os, tarfile, urllib.request
URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
TAR = "speech_commands_v0.02.tar.gz"
WANT = {"up", "down", "left", "right"}

if not os.path.exists(TAR):
    print("Dang tai ~2.3GB, cho vai phut...")
    urllib.request.urlretrieve(URL, TAR)
    print("Tai xong.")

print("Dang giai nen, cho 2-5 phut...")
os.makedirs("data", exist_ok=True)
n = 0
with tarfile.open(TAR) as t:
    for m in t:
        parts = [p for p in m.name.split("/") if p not in ("", ".")]
        if m.isfile() and m.name.endswith(".wav") and len(parts) == 2 and parts[0] in WANT:
            m.name = parts[0] + "/" + parts[1]
            try:
                t.extract(m, "data", filter="data")
            except TypeError:
                t.extract(m, "data")
            n += 1
            if n % 2000 == 0:
                print(f"  da giai nen {n} file")
print(f"Xong {n} file trong thu muc data. Tiep theo: py -3.14 train.py")