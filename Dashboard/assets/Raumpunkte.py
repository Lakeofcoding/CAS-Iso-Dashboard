print("SCRIPT STARTET")
input("Enter drücken zum Weiterlaufen...")



import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

IMG_PATH = "grundriss.png"
CSV_PATH = "Isolationen_Dummy.csv"
OUT_PATH = "room_points.csv"

# 1) Räume aus deinem Datensatz holen
df = pd.read_csv(CSV_PATH, sep=";")
rooms = sorted(df["Raum_ID"].dropna().unique().tolist())

print(f"{len(rooms)} Räume gefunden.")
print("Bedienung: Pro Raum 1x klicken -> Enter im Terminal drücken.")
print("Abbrechen: Fenster schließen oder STRG+C im Terminal.")

# 2) Bild laden + interaktives Fenster
img = Image.open(IMG_PATH)

plt.ion()
fig, ax = plt.subplots(figsize=(14, 7))
ax.imshow(img)
ax.set_axis_off()

points = []

for i, room in enumerate(rooms, start=1):
    ax.set_title(f"[{i}/{len(rooms)}] Klicke Mittelpunkt für: {room}")
    fig.canvas.draw()
    xy = plt.ginput(1, timeout=-1)  # wartet auf 1 Klick
    if not xy:
        print("Kein Klick erkannt -> Abbruch.")
        break
    x, y = xy[0]
    points.append((room, int(x), int(y)))

    # Marker zeichnen, damit du siehst, was schon gemappt ist
    ax.plot([x], [y], marker="x")
    fig.canvas.draw()

# 3) Speichern
out = pd.DataFrame(points, columns=["Raum_ID", "x", "y"])
out.to_csv(OUT_PATH, index=False)
print(f"Gespeichert: {OUT_PATH} ({len(out)} Zeilen)")

plt.ioff()
plt.show()
