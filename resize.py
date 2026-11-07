import cv2
import glob
import os

# Folder sumber
input_folder = "images"

# Folder hasil resize
output_folder = "images_resized"

# Ukuran yang akan disamakan
WIDTH = 1600
HEIGHT = 900

# Buat folder output kalau belum ada
os.makedirs(output_folder, exist_ok=True)

# Ambil semua JPG/JPEG
images = glob.glob(os.path.join(input_folder, "*"))

images = [
    f for f in images
    if f.lower().endswith((".jpg", ".jpeg"))
]

images = sorted(set(images))

print("Jumlah foto:", len(images))
print()

for fname in images:

    img = cv2.imread(fname)

    if img is None:
        print("GAGAL:", fname)
        continue

    # Resize menjadi 1600 x 900
    resized = cv2.resize(
        img,
        (WIDTH, HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # Nama file tetap sama
    filename = os.path.basename(fname)

    output_path = os.path.join(
        output_folder,
        filename
    )

    cv2.imwrite(output_path, resized)

    print(
        f"OK: {filename} "
        f"-> {WIDTH} x {HEIGHT}"
    )

print()
print("Selesai!")
print("Hasil ada di folder:", output_folder)