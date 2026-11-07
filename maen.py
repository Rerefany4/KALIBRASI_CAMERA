import cv2
import numpy as np
import glob
import os


# ============================================================
# 1. PENGATURAN
# ============================================================

# Jumlah inner corners pada chessboard
CHECKERBOARD = (6, 9)

# Ukuran satu kotak chessboard dalam mm
SQUARE_SIZE = 25.0


# ============================================================
# 2. MEMBUAT OBJECT POINTS
# ============================================================

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE


# ============================================================
# 3. MENCARI SEMUA FOTO
# ============================================================

images = glob.glob("images/*")

# Hanya ambil JPG/JPEG
images = [
    f for f in images
    if f.lower().endswith((".jpg", ".jpeg"))
]

# Hapus kemungkinan file duplikat
images = sorted(set(images))

print("Jumlah gambar:", len(images))
print()


# ============================================================
# 4. MENYIAPKAN LIST POINT
# ============================================================

objpoints = []   # Titik 3D chessboard
imgpoints = []   # Titik 2D pada gambar

image_names = []


# ============================================================
# 5. DETEKSI CORNER CHESSBOARD
# ============================================================

for fname in images:

    img = cv2.imread(fname)

    if img is None:
        print("GAGAL membaca:", fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cari corner chessboard
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None
    )

    if ret:

        # Perbaiki posisi corner agar lebih presisi
        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        image_names.append(fname)

        print("OK :", fname)

    else:
        print("GAGAL :", fname)


print()
print("Corner berhasil ditemukan:", len(objpoints))
print()


# ============================================================
# 6. CEK APAKAH DATA CUKUP
# ============================================================

if len(objpoints) < 3:
    print("Data tidak cukup untuk kalibrasi.")
    print("Pastikan minimal beberapa foto chessboard berhasil.")
    exit()


# ============================================================
# 7. KALIBRASI KAMERA
# ============================================================

# Ambil ukuran gambar
sample_img = cv2.imread(image_names[0])
gray_sample = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

image_size = gray_sample.shape[::-1]

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)


# ============================================================
# 8. MENAMPILKAN HASIL KALIBRASI
# ============================================================

print("========================================")
print("        HASIL KALIBRASI KAMERA")
print("========================================")
print()

print("Camera Matrix:")
print(camera_matrix)
print()

print("Distortion Coefficients:")
print(dist_coeffs)
print()

print("RMS Reprojection Error:")
print(ret)
print()


# ============================================================
# 9. MENGHITUNG ERROR SETIAP FOTO
# ============================================================

print("========================================")
print("       ERROR SETIAP FOTO")
print("========================================")
print()

photo_errors = []

for i in range(len(objpoints)):

    # Proyeksikan kembali object point ke gambar
    projected_points, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    # Ubah menjadi bentuk (jumlah_corner, 2)
    detected = imgpoints[i].reshape(-1, 2)
    projected = projected_points.reshape(-1, 2)

    # Hitung Euclidean error setiap corner
    errors = np.sqrt(
        np.sum(
            (detected - projected) ** 2,
            axis=1
        )
    )

    # RMS error untuk satu foto
    rms_error = np.sqrt(
        np.mean(errors ** 2)
    )

    photo_errors.append(
        (image_names[i], rms_error)
    )


# ============================================================
# 10. URUTKAN ERROR DARI TERBESAR
# ============================================================

photo_errors.sort(
    key=lambda x: x[1],
    reverse=True
)


print("ERROR TERBESAR → TERKECIL")
print()

for fname, error in photo_errors:

    print(
        f"{os.path.basename(fname):15s} : "
        f"{error:.4f} px"
    )


# ============================================================
# 11. MENYIMPAN HASIL KALIBRASI
# ============================================================

np.savez(
    "calib.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

print()
print("========================================")
print("File kalibrasi berhasil disimpan:")
print("calib.npz")
print("========================================")