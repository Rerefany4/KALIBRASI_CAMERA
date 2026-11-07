import cv2
import numpy as np
import glob
import os

# =========================
# 1. Pengaturan chessboard
# =========================

CHECKERBOARD = (6, 9)

# Ukuran satu kotak = 25 mm
SQUARE_SIZE = 25.0


# =========================
# 2. Membuat Object Points
# =========================

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE


# =========================
# 3. Menyimpan titik
# =========================

objpoints = []
imgpoints = []


# =========================
# 4. Membaca semua gambar
# =========================

images = glob.glob("images/*.jpg")

print("Jumlah gambar:", len(images))


# =========================
# 5. Mencari corner
# =========================

for fname in images:

    img = cv2.imread(fname)

    if img is None:
        print("Gagal membaca:", fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None
    )

    if ret:

        # Memperhalus posisi corner
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

        objpoints.append(objp)
        imgpoints.append(corners2)

        print("OK :", os.path.basename(fname))

    else:
        print("GAGAL :", os.path.basename(fname))


# =========================
# 6. Cek jumlah foto
# =========================

print()
print("Corner berhasil ditemukan:", len(objpoints))


if len(objpoints) < 10:
    print("Foto yang berhasil dideteksi terlalu sedikit.")
    print("Tambahkan atau periksa kembali foto chessboard.")
    exit()


# =========================
# 7. Kalibrasi kamera
# =========================

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)


# =========================
# 8. Tampilkan hasil
# =========================

print("\n===== HASIL KALIBRASI =====")

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(dist_coeffs)

print("\nRMS Reprojection Error:")
print(ret)


# =========================
# 9. Simpan hasil
# =========================

np.savez(
    "calib.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

print("\nHasil disimpan sebagai: calib.npz")