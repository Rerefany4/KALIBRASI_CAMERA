import cv2
import numpy as np

# =========================
# 1. Load hasil kalibrasi
# =========================
data = np.load("calib.npz")

camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

print("Camera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(dist_coeffs)


# =========================
# 2. Baca foto
# =========================
img = cv2.imread("images/foto1.jpg")

if img is None:
    raise FileNotFoundError("Foto tidak ditemukan!")


# =========================
# 3. Ambil ukuran foto
# =========================
h, w = img.shape[:2]

print("\nUkuran foto:")
print(w, "x", h)


# =========================
# 4. Hitung camera matrix baru
# =========================
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w, h),
    1,
    (w, h)
)


# =========================
# 5. Undistort
# =========================
undistorted = cv2.undistort(
    img,
    camera_matrix,
    dist_coeffs,
    None,
    new_camera_matrix
)


# =========================
# 6. Simpan hasil
# =========================
cv2.imwrite("undistorted.jpg", undistorted)

print("\nHasil undistortion berhasil disimpan:")
print("undistorted.jpg")