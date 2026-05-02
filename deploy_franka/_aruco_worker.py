"""Subprocess worker for ChArUco detection (isolated from pyzed to avoid segfault).

Must be run with the .venv Python (OpenCV 4.13) — the conda env's OpenCV 4.6
segfaults on aruco.detectMarkers.
"""
import pickle
import struct
import sys

import cv2
import numpy as np
from cv2 import aruco

_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
_board = aruco.CharucoBoard((14, 9), 0.020, 0.015, _dict)

_params = aruco.DetectorParameters()
_params.minMarkerPerimeterRate = 0.005
_params.adaptiveThreshWinSizeMin = 3
_params.adaptiveThreshWinSizeMax = 30
_params.adaptiveThreshWinSizeStep = 5
_params.adaptiveThreshConstant = 7
_params.minCornerDistanceRate = 0.02
_params.polygonalApproxAccuracyRate = 0.05

_detector = aruco.ArucoDetector(_dict, _params)
_charuco_params = aruco.CharucoParameters()
_charuco_params.minMarkers = 2
_charuco_detector = aruco.CharucoDetector(_board, _charuco_params, _params)


def _read_msg():
    raw = sys.stdin.buffer.read(4)
    if len(raw) < 4:
        return None
    size = struct.unpack(">I", raw)[0]
    data = sys.stdin.buffer.read(size)
    return pickle.loads(data)


def _write_msg(obj):
    data = pickle.dumps(obj, protocol=2)
    sys.stdout.buffer.write(struct.pack(">I", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def detect_and_pose(image, camera_matrix, dist_coeffs):
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    corners, ids, _, _ = _charuco_detector.detectBoard(gray)
    if corners is None or len(corners) < 4:
        return None

    ok, rvec, tvec = cv2.solvePnP(
        _board.getChessboardCorners()[ids.flatten()],
        corners, camera_matrix, dist_coeffs,
    )
    if not ok:
        return {"rvec": None, "tvec": None}
    return {
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
    }


if __name__ == "__main__":
    while True:
        msg = _read_msg()
        if msg is None:
            break
        image = np.frombuffer(msg["image"], dtype=msg["image_dtype"]).reshape(msg["image_shape"])
        K = np.array(msg["K"], dtype=np.float64)
        dist = np.array(msg["dist"], dtype=np.float64)
        result = detect_and_pose(image, K, dist)
        _write_msg(result)
