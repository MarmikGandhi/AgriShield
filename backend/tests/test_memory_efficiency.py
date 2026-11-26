"""
test_memory_efficiency.py
=========================
Comprehensive tests for memory-efficient AI inference on AgriShield:
1. Verifies health endpoint.
2. Verifies single leaf scan memory overhead and schema compliance.
3. Verifies repeated scans do not leak memory (bounded memory drift).
4. Verifies GradCAM zero graph retention and attribute clearance.
5. Verifies downscaling of high-resolution leaf photos.
"""

import os
import time
import threading
import psutil
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app, seed_initial_accounts
from app.database.session import init_db
from app.api.v1.scans import get_pipeline
from app.ai.pipeline import CropHealthPipeline, get_process_memory_mb

MANIFESTS_DIR = Path("data/manifests")
MODELS_DIR = Path("models")


class PeakMemorySampler:
    """
    Samples process RSS in a dedicated background thread at high frequency (100-200 Hz).
    Provides a sampled peak RSS estimate during inference.
    NOTE: Represents sampled peak RSS, not a guaranteed mathematical supremum of instantaneous memory.
    """
    def __init__(self, interval_sec: float = 0.005):
        self.interval = interval_sec
        self.baseline_rss = 0.0
        self.sampled_peak_rss = 0.0
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self._proc = psutil.Process()

    def start(self):
        self.baseline_rss = self._proc.memory_info().rss / (1024 * 1024)
        self.sampled_peak_rss = self.baseline_rss
        self.samples = [self.baseline_rss]
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                rss = self._proc.memory_info().rss / (1024 * 1024)
                if rss > self.sampled_peak_rss:
                    self.sampled_peak_rss = rss
                self.samples.append(rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            final_rss = self._proc.memory_info().rss / (1024 * 1024)
            if final_rss > self.sampled_peak_rss:
                self.sampled_peak_rss = final_rss
        except Exception:
            pass
        return round(self.sampled_peak_rss, 2)

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()
    seed_initial_accounts()
    get_pipeline()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def auth_headers(client):
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "farmer@agri.in", "password": "farmer123"}
    )
    assert res.status_code == 200
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_endpoint_memory(client):
    """Verify health endpoint responds with 200 and reports healthy status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"


def test_single_leaf_scan_memory_bounded(client, auth_headers):
    """
    Test a single leaf scan.
    Verify that response passes and memory delta remains tightly controlled.
    """
    mem_before = get_process_memory_mb()
    img_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"

    with open(img_path, "rb") as f:
        res = client.post(
            "/api/v1/scans",
            headers=auth_headers,
            files={"file": ("scab_leaf.jpg", f, "image/jpeg")}
        )

    assert res.status_code == 200
    data = res.json()
    assert data["quality_status"].upper() == "PASSED"
    assert "crop" in data
    assert "alternatives" in data
    assert len(data["alternatives"]) > 0

    mem_after = get_process_memory_mb()
    # Memory growth during scan should be bounded (< 120 MB delta for first forward pass)
    delta = mem_after - mem_before
    assert delta < 120.0, f"Single scan memory jump was {delta} MB (from {mem_before} to {mem_after} MB)"


def test_repeated_scans_memory_stability(client, auth_headers):
    """
    Perform 5 sequential scans to test for memory leaks.
    Verifies that garbage collection and tensor deallocation keep memory stable.
    """
    img_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"

    # Execute 1 warmup scan to stabilize runtime allocator caches
    with open(img_path, "rb") as f:
        client.post(
            "/api/v1/scans",
            headers=auth_headers,
            files={"file": ("warmup.jpg", f, "image/jpeg")}
        )

    rss_readings = []
    for i in range(5):
        with open(img_path, "rb") as f:
            res = client.post(
                "/api/v1/scans",
                headers=auth_headers,
                files={"file": (f"scab_{i}.jpg", f, "image/jpeg")}
            )
        assert res.status_code == 200
        rss = get_process_memory_mb()
        rss_readings.append(rss)

    # Memory drift across 5 steady-state scans should be bounded (no continuous runaway leak)
    drift = rss_readings[-1] - rss_readings[0]
    assert drift < 35.0, f"Memory drift across 5 scans was {drift} MB (readings: {rss_readings})"


def test_gradcam_zero_graph_leakage():
    """
    Verify GradCAM cleans up its activations and gradients attributes
    and does not hold references after generate_heatmap.
    """
    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")
    assert pipeline.gradcam is not None

    img_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    import cv2
    from PIL import Image
    import torch

    bgr = cv2.imread(img_path)
    pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    t_grad = pipeline.preprocess_cls(pil_img).unsqueeze(0)
    t_grad.requires_grad = True

    norm_hm, col_hm, target_class, conf = pipeline.gradcam.generate_heatmap(
        t_grad, target_class=0, original_shape=bgr.shape[:2]
    )

    # Verify activations and gradients attributes were cleared
    assert pipeline.gradcam.activations is None
    assert pipeline.gradcam.gradients is None
    assert norm_hm is not None
    assert col_hm is not None


def test_downscaling_large_images():
    """
    Verify that large camera photos (e.g. 1024x1024 tiled) are downscaled
    and processed successfully without exceeding memory.
    """
    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")
    import cv2
    import numpy as np
    import tempfile

    img_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    real_bgr = cv2.imread(img_path)
    assert real_bgr is not None
    # Tile 4x4 into 1024x1024 with crisp edge details
    large_tiled = np.tile(real_bgr, (4, 4, 1))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_name = tmp.name
        cv2.imwrite(tmp_name, large_tiled)

    try:
        mem_before = get_process_memory_mb()
        res = pipeline.process_image(tmp_name, save_visualizations=False)
        mem_after = get_process_memory_mb()

        assert res["success"] is True
        assert res["prediction"]["crop"] is not None
        delta = mem_after - mem_before
        assert delta < 70.0
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def test_large_phone_camera_image_scan(client, auth_headers, tmp_path):
    """
    Test scanning an authentic phone-camera resolution image (4000x3000, 12 MP).
    Verifies that the image is downscaled to 640px before QualityGuard,
    preventing 150-200MB memory spikes, and finishes with bounded memory.
    """
    import cv2
    import numpy as np

    img_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    real_bgr = cv2.imread(img_path)
    assert real_bgr is not None

    # Tile natural leaf to 4000x3000 to preserve sharp leaf texture and crisp edge details
    reps_y = int(np.ceil(3000 / real_bgr.shape[0]))
    reps_x = int(np.ceil(4000 / real_bgr.shape[1]))
    phone_res = np.tile(real_bgr, (reps_y, reps_x, 1))[:3000, :4000]

    test_file = tmp_path / "phone_cam_12mp.jpg"
    cv2.imwrite(str(test_file), phone_res, [cv2.IMWRITE_JPEG_QUALITY, 85])

    sampler = PeakMemorySampler(interval_sec=0.005)
    sampler.start()
    try:
        with open(test_file, "rb") as f:
            res = client.post(
                "/api/v1/scans",
                headers=auth_headers,
                files={"file": ("phone_cam_12mp.jpg", f, "image/jpeg")}
            )
    finally:
        # Guarantee sampler thread termination even if request raises an exception
        sampled_peak_rss = sampler.stop()

    post_rss = get_process_memory_mb()
    baseline_rss = sampler.baseline_rss
    peak_delta = sampled_peak_rss - baseline_rss
    net_drift = post_rss - baseline_rss

    assert res.status_code == 200, f"Scan failed: {res.text}"
    data = res.json()
    assert data["quality_status"].upper() == "PASSED"
    assert data["crop"] is not None
    assert "confidence" in data
    assert "severity_level" in data
    assert len(data["alternatives"]) > 0

    # Validate sampled peak memory during inference vs post-request residual memory
    assert peak_delta < 140.0, (
        f"12MP phone scan sampled peak delta was {peak_delta:.2f} MB "
        f"(baseline: {baseline_rss:.2f} MB, sampled peak: {sampled_peak_rss:.2f} MB)"
    )
    assert net_drift < 120.0, (
        f"12MP phone scan net drift after cleanup was {net_drift:.2f} MB "
        f"(baseline: {baseline_rss:.2f} MB, post-request: {post_rss:.2f} MB)"
    )


def test_invalid_images_rejection(tmp_path):
    """
    Verify rejection of invalid images:
    1. Too low resolution (<128px)
    2. Too high resolution (>4096px)
    3. Severe blur
    4. Pitch black / underexposed
    5. Non-leaf noise
    """
    import cv2
    import numpy as np

    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")

    # 1. Too low resolution (64x64)
    low_res = np.full((64, 64, 3), 128, dtype=np.uint8)
    p_low = tmp_path / "low_res.jpg"
    cv2.imwrite(str(p_low), low_res)
    res_low = pipeline.process_image(str(p_low))
    assert res_low["success"] is False
    assert res_low["stage"] == "QUALITY_GUARD"
    assert res_low["quality_guard"]["reason"] == "RESOLUTION_TOO_LOW"

    # 2. Too high resolution (4200x4200 > 4096)
    from PIL import Image as PILImage
    p_high = tmp_path / "high_res.jpg"
    img_large = PILImage.new("RGB", (4200, 4200), (0, 0, 0))
    img_large.save(str(p_high), format="JPEG")
    del img_large

    res_high = pipeline.process_image(str(p_high))
    assert res_high["success"] is False
    assert res_high["stage"] == "QUALITY_GUARD"
    assert res_high["quality_guard"]["reason"] == "RESOLUTION_TOO_HIGH"

    # 3. Severe blur (Gaussian blurred leaf)
    img_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    real_bgr = cv2.imread(img_path)
    blurred = cv2.GaussianBlur(real_bgr, (41, 41), 0)
    p_blur = tmp_path / "blurred.jpg"
    cv2.imwrite(str(p_blur), blurred)
    res_blur = pipeline.process_image(str(p_blur))
    assert res_blur["success"] is False
    assert res_blur["stage"] == "QUALITY_GUARD"
    assert res_blur["quality_guard"]["reason"] == "BLUR_DETECTED"

    # 4. Pitch black / underexposed
    black = np.zeros((300, 300, 3), dtype=np.uint8)
    p_dark = tmp_path / "dark.jpg"
    cv2.imwrite(str(p_dark), black)
    res_dark = pipeline.process_image(str(p_dark))
    assert res_dark["success"] is False
    assert res_dark["stage"] == "QUALITY_GUARD"
    assert res_dark["quality_guard"]["reason"] == "UNDEREXPOSED_TOO_DARK"

    # 5. Low contrast flat canvas
    flat_canvas = np.full((300, 300, 3), 180, dtype=np.uint8)
    p_flat = tmp_path / "flat.jpg"
    cv2.imwrite(str(p_flat), flat_canvas)
    res_flat = pipeline.process_image(str(p_flat))
    assert res_flat["success"] is False
    assert res_flat["stage"] == "QUALITY_GUARD"
    assert res_flat["quality_guard"]["reason"] == "LOW_CONTRAST"

    # 6. Non-leaf high contrast image (black & white checkerboard, no foliar plant matter)
    x = np.arange(300) // 10
    y = np.arange(300) // 10
    grid = ((x[:, None] + y[None, :]) % 2 * 255).astype(np.uint8)
    nonleaf = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    p_nonleaf = tmp_path / "nonleaf.jpg"
    cv2.imwrite(str(p_nonleaf), nonleaf)
    res_nonleaf = pipeline.process_image(str(p_nonleaf))
    assert res_nonleaf["success"] is False
    assert res_nonleaf["stage"] == "QUALITY_GUARD"
    assert res_nonleaf["quality_guard"]["reason"] == "NO_VALID_LEAF_DETECTED"


def test_oversized_image_rejects_before_cv2_imread(tmp_path, monkeypatch):
    """
    Verify that an image with dimensions exceeding 4096px (e.g. 4200x4200)
    is rejected via pre-decode header inspection without ever invoking cv2.imread().
    """
    from PIL import Image as PILImage
    import cv2

    p_oversized = tmp_path / "huge_oversized.jpg"
    img = PILImage.new("RGB", (4200, 4200), (0, 0, 0))
    img.save(str(p_oversized), format="JPEG")
    del img

    imread_called = False
    original_imread = cv2.imread

    def guarded_imread(*args, **kwargs):
        nonlocal imread_called
        imread_called = True
        return original_imread(*args, **kwargs)

    monkeypatch.setattr(cv2, "imread", guarded_imread)

    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")
    res = pipeline.process_image(str(p_oversized))

    assert res["success"] is False
    assert res["stage"] == "QUALITY_GUARD"
    assert res["quality_guard"]["reason"] == "RESOLUTION_TOO_HIGH"
    assert imread_called is False, "cv2.imread() was called for an oversized image before dimension check!"


def test_image_integrity_and_formats(tmp_path):
    """
    Verify image integrity validation across multiple file formats and corruption states:
    1. Corrupt random-byte file -> rejected with CORRUPTED_IMAGE or IMAGE_DECODE_ERROR
    2. Truncated JPEG stream -> rejected with CORRUPTED_IMAGE or IMAGE_DECODE_ERROR
    3. Valid JPEG leaf image -> passes quality guard and diagnostic pipeline
    4. Valid PNG leaf image -> passes quality guard and diagnostic pipeline
    """
    import cv2
    from PIL import Image as PILImage

    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")
    real_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"

    # 1. Corrupt random-byte file
    p_corrupt = tmp_path / "corrupted_bytes.jpg"
    with open(p_corrupt, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_RAW_GARBAGE_BYTES_1234567890")

    res_corrupt = pipeline.process_image(str(p_corrupt))
    assert res_corrupt["success"] is False
    assert res_corrupt["stage"] == "QUALITY_GUARD"
    assert res_corrupt["quality_guard"]["reason"] == "CORRUPTED_IMAGE"

    # 2. Truncated JPEG stream
    with open(real_path, "rb") as f:
        full_jpeg_bytes = f.read()
    # Slice the valid JPEG to 15% of its byte stream (valid header, incomplete entropy scan)
    truncated_bytes = full_jpeg_bytes[:len(full_jpeg_bytes) // 6]
    p_truncated = tmp_path / "truncated_leaf.jpg"
    with open(p_truncated, "wb") as f:
        f.write(truncated_bytes)

    res_truncated = pipeline.process_image(str(p_truncated))
    assert res_truncated["success"] is False
    assert res_truncated["stage"] == "QUALITY_GUARD"
    assert res_truncated["quality_guard"]["reason"] == "CORRUPTED_IMAGE"

    # 3. Valid JPEG leaf image
    res_jpeg = pipeline.process_image(real_path, save_visualizations=False)
    assert res_jpeg["success"] is True
    assert res_jpeg["quality_guard"]["is_valid"] is True
    assert res_jpeg["prediction"]["crop"] is not None

    # 4. Valid PNG leaf image
    bgr_leaf = cv2.imread(real_path)
    p_png = tmp_path / "valid_leaf.png"
    cv2.imwrite(str(p_png), bgr_leaf)

    res_png = pipeline.process_image(str(p_png), save_visualizations=False)
    assert res_png["success"] is True
    assert res_png["quality_guard"]["is_valid"] is True
    assert res_png["prediction"]["crop"] is not None


def test_truncated_jpeg_rejected_before_inference(tmp_path, monkeypatch):
    """
    Verify that a truncated JPEG file (which cv2.imread might partially decode)
    is caught by pre-decode integrity validation and rejected before reaching
    classification or segmentation model inference.
    """
    import cv2

    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")
    real_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"

    with open(real_path, "rb") as f:
        full_jpeg_bytes = f.read()

    # Truncate at 25% of bytes - OpenCV's libjpeg partially decodes this with a warning to stderr
    # but returns a partial array rather than None.
    truncated_bytes = full_jpeg_bytes[:len(full_jpeg_bytes) // 4]
    p_truncated = tmp_path / "partially_decodable_truncated.jpg"
    with open(p_truncated, "wb") as f:
        f.write(truncated_bytes)

    # Verify OpenCV indeed partially decodes this file (non-None array)
    cv_partial = cv2.imread(str(p_truncated))
    assert cv_partial is not None, "Test fixture error: cv2.imread failed to return partial array for test setup"

    # Track if model forward passes are called
    cls_called = False
    seg_called = False

    original_ensemble = pipeline.ensemble
    original_unet = pipeline.unet

    def spy_ensemble(*args, **kwargs):
        nonlocal cls_called
        cls_called = True
        return original_ensemble(*args, **kwargs)

    def spy_unet(*args, **kwargs):
        nonlocal seg_called
        seg_called = True
        return original_unet(*args, **kwargs)

    if pipeline.ensemble is not None:
        monkeypatch.setattr(pipeline, "ensemble", spy_ensemble)
    if pipeline.unet is not None:
        monkeypatch.setattr(pipeline, "unet", spy_unet)

    # Process image through pipeline
    res = pipeline.process_image(str(p_truncated), save_visualizations=False)

    # Must be rejected at QUALITY_GUARD stage with CORRUPTED_IMAGE
    assert res["success"] is False
    assert res["stage"] == "QUALITY_GUARD"
    assert res["quality_guard"]["status"] == "REJECTED"
    assert res["quality_guard"]["reason"] == "CORRUPTED_IMAGE"

    # Crucial assertion: neither classification nor segmentation was ever executed
    assert cls_called is False, "Classification ensemble was invoked for a truncated image!"
    assert seg_called is False, "U-Net segmentation was invoked for a truncated image!"
