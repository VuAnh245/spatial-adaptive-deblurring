"""Readable labels for figures and exported tables (official table: docs/naming.md).

Configs, result folders and code keep their IDs (A0..A3, SA-xx, split names);
only what a reader sees in a figure or a table goes through these helpers.
Exported tables keep both the ID and the label.
"""
from __future__ import annotations

import re

# Short label (figures, table headers) and full Vietnamese name (text, captions).
MODEL_LABELS = {"A0": "Baseline", "A0+": "Baseline-L", "A1": "SFT-Free", "A2": "SFT-Map", "A3": "SFT-Map+Ch",
                "nafnet_public": "NAFNet gốc (checkpoint công khai)"}
MODEL_NAMES = {"A0": "NAFNet gốc", "A0+": "NAFNet tăng tham số", "A1": "Điều biến tự do",
               "A2": "Điều biến theo bản đồ nhòe (mô hình đề xuất)",
               "A3": "Điều biến theo bản đồ nhòe + kênh", "nafnet_public": "NAFNet gốc (checkpoint công khai)"}

COMPARISON_LABELS = {("A1", "A0"): "Phép thử điều biến",
                     ("A2", "A1"): "Phép thử giám sát (chính)",
                     ("A2", "A0+"): "Phép thử tham số",
                     ("A3", "A2"): "Phép thử nhánh kênh"}

EXPERIMENT_MODEL = {"SA-00": "nafnet_public", "SA-10": "A0", "SA-11": "A0+", "SA-12": "A1", "SA-13": "A2", "SA-14": "A3",
                    "SA-15": "A2", "SA-16": "A2", "SA-17": "A2", "SA-18": "A2", "SA-19": "A2"}
EXPERIMENT_LABELS = {
    "SA-00": "Tái lập NAFNet gốc trên GoPro",
    "SA-10": "Huấn luyện Baseline", "SA-11": "Huấn luyện Baseline-L", "SA-12": "Huấn luyện SFT-Free",
    "SA-13": "Huấn luyện SFT-Map", "SA-14": "Huấn luyện SFT-Map+Ch",
    "SA-15": "SFT-Map chỉ ở decoder", "SA-16": "SFT-Map ở encoder + middle",
    "SA-17": "SFT-Map cắt gradient vào bộ ước lượng", "SA-18": "SFT-Map + loss FFT",
    "SA-19": "SFT-Map + loss perceptual",
    "SA-20": "Trần bản đồ thật", "SA-21": "Phép thử xáo trộn", "SA-22": "Chất lượng bản đồ ước lượng",
    "SA-23": "PSNR vùng nhòe nặng", "SA-24": "Phân tích theo số vùng và độ không đồng nhất",
    "SA-25": "Phân tích theo mức nhòe và ngoài phân phối", "SA-26": "Độ bền với nhiễu",
    "SA-27": "Kernel thẳng và kernel cong", "SA-28": "Ảnh nhòe thật (GoPro, RealBlur-J)",
    "SA-29": "So với phương pháp cổ điển (Wiener, Richardson-Lucy)", "SA-30": "Trực quan hệ số điều biến",
    "SA-31": "Hiệu năng tính toán",
}

SPLIT_LABELS = {"val": "Tập kiểm định", "test_in": "Tổng hợp trong phân phối",
                "test_ood": "Tổng hợp ngoài phân phối", "test_straight": "Kernel thẳng",
                "test_curved": "Kernel cong", "gopro_test": "GoPro thật", "realblur_j_test": "RealBlur-J"}

MODE_LABELS = {"pred": "Bản đồ ước lượng", "oracle": "Trần bản đồ thật", "shuffle": "Phép thử xáo trộn"}
POOL_LABELS = {"tlc": "Suy luận cục bộ (TLC)", "global": "Suy luận toàn cục"}
METRIC_LABELS = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS",
                 "psnr_local_high": "PSNR vùng nhòe nặng (dB)",
                 "psnr_transition": "PSNR vùng chuyển tiếp (dB)",  # band of large |grad S|
                 "map_mae": "Sai số bản đồ", "map_spearman": "Tương quan bản đồ"}
UNITS = {"params": "triệu tham số", "macs": "GMACs", "latency": "ms"}


def variant_of(key: str) -> str | None:
    """'A2', 'SA-13' or a file stem containing 'SA-13' -> 'A2'."""
    if key in MODEL_NAMES:
        return key
    m = re.search(r"SA-\d\d", key)
    return EXPERIMENT_MODEL.get(m.group(0)) if m else None


def experiment_of(key: str) -> str | None:
    m = re.search(r"SA-\d\d", key)
    return m.group(0) if m else None


def model_label(key: str) -> str:
    """Variant ID, experiment ID or any name containing one -> short label ('SFT-Map')."""
    if key in MODEL_LABELS:
        return MODEL_LABELS[key]
    v = variant_of(key)
    return MODEL_LABELS[v] if v else key


def model_name(key: str) -> str:
    v = variant_of(key)
    return MODEL_NAMES[v] if v else key


def comparison_label(a: str, b: str) -> str:
    """Comparison of two models/experiments, e.g. ('SA-13', 'SA-12') -> 'Phép thử giám sát (chính)'."""
    va, vb = variant_of(a), variant_of(b)
    return COMPARISON_LABELS.get((va, vb), f"{model_label(a)} so với {model_label(b)}")


def experiment_label(exp_id: str) -> str:
    return EXPERIMENT_LABELS.get(exp_id, exp_id)


def split_label(split: str) -> str:
    """'test_in' -> 'Tổng hợp trong phân phối'; '_noise2x' suffix -> ' (nhiễu ×2)'."""
    m = re.fullmatch(r"(.+?)(?:_noise(\d+(?:\.\d+)?)x)?", split)
    base, mult = m.group(1), m.group(2)
    label = SPLIT_LABELS.get(base, base)
    return f"{label} (nhiễu ×{mult})" if mult else label


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def vn_num(x: float, digits: int = 2) -> str:
    """Vietnamese decimal comma for figure text: 0.45 -> '0,45'."""
    return f"{x:.{digits}f}".replace(".", ",")
