#!/usr/bin/env python3
"""
TensorFlow trainer for tunx v1, v2, v3, v4 reference models.

XLA is always enabled (via tf.function(jit_compile=True) on every train/eval step,
plus TF_XLA_FLAGS=--tf_xla_auto_jit=2 environment variable).
Supports benchmark mode (warmup + measured steps with per-phase timing),
detailed memory usage breakdown, and per-batch / per-epoch CSV logging.

Usage:
    python tensorflow_trainer.py --config ../configs/tunx_v1.json
    python tensorflow_trainer.py --config ../configs/tunx_v1.json --benchmark

Environment variables (mirror torch trainer):
    EPOCHS, BATCH_SIZE, LR_INITIAL, IMAGENET100_ROOT
"""

import argparse
import csv
import datetime
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Prepend nvidia-wheel CUDA lib dirs to LD_LIBRARY_PATH *before* TF imports.
# TF (unlike PyTorch) does not do this automatically. Without it TF says
# "Cannot dlopen some GPU libraries" even though tensorflow[and-cuda] has
# already installed libcudart.so.12 etc. inside the venv.
# If LD_LIBRARY_PATH is changed we re-exec so the dynamic linker picks it up.
# ---------------------------------------------------------------------------
def _fix_cuda_ld_path() -> None:
    site_pkgs = next(
        (p for p in sys.path if "site-packages" in p and os.path.isdir(p)), None
    )
    if site_pkgs is None:
        return

    nvidia_root = Path(site_pkgs) / "nvidia"
    if not nvidia_root.exists():
        return

    # Collect every lib/ sub-directory that actually has .so files
    extra: list[str] = []
    for lib_dir in sorted(nvidia_root.glob("*/lib")):
        if lib_dir.is_dir() and any(lib_dir.glob("*.so*")):
            extra.append(str(lib_dir))

    if not extra:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_dirs = set(current.split(":")) if current else set()
    new_dirs = [d for d in extra if d not in current_dirs]

    if not new_dirs:
        return  # already set, no re-exec needed

    new_ld = ":".join(new_dirs) + (":" + current if current else "")
    os.environ["LD_LIBRARY_PATH"] = new_ld
    # Re-exec with the updated env so the dynamic linker sees the new paths
    os.execv(sys.executable, [sys.executable] + sys.argv)


_fix_cuda_ld_path()

import tensorflow as tf  # noqa: E402  (must come after LD_LIBRARY_PATH fix)
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# ===========================================================================
# XLA / GPU setup  (always on)
# ===========================================================================

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
# Use the async CUDA allocator: avoids BFC fragmentation that causes OOM
# even when enough total GPU memory is available (TF's own recommendation).
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f">>> GPUs found: {[g.name for g in gpus]}")
else:
    print(">>> No GPU found – running on CPU.")


# ===========================================================================
# Memory utilities
# ===========================================================================

def _gpu_memory_info() -> Dict[str, float]:
    """Returns {current_b, peak_b} in bytes; zeros when no GPU."""
    info: Dict[str, float] = {"current_b": 0.0, "peak_b": 0.0}
    if not gpus:
        return info
    try:
        mem = tf.config.experimental.get_memory_info("GPU:0")
        info["current_b"] = float(mem.get("current", 0))
        info["peak_b"]    = float(mem.get("peak", 0))
    except Exception:
        pass
    return info


def _reset_peak_memory():
    if not gpus:
        return
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


def _model_param_bytes(model: tf.keras.Model) -> int:
    total = 0
    for v in model.trainable_variables:
        total += int(np.prod(v.shape)) * v.dtype.size
    return total


def _optimizer_state_bytes(optimizer: tf.keras.optimizers.Optimizer) -> int:
    total = 0
    try:
        for v in optimizer.variables():
            total += int(np.prod(v.shape)) * v.dtype.size
    except Exception:
        pass
    return total


def _grad_bytes(grads: List[Optional[tf.Tensor]]) -> int:
    total = 0
    for g in grads:
        if g is not None:
            total += int(np.prod(g.shape)) * g.dtype.size
    return total


def _print_memory_breakdown(
    tag: str,
    param_b: int,
    grad_b: int,
    optim_b: int,
    peak_fwd_b: float,
    current_after_fwd_b: float,
    peak_bwd_b: float,
):
    GiB = 1024 ** 3
    baseline_b   = param_b + grad_b + optim_b
    activations_b = max(0.0, current_after_fwd_b - baseline_b)
    print(
        f"\n--- Memory Breakdown ({tag}) ---"
        f"\n  Baseline (params + grads + optim)       : {baseline_b / GiB:.3f} GiB"
        f"\n    -> Model Parameters                   : {param_b / GiB:.3f} GiB"
        f"\n    -> Gradients                          : {grad_b / GiB:.3f} GiB"
        f"\n    -> Optimizer States                   : {optim_b / GiB:.3f} GiB"
        f"\n  Peak Memory During Forward Pass         : {peak_fwd_b / GiB:.3f} GiB"
        f"\n  Saved Activations (End of Forward)      : {activations_b / GiB:.3f} GiB"
        f"\n  Peak Memory During Backward Pass        : {peak_bwd_b / GiB:.3f} GiB"
        f"\n----------------------------------------"
    )


# ===========================================================================
# Dataset  (ImageNet-100 – same directory layout as torch trainer)
# ===========================================================================

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def _preprocess_train(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, [256, 256])
    img = tf.image.random_crop(img, [224, 224, 3])
    img = tf.image.random_flip_left_right(img)
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img, label


def _preprocess_val(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, [256, 256])
    img = img[16:240, 16:240, :]   # centre crop 224x224
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img, label


def build_imagenet100_dataset(
    root: str,
    train: bool,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[tf.data.Dataset, int]:
    root = Path(root)
    all_classes: set = set()
    train_subdirs = ["train.X1", "train.X2", "train.X3", "train.X4"]

    for sd in train_subdirs:
        d = root / sd
        if d.exists():
            all_classes.update(x.name for x in d.iterdir() if x.is_dir())

    classes = sorted(all_classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    paths: List[str] = []
    labels: List[int] = []

    if train:
        for sd in train_subdirs:
            data_dir = root / sd
            if not data_dir.exists():
                continue
            for cls in classes:
                img_dir = data_dir / cls
                if not img_dir.exists():
                    continue
                for ext in ["*.JPEG", "*.jpg", "*.png"]:
                    for p in img_dir.glob(ext):
                        paths.append(str(p))
                        labels.append(class_to_idx[cls])
    else:
        data_dir = root / "val.X"
        for cls in classes:
            img_dir = data_dir / cls
            if not img_dir.exists():
                continue
            for ext in ["*.JPEG", "*.jpg", "*.png"]:
                for p in img_dir.glob(ext):
                    paths.append(str(p))
                    labels.append(class_to_idx[cls])

    num_samples = len(paths)
    if num_samples == 0:
        raise FileNotFoundError(
            f"No images found under {root} ({'train' if train else 'val'})"
        )

    path_ds = tf.data.Dataset.from_tensor_slices(
        (paths, [int(l) for l in labels])
    )

    if train:
        path_ds = path_ds.shuffle(
            buffer_size=min(num_samples, 50000), reshuffle_each_iteration=True
        )
        ds = path_ds.map(_preprocess_train, num_parallel_calls=num_workers)
    else:
        ds = path_ds.map(_preprocess_val, num_parallel_calls=num_workers)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, num_samples


# ===========================================================================
# Models  (TF/Keras, NHWC, mirrors PyTorch architecture exactly)
# ===========================================================================

# ---------------------------------------------------------------------------
# Tunx V1
# ---------------------------------------------------------------------------

def build_tunx_v1(num_classes: int = 100) -> tf.keras.Model:
    """
    Tunx V1 – three parallel up-/down-sampling branches merged by addition.
    Input: (B, 224, 224, 3) NHWC.
    """
    inp = tf.keras.Input(shape=(224, 224, 3), name="input")

    # Stem: [B,112,112,16] -> relu -> maxpool4x4s4 -> [B,28,28,16]
    x = tf.keras.layers.Conv2D(16, 7, strides=2, padding="same", use_bias=False, name="conv1")(inp)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn1")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=4, strides=4, padding="valid", name="pool1")(x)
    # x: [B,28,28,16]

    # Branch 1: up2 -> up4 -> conv3x3 -> maxpool4x4 -> [B,56,56,64]
    b1 = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding="same", use_bias=False, name="b1_up1")(x)
    b1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b1_bn1")(b1)
    b1 = tf.keras.layers.Activation("relu")(b1)
    b1 = tf.keras.layers.Conv2DTranspose(64, 4, strides=4, padding="same", use_bias=False, name="b1_up2")(b1)
    b1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b1_bn2")(b1)
    b1 = tf.keras.layers.Activation("relu")(b1)
    b1 = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="b1_down")(b1)
    b1 = tf.keras.layers.MaxPool2D(pool_size=4, strides=4, padding="valid", name="b1_pool")(b1)
    # b1: [B,56,56,64]

    # Branch 2: trans2 -> up2 -> conv -> conv -> negate -> maxpool2x2 -> [B,56,56,64]
    b2 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same", use_bias=False, name="b2_trans")(x)
    b2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b2_bn1")(b2)
    b2 = tf.keras.layers.Activation("relu")(b2)
    b2 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same", use_bias=False, name="b2_up2")(b2)
    b2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b2_bn2")(b2)
    b2 = tf.keras.layers.Activation("relu")(b2)
    b2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="b2_conv")(b2)
    b2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b2_bn3")(b2)
    b2 = tf.keras.layers.Activation("relu")(b2)
    b2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="b2_conv2")(b2)
    b2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b2_bn4")(b2)
    b2 = tf.keras.layers.Activation("relu")(b2)
    b2 = tf.keras.layers.Lambda(lambda t: -t, name="b2_negate")(b2)
    b2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="b2_pool")(b2)
    # b2: [B,56,56,64]

    # Branch 3: up2 -> up4 -> avgpool4x4 -> conv3x3 -> [B,56,56,64]
    b3 = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding="same", use_bias=False, name="b3_up1")(x)
    b3 = tf.keras.layers.Conv2DTranspose(128, 4, strides=4, padding="same", use_bias=False, name="b3_up2")(b3)
    b3 = tf.keras.layers.AveragePooling2D(pool_size=4, strides=4, padding="valid", name="b3_pool")(b3)
    b3 = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="b3_down1")(b3)
    b3 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b3_bn2")(b3)
    b3 = tf.keras.layers.Activation("relu")(b3)
    # b3: [B,56,56,64]

    y = tf.keras.layers.Add(name="merge")([b1, b2, b3])
    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.Flatten(name="flatten")(y)
    out = tf.keras.layers.Dense(num_classes, use_bias=True, name="fc")(y)
    return tf.keras.Model(inputs=inp, outputs=out, name="tunx_v1")


# ---------------------------------------------------------------------------
# Tunx V2
# ---------------------------------------------------------------------------

def _tunx_v2_block(x: tf.Tensor, prefix: str) -> tf.Tensor:
    """TunxV2Block: dual-upsampling with max+avg fusion, output 128 channels."""
    # Arm 1: up2 -> maxpool3x3s1 (same)
    b1 = tf.keras.layers.Conv2DTranspose(
        256, 2, strides=2, padding="same", use_bias=False, name=f"{prefix}_up1")(x)
    b1 = tf.keras.layers.MaxPool2D(
        pool_size=3, strides=1, padding="same", name=f"{prefix}_pool1")(b1)

    # Arm 2: up4 -> left(negate+max2x2) and right(avg2x2)
    b2 = tf.keras.layers.Conv2DTranspose(
        256, 4, strides=4, padding="same", use_bias=False, name=f"{prefix}_up2")(x)
    b2_left = tf.keras.layers.MaxPool2D(
        pool_size=2, strides=2, padding="valid", name=f"{prefix}_left_pool")(
        tf.keras.layers.Lambda(lambda t: -t, name=f"{prefix}_neg")(b2)
    )
    b2_right = tf.keras.layers.AveragePooling2D(
        pool_size=2, strides=2, padding="valid", name=f"{prefix}_right_pool")(b2)

    c = tf.keras.layers.Add(name=f"{prefix}_add")([b1, b2_left, b2_right])
    return tf.keras.layers.Conv2D(
        128, 3, strides=1, padding="same", use_bias=False, name=f"{prefix}_conv")(c)


def build_tunx_v2(num_classes: int = 100) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(224, 224, 3), name="input")

    # Stem [B,28,28,16]
    x = tf.keras.layers.Conv2D(16, 7, strides=2, padding="same", use_bias=False, name="conv1")(inp)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn1")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=4, strides=4, padding="valid", name="pool1")(x)

    # Branch 1 [B,56,56,128]
    b1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same", use_bias=False, name="b1_up1")(x)
    b1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b1_bn1")(b1)
    b1 = tf.keras.layers.Activation("relu")(b1)
    b1 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same", use_bias=False, name="b1_up2")(b1)
    b1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b1_bn2")(b1)
    b1 = tf.keras.layers.Activation("relu")(b1)
    b1 = tf.keras.layers.Conv2D(128, 3, strides=1, padding="same", use_bias=False, name="b1_down")(b1)
    b1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="b1_pool")(b1)

    # Branch 2 [B,56,56,128]
    b2 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same", use_bias=False, name="b2_trans")(x)
    b2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b2_bn1")(b2)
    b2 = tf.keras.layers.Activation("relu")(b2)
    b2 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same", use_bias=False, name="b2_up2")(b2)
    b2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b2_bn2")(b2)
    b2 = tf.keras.layers.Activation("relu")(b2)
    b2 = tf.keras.layers.Conv2D(128, 3, strides=1, padding="same", use_bias=False, name="b2_conv")(b2)
    b2 = tf.keras.layers.Conv2D(128, 3, strides=1, padding="same", use_bias=False, name="b2_conv2")(b2)
    b2 = tf.keras.layers.Lambda(lambda t: -t, name="b2_negate")(b2)
    b2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="b2_pool")(b2)

    # Branch 3 via TunxV2Block [B,56,56,128]
    b3 = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="b3_conv")(x)
    b3 = tf.keras.layers.Conv2DTranspose(128, 5, strides=1, padding="same", use_bias=False, name="b3_up1")(b3)
    b3 = _tunx_v2_block(b3, prefix="b3_block")

    y = tf.keras.layers.Add(name="merge")([b1, b2, b3])
    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.Flatten(name="flatten")(y)
    out = tf.keras.layers.Dense(num_classes, use_bias=True, name="fc")(y)
    return tf.keras.Model(inputs=inp, outputs=out, name="tunx_v2")


# ---------------------------------------------------------------------------
# Tunx V3
# ---------------------------------------------------------------------------

def _tunx_v3_block(x: tf.Tensor, prefix: str) -> tf.Tensor:
    """TunxV3Block: dual-upsampling with concatenate instead of add."""
    b1 = tf.keras.layers.Conv2DTranspose(
        128, 2, strides=2, padding="same", use_bias=False, name=f"{prefix}_up1")(x)
    b1 = tf.keras.layers.MaxPool2D(
        pool_size=4, strides=4, padding="valid", name=f"{prefix}_pool1")(b1)

    b2 = tf.keras.layers.Conv2DTranspose(
        128, 2, strides=2, padding="same", use_bias=False, name=f"{prefix}_up2")(x)
    b2_left = tf.keras.layers.MaxPool2D(
        pool_size=4, strides=4, padding="valid", name=f"{prefix}_left_pool")(
        tf.keras.layers.Lambda(lambda t: -t, name=f"{prefix}_neg")(b2)
    )
    b2_right = tf.keras.layers.AveragePooling2D(
        pool_size=4, strides=4, padding="valid", name=f"{prefix}_right_pool")(b2)

    c = tf.keras.layers.Concatenate(axis=-1, name=f"{prefix}_cat")([b1, b2_left, b2_right])
    return tf.keras.layers.Conv2D(
        64, 3, strides=1, padding="same", use_bias=False, name=f"{prefix}_conv")(c)


def build_tunx_v3(num_classes: int = 100) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(224, 224, 3), name="input")

    # Stem [B,28,28,16]
    x = tf.keras.layers.Conv2D(16, 7, strides=2, padding="same", use_bias=False, name="conv1")(inp)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn1")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=4, strides=4, padding="valid", name="pool1")(x)

    # Branch 1 [B,56,56,64]
    b1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same", use_bias=False, name="b1_up1")(x)
    b1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b1_bn1")(b1)
    b1 = tf.keras.layers.Activation("relu")(b1)
    b1 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same", use_bias=False, name="b1_up2")(b1)
    b1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b1_bn2")(b1)
    b1 = tf.keras.layers.Activation("relu")(b1)
    b1 = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="b1_down")(b1)
    b1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="b1_pool")(b1)

    # Branch 2 [B,56,56,64]
    b2 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same", use_bias=False, name="b2_trans")(x)
    b2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b2_bn1")(b2)
    b2 = tf.keras.layers.Activation("relu")(b2)
    b2 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same", use_bias=False, name="b2_up2")(b2)
    b2 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="b2_bn2")(b2)
    b2 = tf.keras.layers.Activation("relu")(b2)
    b2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="b2_conv")(b2)
    b2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="b2_conv2")(b2)
    b2 = tf.keras.layers.Lambda(lambda t: -t, name="b2_negate")(b2)
    b2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="b2_pool")(b2)

    # Branch 3 via TunxV3Block [B,56,56,64]
    b3 = tf.keras.layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="b3_conv")(x)
    b3 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same", use_bias=False, name="b3_up1")(b3)
    b3 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same", use_bias=False, name="b3_up2")(b3)
    b3 = _tunx_v3_block(b3, prefix="b3_block")

    y = tf.keras.layers.Add(name="merge")([b1, b2, b3])
    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.Flatten(name="flatten")(y)
    out = tf.keras.layers.Dense(num_classes, use_bias=True, name="fc")(y)
    return tf.keras.Model(inputs=inp, outputs=out, name="tunx_v3")


# ---------------------------------------------------------------------------
# Tunx V4
# ---------------------------------------------------------------------------

def _recursive_wide_fork_join(
    x: tf.Tensor,
    depth: int,
    prefix: str,
) -> tf.Tensor:
    """Recursive fork-join block – exact port of RecursiveWideForkJoin from PyTorch."""
    kOutputChannels = 64
    kLocalChannels  = 64 + depth * 256
    kJoinChannels   = 832 - depth * 256

    if depth == 0:
        expanded = tf.keras.layers.Conv2DTranspose(
            kJoinChannels, 2, strides=2, padding="same",
            use_bias=False, name=f"{prefix}_expand")(x)
        left = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=2, padding="valid", name=f"{prefix}_left_pool")(
            tf.keras.layers.Lambda(lambda t: -t, name=f"{prefix}_neg")(expanded)
        )
        right = tf.keras.layers.AveragePooling2D(
            pool_size=2, strides=2, padding="valid", name=f"{prefix}_right_pool")(expanded)
        mid = tf.keras.layers.Conv2D(
            kJoinChannels, 3, strides=2, padding="same",
            use_bias=False, name=f"{prefix}_mid_conv")(expanded)
        joined = tf.keras.layers.Add(name=f"{prefix}_join")([left, right, mid])
        return tf.keras.layers.Conv2D(
            kOutputChannels, 1, strides=1, padding="same",
            use_bias=False, name=f"{prefix}_compress")(joined)
    else:
        local_ = tf.keras.layers.Conv2DTranspose(
            kLocalChannels, 2, strides=2, padding="same",
            use_bias=False, name=f"{prefix}_local_expand")(x)
        local_ = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=2, padding="valid",
            name=f"{prefix}_local_pool")(local_)
        local_ = tf.keras.layers.Conv2D(
            kJoinChannels, 1, strides=1, padding="same",
            use_bias=False, name=f"{prefix}_local_widen")(local_)

        nested = tf.keras.layers.Conv2D(
            kOutputChannels, 1, strides=1, padding="same",
            use_bias=False, name=f"{prefix}_nested_seed")(x)
        nested = _recursive_wide_fork_join(nested, depth - 1, prefix=f"{prefix}_d{depth-1}")
        nested = tf.keras.layers.Conv2D(
            kJoinChannels, 1, strides=1, padding="same",
            use_bias=False, name=f"{prefix}_nested_widen")(nested)

        joined = tf.keras.layers.Add(name=f"{prefix}_join")([local_, nested])
        return tf.keras.layers.Conv2D(
            kOutputChannels, 1, strides=1, padding="same",
            use_bias=False, name=f"{prefix}_compress")(joined)


def build_tunx_v4(num_classes: int = 100) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(224, 224, 3), name="input")

    # Stem [B,56,56,64]
    x = tf.keras.layers.Conv2D(32, 7, strides=2, padding="same", use_bias=False, name="stem_conv")(inp)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="stem_bn")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid", name="stem_pool")(x)
    x = tf.keras.layers.Conv2D(64, 1, strides=1, padding="same", use_bias=False, name="stem_project")(x)

    x = _recursive_wide_fork_join(x, depth=3, prefix="rfj")

    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=4, strides=4, padding="valid", name="out_pool")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    out = tf.keras.layers.Dense(num_classes, use_bias=True, name="fc")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="tunx_v4")


# ===========================================================================
# Model registry
# ===========================================================================

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Returns training config dict for the given tunx model."""
    imagenet100_root = os.getenv("IMAGENET100_ROOT", "data/imagenet-100")

    builders = {
        "tunx_v1": build_tunx_v1,
        "tunx_v2": build_tunx_v2,
        "tunx_v3": build_tunx_v3,
        "tunx_v4": build_tunx_v4,
    }

    if model_name not in builders:
        raise ValueError(
            f"Unknown model: {model_name!r}. Available: {list(builders.keys())}"
        )

    def _make_builder(name):
        def _build():
            return builders[name](num_classes=100)
        return _build

    cfg: Dict[str, Any] = {
        "model_builder":    _make_builder(model_name),
        "imagenet100_root": imagenet100_root,
        "num_classes":      100,
        "epochs":           int(os.getenv("EPOCHS", "90")),
        "batch_size":       int(os.getenv("BATCH_SIZE", "64")),
        "lr":               float(os.getenv("LR_INITIAL", "0.001")),
        "num_workers":      4,
        "optimizer_type":   "adam",
        "scheduler_type":   "step_lr",
    }
    return cfg


# ===========================================================================
# XLA-compiled train / eval steps
# ===========================================================================

def make_train_step(model, optimizer, loss_fn):
    """Returns a tf.function(jit_compile=True) train step."""

    @tf.function(jit_compile=True)
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            # Cast to float32 for numerically stable loss (required for BF16/FP16 mixed precision)
            loss = loss_fn(labels, tf.cast(logits, tf.float32))
            loss = loss + tf.add_n(model.losses) if model.losses else loss
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        preds = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        return loss, preds, grads

    return train_step


def make_eval_step(model, loss_fn):
    """Returns a tf.function(jit_compile=True) eval step."""

    @tf.function(jit_compile=True)
    def eval_step(images, labels):
        logits = model(images, training=False)
        loss = loss_fn(labels, tf.cast(logits, tf.float32))
        preds = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        return loss, preds

    return eval_step


# ===========================================================================
# Benchmark mode
# ===========================================================================

def run_benchmark(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn,
    dataset: tf.data.Dataset,
    warmup_steps: int = 50,
    measure_steps: int = 2000,
):
    """
    Benchmark: warmup_steps XLA-compiled iterations followed by measure_steps
    timed iterations with forward / backward / optimizer phase breakdown.
    """
    print(f">>> Starting Benchmark Mode "
          f"({warmup_steps} warmup + {measure_steps} measured steps)...")

    # Grab a single fixed batch
    for images, labels in dataset:
        break
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.int32)
    batch_size = int(images.shape[0])

    train_step = make_train_step(model, optimizer, loss_fn)

    # Warm-up (triggers XLA compilation)
    print(f"Running {warmup_steps} warmup steps...")
    for _ in range(warmup_steps):
        train_step(images, labels)

    print(f"Warmup complete. Running {measure_steps} measured steps...")
    wall_start = time.perf_counter()

    for _ in range(measure_steps):
        loss, preds, grads = train_step(images, labels)

    # Force GPU sync before stopping timer
    _ = loss.numpy()
    
    wall_elapsed = time.perf_counter() - wall_start
    throughput   = (measure_steps * batch_size) / wall_elapsed

    print(f"\n{'=' * 55}")
    print(f"  Benchmark Results ({measure_steps} steps)")
    print(f"{'=' * 55}")
    print(f"  Batch size                    : {batch_size}")
    print(f"  Throughput                    : {throughput:.2f} samples/s")
    print(f"  Elapsed time ({measure_steps} steps)      : {wall_elapsed:.3f} s")
    print(f"  Total Forward time            : N/A (Fused by XLA)")
    print(f"  Total Backward time           : N/A (Fused by XLA)")
    print(f"  Total Optimizer time          : N/A (Fused by XLA)")
    print(f"  Avg step time                 : {wall_elapsed / measure_steps * 1000:.3f} ms")
    print(f"{'=' * 55}\n")


# ===========================================================================
# Training / validation loops
# ===========================================================================

def train_epoch(
    model: tf.keras.Model,
    train_step,
    optimizer: tf.keras.optimizers.Optimizer,
    dataset: tf.data.Dataset,
    loss_fn,
    epoch: int,
    batch_writer,
    progress_interval: int = 100,
    print_memory: bool = False,
) -> Tuple[float, float]:
    running_loss    = 0.0
    running_correct = 0
    running_total   = 0

    for batch_idx, (images, labels) in enumerate(dataset):
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.int32)

        step_start    = time.perf_counter()
        should_log_mem = (
            print_memory
            and bool(gpus)
            and (batch_idx + 1) % progress_interval == 0
        )

        if should_log_mem:
            _reset_peak_memory()

        loss_val, preds, grads = train_step(images, labels)

        if should_log_mem:
            mem_fwd = _gpu_memory_info()
            peak_fwd_b        = mem_fwd["peak_b"]
            current_after_fwd = mem_fwd["current_b"]
            _reset_peak_memory()
            mem_bwd    = _gpu_memory_info()
            peak_bwd_b = mem_bwd["peak_b"]

            param_b = _model_param_bytes(model)
            grad_b  = _grad_bytes(grads)
            optim_b = _optimizer_state_bytes(optimizer)

            _print_memory_breakdown(
                tag=f"Batch {batch_idx + 1}",
                param_b=param_b,
                grad_b=grad_b,
                optim_b=optim_b,
                peak_fwd_b=peak_fwd_b,
                current_after_fwd_b=current_after_fwd,
                peak_bwd_b=peak_bwd_b,
            )

        step_ms    = int((time.perf_counter() - step_start) * 1000)
        batch_size = int(images.shape[0])
        correct    = int(tf.reduce_sum(
            tf.cast(tf.equal(preds, labels), tf.int32)
        ).numpy())

        running_loss    += float(loss_val) * batch_size
        running_correct += correct
        running_total   += batch_size

        batch_loss = float(loss_val)
        batch_acc  = 100.0 * correct / batch_size

        batch_writer.writerow([
            epoch, batch_idx + 1,
            f"{batch_loss:.6f}", f"{batch_acc:.4f}", step_ms,
        ])

        if (batch_idx + 1) % progress_interval == 0:
            mem_info = ""
            if gpus:
                m = _gpu_memory_info()
                cur_gib  = m["current_b"] / (1024 ** 3)
                peak_gib = m["peak_b"]    / (1024 ** 3)
                mem_info = (
                    f" | GPU Cur: {cur_gib:.3f} GiB"
                    f" | GPU Peak: {peak_gib:.3f} GiB"
                )
            print(
                f"[Train Batch {batch_idx + 1}] "
                f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}% | "
                f"Step: {step_ms}ms{mem_info}"
            )

    return running_loss / running_total, 100.0 * running_correct / running_total


def validate_epoch(
    model: tf.keras.Model,
    eval_step,
    dataset: tf.data.Dataset,
    epoch: int,
    val_writer,
) -> Tuple[float, float]:
    running_loss    = 0.0
    running_correct = 0
    running_total   = 0

    for val_step_idx, (images, labels) in enumerate(dataset):
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.int32)

        loss_val, preds = eval_step(images, labels)

        batch_size = int(images.shape[0])
        correct    = int(tf.reduce_sum(
            tf.cast(tf.equal(preds, labels), tf.int32)
        ).numpy())

        running_loss    += float(loss_val) * batch_size
        running_correct += correct
        running_total   += batch_size

        step_loss = float(loss_val)
        step_acc  = 100.0 * correct / batch_size
        val_writer.writerow(
            [epoch, val_step_idx + 1, f"{step_loss:.6f}", f"{step_acc:.4f}"]
        )

    return running_loss / running_total, 100.0 * running_correct / running_total


# ===========================================================================
# Utilities
# ===========================================================================

def load_jsonc(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        content = f.read()
    content = re.sub(r"//.*", "", content)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    return json.loads(content)


def resolve_model_name(json_name: str) -> str:
    n = json_name.lower()
    for v in ("tunx_v1", "tunx_v2", "tunx_v3", "tunx_v4"):
        if v in n or n == v:
            return v
    raise ValueError(
        f"Cannot map model name {json_name!r} to a tunx_v1..v4 model. "
        "Make sure 'model_name' in your config contains one of: "
        "tunx_v1, tunx_v2, tunx_v3, tunx_v4."
    )


# ===========================================================================
# main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TensorFlow trainer for tunx v1-v4 (XLA always on)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to JSON/JSONC config file (same format as torch trainer)"
    )
    parser.add_argument(
        "--benchmark", action="store_true", default=False,
        help="Run benchmark mode (overrides config benchmark_mode field)"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=50,
        help="Warmup steps for benchmark mode (default: 50)"
    )
    parser.add_argument(
        "--measure-steps", type=int, default=2000,
        help="Measured steps for benchmark mode (default: 2000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size from config (useful to reduce for memory-constrained GPUs)"
    )
    args = parser.parse_args()

    json_cfg   = load_jsonc(args.config)
    model_name = resolve_model_name(json_cfg.get("model_name", ""))
    cfg        = get_model_config(model_name)

    # Override with JSON config (mirrors torch trainer field names exactly)
    cfg["epochs"]               = json_cfg.get("epochs", cfg["epochs"])
    cfg["batch_size"]           = json_cfg.get("batch_size", cfg["batch_size"])
    if args.batch_size is not None:             # CLI wins over config
        cfg["batch_size"] = args.batch_size
    cfg["num_workers"]          = json_cfg.get("num_threads", cfg["num_workers"])
    cfg["progress_print_interval"] = json_cfg.get("progress_print_interval", 100)
    cfg["print_memory_usage"]   = json_cfg.get("print_memory_usage", False)
    cfg["benchmark_mode"]       = args.benchmark or json_cfg.get("benchmark_mode", False)

    if "imagenet100_root" in json_cfg:
        cfg["imagenet100_root"] = json_cfg["imagenet100_root"]

    # dtype config – mirrors torch trainer io_dtype / param_dtype / compute_dtype
    _DTYPE_MAP = {
        "fp32": "float32", "float32": "float32",
        "fp16": "float16", "float16": "float16",
        "bf16": "bfloat16", "bfloat16": "bfloat16",
    }
    param_dtype   = _DTYPE_MAP.get(json_cfg.get("param_dtype",   "FP32").lower(), "float32")
    compute_dtype = _DTYPE_MAP.get(json_cfg.get("compute_dtype", "FP32").lower(), "float32")
    cfg["param_dtype"]   = param_dtype
    cfg["compute_dtype"] = compute_dtype

    # Enable Keras mixed precision when BF16/FP16 is requested
    if param_dtype in ("bfloat16", "float16"):
        tf.keras.mixed_precision.set_global_policy(param_dtype)
        print(f">>> Mixed precision    : {param_dtype} (Keras global policy)")

    opt_cfg = json_cfg.get("optimizer", {})
    cfg["optimizer_type"] = opt_cfg.get("type", cfg["optimizer_type"])
    cfg["lr"]             = opt_cfg.get("learning_rate", cfg["lr"])
    cfg["weight_decay"]   = opt_cfg.get("weight_decay", 0.0)

    sched_cfg = json_cfg.get("scheduler", {})
    cfg["scheduler_type"]   = sched_cfg.get("type", cfg.get("scheduler_type", "step_lr"))
    cfg["scheduler_params"] = sched_cfg

    print(f">>> Framework         : TensorFlow {tf.__version__} (XLA always enabled)")
    print(f">>> Model             : {model_name}")
    print(f">>> Config            : {args.config}")
    print(f">>> Epochs            : {cfg['epochs']}")
    print(f">>> Batch size        : {cfg['batch_size']}")
    print(f">>> Learning rate     : {cfg['lr']}")
    print(f">>> Benchmark mode    : {cfg['benchmark_mode']}")
    print(f">>> Print memory      : {cfg['print_memory_usage']}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print(">>> Loading datasets...")
    train_ds, n_train = build_imagenet100_dataset(
        root=cfg["imagenet100_root"],
        train=True,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    val_ds, n_val = build_imagenet100_dataset(
        root=cfg["imagenet100_root"],
        train=False,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    print(f">>> Train samples     : {n_train}")
    print(f">>> Val samples       : {n_val}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(">>> Building model...")
    model = cfg["model_builder"]()
    # Build model graph by running a dummy forward pass
    dummy = tf.zeros([1, 224, 224, 3])
    model(dummy, training=False)
    total_params = model.count_params()
    print(f">>> Parameters        : {total_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if cfg["optimizer_type"].lower() == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0.0),
            beta_1=0.9, beta_2=0.95, epsilon=1e-8,
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg["lr"],
            beta_1=0.9, beta_2=0.999, epsilon=1e-3,
        )

    # ── Benchmark mode ────────────────────────────────────────────────────────
    if cfg["benchmark_mode"]:
        run_benchmark(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            dataset=train_ds,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )
        return

    # ── Logging ───────────────────────────────────────────────────────────────
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    batch_csv_path = os.path.join(log_dir, f"tf_{model_name}_{ts}_batch.csv")
    epoch_csv_path = os.path.join(log_dir, f"tf_{model_name}_{ts}_epoch.csv")
    val_csv_path   = os.path.join(log_dir, f"tf_{model_name}_{ts}_val.csv")

    batch_csv_file = open(batch_csv_path, "w", newline="")
    epoch_csv_file = open(epoch_csv_path, "w", newline="")
    val_csv_file   = open(val_csv_path,   "w", newline="")

    batch_writer = csv.writer(batch_csv_file)
    epoch_writer = csv.writer(epoch_csv_file)
    val_writer   = csv.writer(val_csv_file)

    batch_writer.writerow(["epoch", "step", "loss", "accuracy_pct", "time_ms"])
    epoch_writer.writerow(
        ["epoch", "train_loss", "train_accuracy_pct", "val_loss", "val_accuracy_pct"]
    )
    val_writer.writerow(["epoch", "step", "loss", "accuracy_pct"])

    # ── Compile steps (XLA via jit_compile=True) ──────────────────────────────
    train_step = make_train_step(model, optimizer, loss_fn)
    eval_step  = make_eval_step(model, loss_fn)

    # ── Scheduler state ───────────────────────────────────────────────────────
    sched_type   = cfg["scheduler_type"]
    sched_params = cfg.get("scheduler_params", {})

    steps_per_epoch = math.ceil(n_train / cfg["batch_size"])

    if sched_type == "warmup_cosine_annealing":
        warmup_steps = sched_params.get(
            "warmup_steps", int(0.1 * steps_per_epoch * cfg["epochs"])
        )
        total_steps = sched_params.get(
            "total_steps", steps_per_epoch * cfg["epochs"]
        )
        base_lr = float(sched_params.get("base_lr", cfg["lr"]))
        eta_min = float(sched_params.get("eta_min", 0.0))

        def cosine_lr(step: int) -> float:
            if step < warmup_steps:
                return base_lr * (step + 1) / max(warmup_steps, 1)
            progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
            return eta_min + (base_lr - eta_min) * (1.0 + math.cos(math.pi * progress)) / 2.0

        cfg["_cosine_lr"] = cosine_lr
        cfg["_global_step"] = 0

    elif sched_type == "step_lr":
        cfg["_step_lr_step_size"] = int(sched_params.get("step_size", 1000))
        cfg["_step_lr_gamma"]     = float(sched_params.get("gamma", 0.1))
        cfg["_current_lr"]        = cfg["lr"]

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n>>> Starting training for {cfg['epochs']} epochs...")

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\n===== Epoch {epoch}/{cfg['epochs']} =====")
        epoch_start = time.perf_counter()

        # Per-step LR update for cosine schedule is handled inside train_epoch
        # via the _global_step counter; for step_lr we do it per epoch below.

        train_loss, train_acc = train_epoch(
            model=model,
            train_step=train_step,
            optimizer=optimizer,
            dataset=train_ds,
            loss_fn=loss_fn,
            epoch=epoch,
            batch_writer=batch_writer,
            progress_interval=cfg.get("progress_print_interval", 100),
            print_memory=cfg.get("print_memory_usage", False),
        )
        batch_csv_file.flush()

        # Step LR schedule (per epoch)
        if sched_type == "step_lr":
            step_size = cfg.get("_step_lr_step_size", 1000)
            gamma     = cfg.get("_step_lr_gamma", 0.1)
            if epoch % step_size == 0:
                cfg["_current_lr"] = cfg["_current_lr"] * gamma
                optimizer.learning_rate.assign(cfg["_current_lr"])
                print(f">>> LR decayed to {cfg['_current_lr']:.6e}")

        val_loss, val_acc = validate_epoch(
            model=model,
            eval_step=eval_step,
            dataset=val_ds,
            epoch=epoch,
            val_writer=val_writer,
        )
        val_csv_file.flush()

        epoch_time = time.perf_counter() - epoch_start

        epoch_writer.writerow([
            epoch,
            f"{train_loss:.6f}", f"{train_acc:.4f}",
            f"{val_loss:.6f}",   f"{val_acc:.4f}",
        ])
        epoch_csv_file.flush()

        print(
            f"Epoch {epoch}/{cfg['epochs']} completed in {epoch_time:.2f}s\n"
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
            f"Val   Loss: {val_loss:.4f}  | Val   Acc: {val_acc:.2f}%"
        )

    batch_csv_file.close()
    epoch_csv_file.close()
    val_csv_file.close()

    print(f"\n>>> Logs saved to {log_dir}/tf_{model_name}_*_{ts}.csv")
    print(f">>> Training completed for {model_name}!")


if __name__ == "__main__":
    main()
