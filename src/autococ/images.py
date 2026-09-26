"""Bounded image reuse, invalidated when a screenshot/template changes on disk."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def _key(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=2)
def _frame(path, modified, size):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def read_frame(path, flags=cv2.IMREAD_COLOR):
    image = _frame(*_key(path))
    if image is not None and flags == cv2.IMREAD_GRAYSCALE:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


@lru_cache(maxsize=128)
def _template(path, modified, size, scale):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is not None and scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return image


def read_template(path, scale=1.0):
    return _template(*_key(path), scale)
