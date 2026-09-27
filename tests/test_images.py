import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from autococ.images import read_frame


class ImageCacheTests(unittest.TestCase):
    def test_atomic_replacement_with_same_mtime_and_size_reads_new_frame(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "frame.png"
            first = np.full((4, 4, 3), 10, dtype=np.uint8)
            second = np.full((4, 4, 3), 80, dtype=np.uint8)
            options = [cv2.IMWRITE_PNG_COMPRESSION, 0]
            path.write_bytes(cv2.imencode(".png", first, options)[1].tobytes())
            old_stat = path.stat()
            np.testing.assert_array_equal(read_frame(path), first)

            replacement = Path(temp) / "replacement.png"
            replacement.write_bytes(cv2.imencode(".png", second, options)[1].tobytes())
            self.assertEqual(replacement.stat().st_size, old_stat.st_size)
            os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            replacement.replace(path)

            np.testing.assert_array_equal(read_frame(path), second)


if __name__ == "__main__":
    unittest.main()
