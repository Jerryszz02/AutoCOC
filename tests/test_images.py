import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from autococ.images import _frame, read_frame


class ImageCacheTests(unittest.TestCase):
    def test_three_frame_working_set_reuses_decodes_and_cache_is_bounded(self) -> None:
        with TemporaryDirectory() as temp:
            paths = []
            for index in range(5):
                path = Path(temp) / f"frame-{index}.png"
                pixels = np.full((4, 4, 3), index * 20, dtype=np.uint8)
                path.write_bytes(cv2.imencode(".png", pixels)[1].tobytes())
                paths.append(path)
            _frame.cache_clear()
            try:
                decode = cv2.imdecode
                with patch("autococ.images.cv2.imdecode", wraps=decode) as decoded:
                    for index in (0, 1, 2, 0, 1, 2, 0):
                        np.testing.assert_array_equal(
                            read_frame(paths[index]),
                            np.full((4, 4, 3), index * 20, dtype=np.uint8))
                    self.assertEqual(decoded.call_count, 3)
                    self.assertEqual(_frame.cache_info().maxsize, 4)
                    for index in (3, 4):
                        read_frame(paths[index])
                    self.assertEqual(_frame.cache_info().currsize, 4)
                    read_frame(paths[1])
                    self.assertEqual(decoded.call_count, 6)
            finally:
                _frame.cache_clear()

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
