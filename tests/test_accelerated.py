"""Small generated fixtures only; no real dataset or output arrays are committed."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from accelerated import geometry
from accelerated.inputs import inspect_inputs, read_index
from accelerated.lossless_distance import DistanceReader, write_distance


class LosslessFormatTests(unittest.TestCase):
    def test_bitwise_roundtrip_and_cross_chunk_window(self):
        # Includes -0, subnormal, infinities and a noncanonical NaN payload.
        bits = np.array([0, 0x80000000, 1, 0x3f800001, 0x7f800000, 0xff800000, 0x7fc12345, 0x442f1234], np.uint32)
        values = np.resize(bits, (19, 23)).view(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'map.dmap'
            write_distance(path, values, chunk_size=7)
            reader = DistanceReader(path)
            self.assertEqual(reader.read().tobytes(), values.tobytes())
            self.assertEqual(reader.read_window(4, 17, 5, 22).tobytes(), values[4:17, 5:22].tobytes())

    def test_does_not_silently_quantize_float64(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(TypeError):
                write_distance(Path(folder) / 'map.dmap', np.ones((3, 3), np.float64))


class ExternalInputTests(unittest.TestCase):
    def test_index_selection_geometry_coordinates_and_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            polygons = root / 'polygon/buildings_complete'
            antennas = root / 'antenna'
            polygons.mkdir(parents=True)
            antennas.mkdir()
            # Generated geometry, never a copied production city.
            path = polygons / '42.json'
            path.write_text(json.dumps([[[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]]))
            (antennas / '42.json').write_text(json.dumps([[5, 5], [25, 25], [15, 15]]))
            index_path = root / 'selection.csv'
            index_path.write_text('scene,tx\n42,2\n42,0\n')
            index = read_index(index_path)
            self.assertEqual(index, {42: [2, 0]})
            result = inspect_inputs(root, index)
            self.assertEqual(result[0]['wall_count'], 4)
            self.assertEqual(len(result[0]['polygon_sha256']), 64)
            p = geometry.load_polygons(path)
            self.assertEqual(p[0][0].tolist(), [10., 245.])
            self.assertTrue(geometry.occupied(np.array([15., 240.]), p))

    def test_duplicate_selection_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'selection.csv'
            path.write_text('scene,tx\n1,2\n1,2\n')
            with self.assertRaises(ValueError):
                read_index(path)

    def test_fractional_source_not_silently_rounded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'polygon/buildings_complete').mkdir(parents=True)
            (root / 'antenna').mkdir()
            (root / 'polygon/buildings_complete/1.json').write_text(json.dumps([[[10,10],[20,10],[20,20],[10,20]]]))
            (root / 'antenna/1.json').write_text(json.dumps([[5.5, 6]]))
            with self.assertRaisesRegex(ValueError, 'integer XY'):
                inspect_inputs(root, {1: [0]})


if __name__ == '__main__':
    unittest.main()
