import unittest

import numpy as np

from eggcv.core import images, make_split, metrics


class CoreTests(unittest.TestCase):
    def test_metric_counts(self):
        truth = np.array([[1, 1], [0, 0]], np.uint8)
        pred = np.array([[1, 0], [1, 0]], np.uint8)
        result = metrics(truth, pred)
        self.assertAlmostEqual(result["iou"], 1/3)
        self.assertAlmostEqual(result["dice"], .5)
        self.assertAlmostEqual(result["pixel_accuracy"], .5)
        self.assertAlmostEqual(result["balanced_accuracy"], .5)

    def test_scene_groups_do_not_cross_split(self):
        split = make_split(images())
        dev, evaluation = set(split["dev"]), set(split["eval"])
        self.assertEqual(len(dev), 88)
        self.assertEqual(len(evaluation), 22)
        self.assertFalse(dev & evaluation)
        for group in split["groups"]:
            self.assertTrue(set(group) <= dev or set(group) <= evaluation)
        for first, second in [("151", "152"), ("113", "135")]:
            self.assertTrue(any(
                any(f"not_damaged_{first}_jpg" in path for path in group)
                and any(f"not_damaged_{second}_jpg" in path for path in group)
                for group in split["groups"]
            ))


if __name__ == "__main__":
    unittest.main()
