import os
import tempfile
import unittest

import nibabel as nib
import numpy as np

from code_gaussian.src.data_loaders.mitea_loader import MITEADataset


class DatasetTests(unittest.TestCase):
    def test_nifti_arrays_are_returned_as_depth_height_width(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "images"))
            os.makedirs(os.path.join(root, "labels"))
            # Nibabel stores x,y,z. The loader must expose z,y,x.
            data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
            affine = np.eye(4)
            for folder, suffix in (("images", ""), ("labels", "")):
                for phase in ("ED", "ES"):
                    name = f"MITEA_001_scan1_{phase}.nii.gz"
                    value = data if folder == "images" else np.zeros_like(data)
                    if folder == "labels":
                        value[3, 0, 1] = 1
                    nib.save(nib.Nifti1Image(value, affine), os.path.join(root, folder, name))

            sample = MITEADataset(root, split="train")[0]
            self.assertEqual(tuple(sample["occupancy_ed"].shape), (6, 5, 4))
            self.assertEqual(sample["occupancy_ed"][0, 0, 0].item(), 0.0)
            self.assertEqual(sample["occupancy_ed"][0, 0, 3].item(), 0.0)
            self.assertEqual(sample["occupancy_ed"][1, 0, 3].item(), 1.0)
