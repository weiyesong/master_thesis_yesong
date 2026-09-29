import tempfile
import unittest
from pathlib import Path

from scripts.create_eurosat_splits import CLASS_NAMES, assign_group_stratified, validate


class EuroSATSplitTests(unittest.TestCase):
    def make_records(self, root: Path):
        records = []
        for class_index, label in enumerate(CLASS_NAMES):
            for item in range(10):
                index = class_index * 10 + item
                path = Path(label) / f"{label}_{item}.tif"
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).touch()
                records.append(
                    {
                        "sample_id": path.stem,
                        "dataset_index": index,
                        "file_path": path.as_posix(),
                        "class_label": label,
                        "class_index": class_index,
                        "official_split": "unknown",
                        "sha256": f"hash-{index}",
                        "spatial_group": f"geo_{index:05d}",
                    }
                )
        return records

    def test_assignment_is_fixed_stratified_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.make_records(root)
            second = [dict(record) for record in first]
            assign_group_stratified(first, 17)
            assign_group_stratified(second, 17)
            self.assertEqual([record["split"] for record in first], [record["split"] for record in second])
            report = validate(first, root)
            self.assertTrue(report["valid"])
            self.assertEqual(report["split_counts"], {"train": 70, "val": 10, "calibration": 10, "test": 10})
            for split, expected in (("train", 7), ("val", 1), ("calibration", 1), ("test", 1)):
                self.assertTrue(all(value == expected for value in report["class_distribution"][split].values()))

    def test_duplicate_sample_id_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self.make_records(root)
            assign_group_stratified(records, 17)
            records[1]["sample_id"] = records[0]["sample_id"]
            self.assertFalse(validate(records, root)["valid"])

    def test_spatial_group_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self.make_records(root)
            assign_group_stratified(records, 17)
            records[0]["spatial_group"] = records[-1]["spatial_group"]
            if records[0]["split"] == records[-1]["split"]:
                records[-1]["split"] = "test" if records[0]["split"] != "test" else "val"
            self.assertFalse(validate(records, root)["valid"])


if __name__ == "__main__":
    unittest.main()
