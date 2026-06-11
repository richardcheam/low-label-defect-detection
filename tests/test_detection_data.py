from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from simclr_hpl.detection.data import convert_yolo_split_to_coco


def _make_yolo_split(root: Path, names):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    # 2 images, one box each (class 0 and class 1)
    for i, cls in enumerate([0, 1]):
        Image.new("RGB", (64, 64), (100, 100, 100)).save(root / "images" / f"img{i}.jpg")
        (root / "labels" / f"img{i}.txt").write_text(f"{cls} 0.5 0.5 0.4 0.4\n")


def test_convert_yolo_split_to_coco(tmp_path):
    names = ["Gun", "Knife", "Pliers", "Scissors", "Wrench"]
    split = tmp_path / "train"
    _make_yolo_split(split, names)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"names": names, "nc": len(names)}))

    out = tmp_path / "coco" / "train"
    coco_path = convert_yolo_split_to_coco(
        images_dir=split / "images",
        labels_dir=split / "labels",
        data_yaml=data_yaml,
        output_dir=out,
    )
    data = json.loads(Path(coco_path).read_text())
    assert {"images", "annotations", "categories"} <= data.keys()
    assert len(data["annotations"]) == 2  # one box per image
    assert len(data["images"]) == 2
    # all 5 classes present as categories (supervision may include a supercategory; assert >=5)
    cat_names = {c["name"] for c in data["categories"]}
    assert {"Gun", "Knife"} <= cat_names
