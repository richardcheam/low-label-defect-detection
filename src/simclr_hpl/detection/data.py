from __future__ import annotations

from pathlib import Path

import supervision as sv

from simclr_hpl.utils import ensure_dir

COCO_ANNOTATIONS_FILENAME = "_annotations.coco.json"

DEFAULT_SPLITS = ("train", "valid", "test")


def convert_yolo_split_to_coco(
    images_dir: str | Path,
    labels_dir: str | Path,
    data_yaml: str | Path,
    output_dir: str | Path,
) -> Path:
    """Convert a single YOLO-format split into a COCO dataset directory.

    Loads ``images_dir``/``labels_dir`` (YOLO ``class cx cy w h`` labels) using
    ``data_yaml`` for class names, then writes the images and a
    ``_annotations.coco.json`` file into ``output_dir``. Returns the path to the
    written annotations file.
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    data_yaml_path = Path(data_yaml)
    if not images_path.exists():
        msg = f"YOLO images directory does not exist: {images_path}"
        raise FileNotFoundError(msg)
    if not labels_path.exists():
        msg = f"YOLO labels directory does not exist: {labels_path}"
        raise FileNotFoundError(msg)
    if not data_yaml_path.exists():
        msg = f"data.yaml file does not exist: {data_yaml_path}"
        raise FileNotFoundError(msg)

    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=str(images_path),
        annotations_directory_path=str(labels_path),
        data_yaml_path=str(data_yaml_path),
    )

    output_path = ensure_dir(output_dir)
    annotations_path = output_path / COCO_ANNOTATIONS_FILENAME
    dataset.as_coco(
        images_directory_path=str(output_path),
        annotations_path=str(annotations_path),
    )
    return annotations_path


def convert_roboflow_yolo_dataset(
    yolo_root: str | Path,
    output_root: str | Path,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> dict[str, Path]:
    """Convert a Roboflow YOLOv11 export into per-split COCO datasets.

    For each split in ``splits`` that exists under ``yolo_root`` (with
    ``images/`` and ``labels/`` subdirectories), writes
    ``<output_root>/<split>/_annotations.coco.json`` (plus images). Returns a
    mapping from split name to the written annotations file path.
    """
    yolo_root_path = Path(yolo_root)
    data_yaml = yolo_root_path / "data.yaml"
    if not data_yaml.exists():
        msg = f"data.yaml file does not exist: {data_yaml}"
        raise FileNotFoundError(msg)

    coco_paths: dict[str, Path] = {}
    for split in splits:
        split_root = yolo_root_path / split
        images_dir = split_root / "images"
        labels_dir = split_root / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        coco_paths[split] = convert_yolo_split_to_coco(
            images_dir=images_dir,
            labels_dir=labels_dir,
            data_yaml=data_yaml,
            output_dir=Path(output_root) / split,
        )
    return coco_paths
