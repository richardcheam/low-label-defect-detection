from pathlib import Path

from PIL import Image

from simclr_hpl.data import load_binary_image_records, subsample_per_class


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(120, 120, 120)).save(path)


def test_load_binary_image_records_assigns_labels(tmp_path):
    _make_image(tmp_path / "positive" / "a.png")
    _make_image(tmp_path / "positive" / "b.png")
    _make_image(tmp_path / "negative" / "c.png")

    records = load_binary_image_records(
        tmp_path,
        class_to_label={"positive": 1, "negative": 0},
    )

    labels = sorted(int(r["label"]) for r in records)
    assert labels == [0, 1, 1]
    assert all(Path(str(r["path"])).exists() for r in records)


def test_load_binary_image_records_errors_when_empty(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_binary_image_records(tmp_path, class_to_label={"positive": 1})


def test_subsample_per_class_caps_each_label():
    records = [{"path": f"{i}.png", "label": i % 2} for i in range(10)]  # 5 per label
    capped = subsample_per_class(records, max_per_class=2, seed=0)
    labels = sorted(int(r["label"]) for r in capped)
    assert labels == [0, 0, 1, 1]


def test_subsample_per_class_none_keeps_all():
    records = [{"path": f"{i}.png", "label": i % 2} for i in range(6)]
    assert len(subsample_per_class(records, None, seed=0)) == 6
