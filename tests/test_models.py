import torch
from torch.utils.data import ConcatDataset, TensorDataset

from simclr_hpl.data import collect_labels
from simclr_hpl.models import Encoder, EncoderClassifier, ProjectionHead, SemiSupervisedCNN
from simclr_hpl.business import compute_review_queue_metrics
from simclr_hpl.visualization import infer_metrics_type


def test_encoder_and_projection_shapes():
    inputs = torch.randn(4, 1, 28, 28)
    encoder = Encoder()
    projection = ProjectionHead()
    features = encoder(inputs)
    outputs = projection(features)
    assert features.shape == (4, 4096)
    assert outputs.shape == (4, 64)


def test_semisupervised_cnn_output_shape():
    model = SemiSupervisedCNN()
    inputs = torch.randn(4, 1, 28, 28)
    logits = model(inputs)
    assert logits.shape == (4, 10)


def test_encoder_classifier_output_shape():
    model = EncoderClassifier()
    inputs = torch.randn(4, 1, 28, 28)
    logits = model(inputs)
    assert logits.shape == (4, 10)


def test_collect_labels_from_concat_dataset():
    first = TensorDataset(torch.randn(2, 1), torch.tensor([1, 2]))
    second = TensorDataset(torch.randn(2, 1), torch.tensor([3, 4]))
    combined = ConcatDataset([first, second])
    assert collect_labels(combined) == [1, 2, 3, 4]


def test_infer_metrics_type_for_supported_payloads():
    assert infer_metrics_type({"simclr": {}, "linear_probe": {}, "mlp_probe": {}}) == "simclr"
    assert (
        infer_metrics_type({"baseline": {}, "single_round_pseudo_labeling": {}})
        == "pseudo_label"
    )
    assert infer_metrics_type({"benchmark_results": {}, "summary": []}) == "transfer"
    assert infer_metrics_type({"dataset": "mvtec_ad", "results": {}, "summary": []}) == "mvtec"


def test_review_queue_metrics_are_computed():
    metrics = compute_review_queue_metrics(
        predictions=[0, 1, 1, 0],
        targets=[0, 1, 0, 1],
        confidences=[0.99, 0.97, 0.60, 0.95],
        auto_decision_threshold=0.95,
        defect_label=1,
    )
    assert metrics["auto_decision_rate"] == 0.75
    assert metrics["review_queue_rate"] == 0.25
