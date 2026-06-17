import torch
from torch.utils.data import ConcatDataset, TensorDataset

from simclr_hpl.data import collect_labels
from simclr_hpl.models import (
    Encoder,
    EncoderClassifier,
    LinearProbe,
    MLPProbe,
    ProjectionHead,
    ResNet18Encoder,
    SemiSupervisedCNN,
    build_encoder,
)
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
    assert (
        infer_metrics_type(
            {"n_labeled_images": 116, "baseline": {}, "iterations": [], "label_fraction": 0.01}
        )
        == "pseudo_box"
    )


def test_resnet18_encoder_output_shape():
    encoder = ResNet18Encoder(input_channels=3, pretrained=False)
    assert encoder.output_dim == 512
    out = encoder(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 512)


def test_build_encoder_selects_variant():
    assert isinstance(build_encoder("small", input_channels=1), Encoder)
    assert isinstance(build_encoder("resnet18", input_channels=3), ResNet18Encoder)


def test_probes_support_binary_num_classes():
    encoder = ResNet18Encoder(input_channels=3)
    linear = LinearProbe(encoder, num_classes=2)
    mlp = MLPProbe(encoder, num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    assert linear(x).shape == (2, 2)
    assert mlp(x).shape == (2, 2)


def test_probe_keeps_encoder_in_eval_and_freezes_batchnorm():
    from simclr_hpl.models import LinearProbe, ResNet18Encoder
    from simclr_hpl.training import freeze_module

    encoder = ResNet18Encoder(input_channels=3)
    freeze_module(encoder)
    probe = LinearProbe(encoder, num_classes=2)
    probe.train()  # mimic train_classifier's model.train()

    assert encoder.training is False  # encoder stays in eval despite probe.train()

    # running stats must not change across a training-mode forward+backward
    bn = encoder.backbone.bn1
    before_mean = bn.running_mean.clone()
    optimizer = torch.optim.Adam(
        [p for p in probe.parameters() if p.requires_grad], lr=1e-3
    )
    x = torch.randn(4, 3, 32, 32)
    target = torch.tensor([0, 1, 0, 1])
    loss = torch.nn.functional.cross_entropy(probe(x), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.equal(bn.running_mean, before_mean)  # BN running stats frozen


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
