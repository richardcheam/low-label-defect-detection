import torch
from torch.utils.data import ConcatDataset, TensorDataset

from simclr_hpl.data import collect_labels
from simclr_hpl.models import Encoder, EncoderClassifier, ProjectionHead, SemiSupervisedCNN


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
