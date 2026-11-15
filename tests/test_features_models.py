import numpy as np
import pandas as pd
import pytest
import torch

from cs622.features import FeatureBuilder, Standardizer
from cs622.models import DeepSynergy, DoseNetwork


def data():
    return pd.DataFrame(dict(drug_a=["A", "B", "A"], drug_b=["B", "C", "C"], cell_line=["X"] * 3,
        dose_a=[.1, 1., 10.], dose_b=[1., 10., 100.], unit_a=["uM"]*3, unit_b=["uM"]*3,
        pair_id=["AB", "BC", "AC"]))


def test_train_only_normalization():
    scaler = Standardizer().fit([[0], [2]])
    assert scaler.transform([[100]])[0, 0] == 99
    frame = data()
    builder = FeatureBuilder().fit(frame)
    before = builder.endpoint_scaler.mean.copy()
    builder.transform(frame.assign(dose_a=1e100))
    np.testing.assert_array_equal(before, builder.endpoint_scaler.mean)
    restored = FeatureBuilder.from_state_dict(builder.state_dict())
    for a, b in zip(builder.transform(frame), restored.transform(frame)):
        np.testing.assert_array_equal(a, b)


def test_monotherapy_leakage_rejected():
    train = data()
    with pytest.raises(ValueError, match="leakage"):
        FeatureBuilder("F2").fit(train, train.iloc[[0]].assign(pair_id="HELDOUT"))
    with pytest.raises(ValueError, match="F1"):
        FeatureBuilder("F1")
    with pytest.raises(ValueError, match="Unknown drug"):
        FeatureBuilder().fit(train).transform(train.assign(drug_a="UNKNOWN"))


def test_categorical_unobserved_labels():
    frame = data()
    frame["cell_line"] = pd.Categorical(frame.cell_line, categories=["X", "UNSEEN"])
    frame["drug_a"] = pd.Categorical(frame.drug_a, categories=["A", "B", "C", "UNSEEN"])
    frame["drug_b"] = pd.Categorical(frame.drug_b, categories=["A", "B", "C", "UNSEEN"])
    a, b, cell = FeatureBuilder().fit(frame).transform(frame)
    assert cell.shape == (3, 1)
    assert np.isfinite(a).all() and np.isfinite(b).all()


@pytest.mark.parametrize("model_class", [DoseNetwork, DeepSynergy])
def test_network_symmetry_and_backward(model_class):
    torch.manual_seed(9)
    model = model_class(5, 3, hidden=[16, 8], dropout=0)
    a, b, c = torch.randn(9, 5), torch.randn(9, 5), torch.randn(9, 3)
    model.eval()
    torch.testing.assert_close(model(a, b, c), model(b, a, c), rtol=0, atol=0)
    loss = model(a, b, c).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_dose_stays_with_drug():
    frame = data()
    builder = FeatureBuilder().fit(frame)
    swapped = frame.rename(columns={"drug_a": "drug_b", "drug_b": "drug_a", "dose_a": "dose_b", "dose_b": "dose_a", "unit_a": "unit_b", "unit_b": "unit_a"})
    a, b, c = builder.transform(frame)
    sb, sa, sc = builder.transform(swapped)
    np.testing.assert_array_equal(a, sa)
    np.testing.assert_array_equal(b, sb)
    np.testing.assert_array_equal(c, sc)
    with pytest.raises(ValueError, match="finite"):
        Standardizer().fit([[np.inf]])


def test_symmetric_model_retains_drug_dose_association():
    model = DoseNetwork(3, 1, hidden=(), encoder_dim=2, dropout=0.)
    with torch.no_grad():
        model.encoder[0].weight.copy_(torch.tensor([[1., 0., 1.], [0., 1., 1.]]))
        model.encoder[0].bias.zero_()
        model.head[0].weight.zero_()
        model.head[0].weight[0, 2] = 1.
        model.head[0].bias.zero_()
    model.eval()
    cell = torch.zeros(1, 1)
    a, b = torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 1., 2.]])
    reversed_doses_a, reversed_doses_b = torch.tensor([[1., 0., 2.]]), torch.tensor([[0., 1., 0.]])
    assert model(a, b, cell).item() == 1.
    assert model(b, a, cell).item() == 1.
    assert model(reversed_doses_a, reversed_doses_b, cell).item() == 3.
