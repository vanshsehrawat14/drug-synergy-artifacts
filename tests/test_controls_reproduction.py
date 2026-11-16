import numpy as np
import pytest

from synergy_artifacts.controls import bliss_null, bliss_reference, hsa_reference
from synergy_artifacts.features import FeatureBuilder, Standardizer
from test_features_models import data


def test_references_and_explicit_null():
    np.testing.assert_allclose(bliss_reference([20, -10], [30, 40]), [44, 34])
    np.testing.assert_allclose(hsa_reference([20, -10], [30, 40]), [30, 40])
    np.testing.assert_array_equal(bliss_null([20, 50], [30, 60]), bliss_reference([20, 50], [30, 60]))
    np.testing.assert_array_equal(bliss_null([20], [30], noise_sd=2), bliss_null([20], [30], noise_sd=2))


@pytest.mark.parametrize("normalization", ["standard", "tanh", "tanh_standard"])
def test_normalization_and_recovery(normalization):
    frame = data()
    features = FeatureBuilder(normalization=normalization).fit(frame)
    restored = FeatureBuilder.from_state_dict(features.state_dict())
    for x, y in zip(features.transform(frame), restored.transform(frame)):
        np.testing.assert_array_equal(x, y)
    if normalization == "tanh":
        x = np.concatenate([features._continuous(frame, "a"), features._continuous(frame, "b")])
        expected = np.tanh((x-x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1))
        np.testing.assert_allclose(features._normalize(x, "endpoint"), expected, atol=1e-7)


def test_drop_constant_and_f2_coverage():
    assert Standardizer().fit([[1, 2], [1, 4]], remove_constant=True).transform([[1, 3]]).shape == (1, 1)
    frame = data()
    mono = frame.copy().assign(dose_b=0., inhibition=[10., 20., 30.])
    builder = FeatureBuilder("F2").fit(frame, mono)
    assert builder.monotherapy_coverage(frame)["a"]["available"] == 3
    assert builder.monotherapy_coverage(frame)["b"]["available"] == 1
    assert np.isfinite(builder.transform(frame)[0]).all()
    restored = FeatureBuilder.from_state_dict(builder.state_dict())
    np.testing.assert_array_equal(builder.monotherapy_values(frame, "a"), restored.monotherapy_values(frame, "a"))


def test_streaming_normalizer_matches_dense():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(101, 8))
    x[:, 2] = 5
    dense = Standardizer().fit(x, True)
    stream = Standardizer().fit_batches([x[:10], x[10:51], x[51:]], True)
    np.testing.assert_array_equal(dense.keep, stream.keep)
    np.testing.assert_allclose(dense.transform(x), stream.transform(x), atol=1e-7)
