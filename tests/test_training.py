import numpy as np
import pytest

from cs622.training import NeuralRegressor


def problem():
    rng = np.random.default_rng(3)
    a, b, cell = [rng.normal(size=(24, n)).astype(np.float32) for n in [3, 3, 2]]
    y = (a[:, 0] + b[:, 0] + cell[:, 0]) * 3
    return (a, b, cell), y


def test_overfit_checkpoint_and_exact_resume(tmp_path):
    x, y = problem()
    spec = dict(kind="dose", endpoint_dim=3, cell_dim=2, hidden=[16], encoder_dim=8, dropout=.1)
    config = dict(epochs=100, batch_size=24, lr=.02, patience=200, threads=1)
    full = NeuralRegressor(spec, config, 19).fit(x, y, (x, y), tmp_path / "full", {"data": "synthetic"})
    assert np.mean((full.predict(x) - y) ** 2) < .7
    restored, context = NeuralRegressor.load(tmp_path / "full" / "best.pt", str(full.device))
    np.testing.assert_array_equal(full.predict(x), restored.predict(x))
    assert context == {"data": "synthetic"}
    short = NeuralRegressor(spec, {**config, "epochs": 40}, 19)
    short.fit(x, y, (x, y), tmp_path / "short", context)
    resumed = NeuralRegressor(spec, config, 19).fit(x, y, (x, y), tmp_path / "short", context,
        resume=tmp_path / "short" / "latest.pt")
    np.testing.assert_array_equal(full.predict(x), resumed.predict(x))
    with pytest.raises(ValueError, match="differ"):
        NeuralRegressor(spec, config, 19).fit(x, y, (x, y), tmp_path / "bad", {"data": "changed"},
            resume=tmp_path / "short" / "latest.pt")


def test_invalid_target(tmp_path):
    x, y = problem()
    y[0] = np.nan
    model = NeuralRegressor(dict(kind="dose", endpoint_dim=3, cell_dim=2, hidden=[8]), dict(epochs=1))
    with pytest.raises(ValueError, match="finite"):
        model.fit(x, y, (x, y), tmp_path, {})
