"""Train-only transforms with shared endpoint scaling and explicit information regimes."""
import numpy as np
import pandas as pd

from .io import digest, read_json
from .splits import normalize_name


class Standardizer:
    def fit(self, x, remove_constant=False):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected finite nonempty feature matrix")
        scale = x.std(axis=0)
        self.keep = scale != 0 if remove_constant else np.ones(x.shape[1], dtype=bool)
        self.mean = x[:, self.keep].mean(axis=0)
        self.scale = scale[self.keep]
        self.scale[self.scale == 0] = 1
        return self

    def fit_batches(self, batches, remove_constant=False):
        count, mean, m2 = 0, None, None
        for x in batches:
            x = np.asarray(x, dtype=float)
            if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
                raise ValueError("Expected finite nonempty feature batch")
            n = len(x)
            batch_mean = x.mean(0)
            batch_m2 = ((x - batch_mean) ** 2).sum(0)
            if count == 0:
                mean, m2 = batch_mean, batch_m2
            else:
                delta = batch_mean - mean
                m2 += batch_m2 + delta ** 2 * count * n / (count + n)
                mean += delta * n / (count + n)
            count += n
        if count == 0:
            raise ValueError("No training features")
        scale = np.sqrt(m2 / count)
        self.keep = scale != 0 if remove_constant else np.ones(len(mean), dtype=bool)
        self.mean, self.scale = mean[self.keep], scale[self.keep]
        self.scale[self.scale == 0] = 1
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite features")
        return ((x[:, self.keep] - self.mean) / self.scale).astype(np.float32)


def load_independent_features(path, provenance):
    meta = read_json(provenance)
    if meta.get("independent_of_combination_outcomes") is not True or not meta.get("source"):
        raise ValueError("Feature provenance must declare an independent source")
    if meta.get("sha256") != digest(path):
        raise ValueError("Feature checksum differs from provenance")
    with np.load(path, allow_pickle=False) as z:
        result = {}
        for entity in ["drug", "cell"]:
            ids = [normalize_name(x) for x in z[entity + "_ids"].tolist()]
            values = z[entity + "_features"].astype(float)
            if len(ids) != len(set(ids)) or values.ndim != 2 or len(ids) != len(values):
                raise ValueError(f"Invalid {entity} feature identifiers or dimensions")
            if not np.isfinite(values).all():
                raise ValueError("External features contain nonfinite values")
            result[entity] = dict(zip(ids, values.tolist()))
        result["provenance"] = meta
        return result


class FeatureBuilder:
    def __init__(self, regime="F0", dose=True, identities=True, external=None, normalization="standard", remove_constant=False,
                 mono_missing="mean", dose_input=True, reference_inputs=(), drug_identity=True, cell_identity=True):
        if regime not in {"F0", "F1", "F2"}:
            raise ValueError("Unknown information regime")
        if regime == "F1" and external is None:
            raise ValueError("F1 requires independent drug and cell features")
        self.regime, self.dose, self.identities, self.external = regime, dose, identities, external
        self.drug_identity, self.cell_identity = drug_identity, cell_identity
        if normalization not in {"standard", "tanh", "tanh_standard"}:
            raise ValueError("Unknown normalization")
        self.normalization, self.remove_constant = normalization, remove_constant
        if mono_missing not in {"mean", "interpolate"} or not set(reference_inputs) <= {"bliss", "hsa"}:
            raise ValueError("Unknown monotherapy missing-value policy or reference input")
        if (mono_missing != "mean" or reference_inputs) and regime != "F2":
            raise ValueError("Monotherapy options require F2")
        # "interpolate": a dose absent from the training lookup takes the log-dose interpolation of
        # the same drug/cell training-pair values (constant beyond the observed range).
        self.mono_missing, self.dose_input, self.reference_inputs = mono_missing, dose_input, list(reference_inputs)
        self.mono = {}

    def _continuous(self, frame, endpoint):
        parts = []
        if self.dose:
            if not (set(frame["unit_" + endpoint]) <= {"uM"}):
                raise ValueError("Features require doses in uM")
            doses = frame["dose_" + endpoint].to_numpy(float)
            if not np.isfinite(doses).all() or (doses < 0).any():
                raise ValueError("Invalid dose")
            # Zero has an explicit presence indicator, never log(0) or an arbitrary pseudodose.
            if self.dose_input:
                parts += [np.where(doses > 0, np.log10(np.maximum(doses, np.finfo(float).tiny)), 0)[:, None],
                          (doses > 0).astype(float)[:, None]]
        if self.external is not None:
            try:
                parts.append(np.array([self.external["drug"][d] for d in frame["drug_" + endpoint]]))
            except KeyError as e:
                raise ValueError(f"Missing independent drug feature: {e}") from e
        if self.regime == "F2":
            values = self.monotherapy_values(frame, endpoint)
            missing = ~np.isfinite(values)
            parts += [np.where(missing, self.mono_mean, values)[:, None], missing.astype(float)[:, None]]
        return np.concatenate(parts, axis=1) if parts else np.zeros((len(frame), 0))

    def _curves(self):
        if getattr(self, "_curve_cache", None) is None:
            points = {}
            for (drug, cell, dose), value in self.mono.items():
                points.setdefault((drug, cell), []).append((np.log10(dose), value))
            self._curve_cache = {key: tuple(np.array(z) for z in zip(*sorted(v))) for key, v in points.items()}
        return self._curve_cache

    def monotherapy_lookup(self, frame, endpoint):
        """Training-pair monotherapy and status per row: 0 exact key, 1 interpolated, 2 unavailable."""
        values = np.array([self.mono.get((d, c, float(x)), np.nan) for d, c, x in
                           zip(frame["drug_" + endpoint], frame.cell_line, frame["dose_" + endpoint])])
        status = np.where(np.isfinite(values), 0, 2)
        if self.mono_missing == "interpolate" and (status == 2).any():
            curves = self._curves()
            rows = np.flatnonzero(status == 2)
            sub = frame.iloc[rows]
            for i, d, c, x in zip(rows, sub["drug_" + endpoint], sub.cell_line, sub["dose_" + endpoint]):
                curve = curves.get((d, c))
                if curve is not None and x > 0:
                    values[i], status[i] = np.interp(np.log10(x), *curve), 1
        return values, status

    def monotherapy_values(self, frame, endpoint):
        return self.monotherapy_lookup(frame, endpoint)[0]

    def monotherapy_status(self, frame):
        return np.maximum(self.monotherapy_lookup(frame, "a")[1], self.monotherapy_lookup(frame, "b")[1])

    def monotherapy_coverage(self, frame):
        result = {}
        for endpoint in ["a", "b"]:
            status = self.monotherapy_lookup(frame, endpoint)[1]
            result[endpoint] = {"available": int((status == 0).sum()), "interpolated": int((status == 1).sum()),
                                "unavailable": int((status == 2).sum()), "rows": len(frame)}
        return result

    def reference(self, frame, kind):
        """Noninteraction reference from training-pair monotherapy only, never held-out block edges."""
        from .controls import bliss_reference, hsa_reference
        if kind not in {"bliss", "hsa"}:
            raise ValueError("Unknown noninteraction reference")
        a, b = self.monotherapy_values(frame, "a"), self.monotherapy_values(frame, "b")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError("Reference requires training-pair monotherapy for every row")
        return (bliss_reference if kind == "bliss" else hsa_reference)(a, b)

    def reference_matrix(self, frame):
        return np.column_stack([self.reference(frame, kind) for kind in self.reference_inputs])

    def fit(self, train, monotherapy=None):
        self.drugs = sorted(set(train.drug_a) | set(train.drug_b))
        self.cells = sorted(set(train.cell_line))
        if self.regime == "F2":
            if not self.dose or monotherapy is None:
                raise ValueError("F2 requires dose rows and training-block monotherapy")
            if set(monotherapy.pair_id) - set(train.pair_id):
                raise ValueError("Held-out-pair monotherapy leakage")
            rows = []
            for endpoint in ["a", "b"]:
                other = "b" if endpoint == "a" else "a"
                sub = monotherapy[(monotherapy["dose_" + endpoint] > 0) & (monotherapy["dose_" + other] == 0)]
                rows.extend(zip(sub["drug_" + endpoint], sub.cell_line, sub["dose_" + endpoint], sub.inhibition))
            mono = pd.DataFrame(rows, columns=["drug", "cell", "dose", "response"])
            if mono.empty or not np.isfinite(mono.response).all():
                raise ValueError("No finite training-block monotherapy")
            grouped = mono.groupby(["drug", "cell", "dose"], sort=True).response.mean()
            self.mono = {key: float(value) for key, value in grouped.items()}
            self.mono_mean = float(grouped.mean())
            self._curve_cache = None
            if self.reference_inputs:
                self.reference_scaler = Standardizer().fit(self.reference_matrix(train))
        def endpoint_batches():
            for start in range(0, len(train), 2048):
                batch = train.iloc[start:start+2048]
                for endpoint in ["a", "b"]:
                    yield self._continuous(batch, endpoint)
        def cell_batches():
            for start in range(0, len(train), 2048):
                yield np.array([self.external["cell"][c] for c in train.cell_line.iloc[start:start+2048]])
        self.endpoint_scaler = Standardizer().fit_batches(endpoint_batches(), self.remove_constant)
        if self.external is not None:
            self.cell_scaler = Standardizer().fit_batches(cell_batches(), self.remove_constant)
        if self.normalization == "tanh_standard":
            self.endpoint_second = Standardizer().fit_batches(np.tanh(self.endpoint_scaler.transform(x)) for x in endpoint_batches())
            if self.external is not None:
                self.cell_second = Standardizer().fit_batches(np.tanh(self.cell_scaler.transform(x)) for x in cell_batches())
        return self

    def _normalize(self, x, entity):
        result = getattr(self, entity + "_scaler").transform(x)
        if self.normalization.startswith("tanh"):
            result = np.tanh(result)
        if self.normalization == "tanh_standard":
            result = getattr(self, entity + "_second").transform(result)
        return result

    def transform(self, frame):
        if (set(frame.drug_a) | set(frame.drug_b)) - set(self.drugs):
            raise ValueError("Unknown drug in identity vocabulary")
        if set(frame.cell_line) - set(self.cells):
            raise ValueError("Unknown cell line in training vocabulary")
        di, ci = {d: i for i, d in enumerate(self.drugs)}, {c: i for i, c in enumerate(self.cells)}
        endpoints = []
        for endpoint in ["a", "b"]:
            continuous = self._normalize(self._continuous(frame, endpoint), "endpoint")
            identity = np.eye(len(di), dtype=np.float32)[frame["drug_" + endpoint].map(di).to_numpy(dtype=np.int64)]
            endpoints.append(np.concatenate([identity, continuous], axis=1) if self.identities and self.drug_identity else continuous)
        cell = np.eye(len(ci), dtype=np.float32)[frame.cell_line.map(ci).to_numpy(dtype=np.int64)]
        if not self.identities or not self.cell_identity:
            cell = np.zeros((len(frame), 0), dtype=np.float32)
        if self.external is not None:
            try:
                extra = self._normalize(np.array([self.external["cell"][c] for c in frame.cell_line]), "cell")
            except KeyError as e:
                raise ValueError(f"Missing independent cell feature: {e}") from e
            cell = np.concatenate([cell, extra], axis=1)
        if self.reference_inputs:
            # Pair-level references join the cell context in the head, after symmetric pooling.
            cell = np.concatenate([cell, self.reference_scaler.transform(self.reference_matrix(frame))], axis=1)
        if endpoints[0].shape[1] == 0 or (cell.shape[1] == 0 and self.regime != "F2"):
            raise ValueError("Ablation removed every drug or cell feature")
        return (*endpoints, cell)

    def state_dict(self):
        scalers = ["endpoint_scaler", "cell_scaler", "endpoint_second", "cell_second", "reference_scaler"]
        state = {k: v for k, v in self.__dict__.items() if k not in {*scalers, "mono"} and not k.startswith("_")}
        state["mono"] = [[*key, value] for key, value in self.mono.items()]
        for name in scalers:
            if hasattr(self, name):
                scaler = getattr(self, name)
                state[name] = {"mean": scaler.mean.tolist(), "scale": scaler.scale.tolist(), "keep": scaler.keep.tolist()}
        return state

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state["regime"], state["dose"], state["identities"], state["external"], state.get("normalization", "standard"), state.get("remove_constant", False))
        for key, value in state.items():
            if key.endswith(("scaler", "second")):
                scaler = Standardizer()
                scaler.mean, scaler.scale = np.array(value["mean"]), np.array(value["scale"])
                scaler.keep = np.array(value.get("keep", [True] * len(scaler.mean)), dtype=bool)
                setattr(obj, key, scaler)
            elif key == "mono":
                obj.mono = {(d, c, x): y for d, c, x, y in value}
            else:
                setattr(obj, key, value)
        return obj


class LazyFeatures:
    """Expand independent descriptors per minibatch, never once per dose row."""
    def __init__(self, builder, frame):
        self.builder, self.frame = builder, frame

    def __len__(self):
        return len(self.frame)

    def batch(self, indices):
        return self.builder.transform(self.frame.iloc[indices])
