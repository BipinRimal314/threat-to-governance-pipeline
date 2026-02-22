"""Load CMU-CERT dataset via the MSc thesis data pipeline.

Bridges the insider-detection project's feature engineering output
into the threat-to-governance pipeline. Uses importlib to load
modules from the sibling project by file path, avoiding namespace
collisions (both projects have ``src.data``).

Typical directory layout:
    Thesis/
    ├── insider-detection/       <- MSc thesis code + data
    │   ├── data/r4.2/          <- Raw CMU-CERT CSVs
    │   └── src/                <- Feature engineering pipeline
    └── threat-to-governance-pipeline/  <- This project

Usage:
    from src.data.cert_loader import load_cert_features
    X, entity_ids, timestamps, labels = load_cert_features()
"""

import importlib.util
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _get_insider_detection_path() -> Path:
    """Locate the insider-detection project directory."""
    candidates = [
        Path(__file__).parents[3] / "insider-detection",
        Path.cwd().parent / "insider-detection",
        Path.home() / "Downloads" / "Thesis" / "insider-detection",
    ]
    for path in candidates:
        if path.exists() and (path / "src").exists():
            return path
    raise FileNotFoundError(
        "Cannot find insider-detection project. "
        "Expected at ../insider-detection/ relative to this project."
    )


def _load_insider_modules(id_path: Path):
    """Load insider-detection data modules under a synthetic package.

    Registers modules as ``_id_data.loader``, ``_id_data.preprocessing``,
    ``_id_data.features`` so that relative imports between them
    (e.g. ``from .preprocessing import ...``) resolve correctly.
    """
    import sys
    import types

    pkg_name = "_id_data"
    id_src = id_path / "src" / "data"

    # Create synthetic parent package
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(id_src)]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg

    module_names = ["loader", "preprocessing", "features"]
    loaded = {}
    for mod_name in module_names:
        fqn = f"{pkg_name}.{mod_name}"
        file_path = id_src / f"{mod_name}.py"
        spec = importlib.util.spec_from_file_location(
            fqn, str(file_path),
            submodule_search_locations=[],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg_name
        sys.modules[fqn] = mod
        setattr(pkg, mod_name, mod)
        spec.loader.exec_module(mod)
        loaded[mod_name] = mod

    return loaded


def load_cert_features(
    dataset_version: str = "r4.2",
    min_user_days: int = 30,
    insider_detection_path: Optional[str] = None,
) -> Tuple[np.ndarray, list, list, np.ndarray]:
    """Load CMU-CERT data using the MSc thesis pipeline.

    Loads the raw CSVs, runs preprocessing and feature engineering
    from the insider-detection project, then extracts UBFS vectors
    using CERTFeatureExtractor.

    Args:
        dataset_version: CMU-CERT version (e.g., "r4.2").
        min_user_days: Minimum days per user to include.
        insider_detection_path: Override path to insider-detection.

    Returns:
        Tuple of:
            X: UBFS feature matrix (n_samples, 20)
            entity_ids: List of user IDs
            timestamps: List of date strings
            labels: Binary ground truth labels
    """
    if insider_detection_path:
        id_path = Path(insider_detection_path)
    else:
        id_path = _get_insider_detection_path()

    # Load insider-detection modules under a synthetic package
    mods = _load_insider_modules(id_path)
    loader_mod = mods["loader"]
    preproc_mod = mods["preprocessing"]
    features_mod = mods["features"]

    print(f"  Loading CMU-CERT {dataset_version} from {id_path}...")
    data = loader_mod.load_raw_data(id_path / "data", dataset_version)
    preprocessed = preproc_mod.preprocess_all(data, min_user_days=min_user_days)
    features_df = features_mod.compute_daily_features(preprocessed)
    labeled_df = features_mod.add_labels(
        features_df, preprocessed["ground_truth"]
    )

    # Extract UBFS vectors using this project's own extractor
    from src.features.cert_extractor import CERTFeatureExtractor
    extractor = CERTFeatureExtractor()
    X, entity_ids, timestamps = extractor.extract_batch(labeled_df)

    # Extract labels
    if "is_insider" in labeled_df.columns:
        labels = labeled_df["is_insider"].to_numpy().astype(np.int32)
    elif "label" in labeled_df.columns:
        labels = labeled_df["label"].to_numpy().astype(np.int32)
    else:
        labels = np.zeros(len(X), dtype=np.int32)

    n_pos = labels.sum()
    print(f"  CMU-CERT: {len(X)} user-days, "
          f"{n_pos} insider ({100*n_pos/max(len(X),1):.2f}%)")

    return X, entity_ids, timestamps, labels
