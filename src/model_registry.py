
import datetime as dt
import json
import shutil

from .config import MODEL_PATH, MODELS_DIR, REGISTRY_PATH


def load_registry():
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())

    return {"current": None, "versions": []}


def current_model_info():
    registry = load_registry()

    return next(
        (
            version for version in registry["versions"]
            if version["version"] == registry["current"]
        ),
        None
    )


def register_model(
    model, history, metrics, kind,
    n_train, n_new, force_promote=False
):
    registry = load_registry()
    version = len(registry["versions"]) + 1
    previous = current_model_info()

    promoted = bool(
        force_promote
        or previous is None
        or metrics["accuracy"] >=
        previous["metrics"]["accuracy"] - 0.02
    )

    versions_dir = MODELS_DIR / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    version_file = (
        versions_dir / f"urbansound_cnn_v{version}.h5"
    )

    model.save(version_file)

    if promoted:
        shutil.copy2(version_file, MODEL_PATH)
        registry["current"] = version

    entry = {
        "version": version,
        "kind": kind,
        "promoted": promoted,
        "file": version_file.name,
        "trained_at": (
            dt.datetime.now(dt.timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        ),
        "n_train": int(n_train),
        "n_new_files": int(n_new),
        "metrics": metrics,
        "history": history
    }

    registry["versions"].append(entry)

    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2)
    )

    return entry

_register = register_model
