"""Train a local-feature classifier from sparse, human-labeled point seeds."""
from pathlib import Path
import csv
import json

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent
SEEDS = ROOT / "seed_annotations.csv"
SOURCE = ROOT / "fábrica.ply"
INPUT = ROOT / "results" / "ply_by_class.ply"
OUTPUT = ROOT / "results" / "ply_seed_model.ply"
STATS = ROOT / "results" / "seed_model_stats.json"

CLASSES = ("piso", "tuberia", "viga", "persona", "maquina_vehiculo")
RGB_BY_CLASS = {
    "piso": (230, 180, 34), "tuberia": (22, 166, 161), "viga": (214, 90, 49),
    "techo": (40, 120, 181), "columna": (101, 163, 67), "persona": (224, 82, 156),
    "maquina_vehiculo": (121, 80, 181), "otro": (119, 119, 119),
}
NAME_BY_RGB = {v: k for k, v in RGB_BY_CLASS.items()}
TARGET_RGB = np.array([RGB_BY_CLASS[c] for c in CLASSES], dtype=np.uint8)
CANDIDATE_CLASSES = (*CLASSES, "otro")
CANDIDATE_RGB = np.array([RGB_BY_CLASS[c] for c in CANDIDATE_CLASSES], dtype=np.uint8)
SEED_EXPANSION_RADIUS = 0.30
SEED_NORMAL_AGREEMENT = 0.82


def local_features(points, normals, colors, k=40, batch=20000):
    """Translation- and scale-independent local shape, normal, and color descriptors."""
    tree = cKDTree(points)
    result = np.empty((len(points), 22), dtype=np.float32)
    for lo in range(0, len(points), batch):
        hi = min(lo + batch, len(points))
        d, ix = tree.query(points[lo:hi], k=k, workers=-1)
        q = points[ix]
        delta = q - q.mean(axis=1, keepdims=True)
        cov = np.einsum("bki,bkj->bij", delta, delta) / k
        eig = np.linalg.eigvalsh(cov)[:, ::-1]
        eig = np.maximum(eig, 1e-12)
        eig /= eig.sum(axis=1, keepdims=True)
        an = np.abs(normals[ix])
        nc = colors[ix]
        result[lo:hi] = np.column_stack((
            np.abs(normals[lo:hi]), colors[lo:hi], eig,
            an.mean(axis=1), an.std(axis=1),
            nc.mean(axis=1), nc.std(axis=1), d[:, -1],
        ))
    return result


def read_seeds(points, tree):
    rows = list(csv.DictReader(SEEDS.open(encoding="utf-8")))
    coords = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in rows])
    labels = np.array([r["class"].strip().lower() for r in rows])
    if not set(labels).issubset(CLASSES):
        raise ValueError(f"Las etiquetas permitidas son {CLASSES}.")
    dist, index = tree.query(coords, k=1, workers=-1)
    if np.max(dist) > 0.08:
        raise ValueError(f"Una anotación no coincide con el PLY (distancia máxima {np.max(dist):.3f} m).")
    return rows, coords, labels, index, dist


def main():
    source = o3d.io.read_point_cloud(str(SOURCE))
    original = np.asarray(source.points)
    original_normals = np.asarray(source.normals)
    original_colors = np.asarray(source.colors)
    cloud = o3d.io.read_point_cloud(str(INPUT))
    points = np.asarray(cloud.points)
    base_rgb = np.rint(np.asarray(cloud.colors) * 255).astype(np.uint8)
    _, source_ix = cKDTree(original).query(points, k=1, workers=-1)
    normals = original_normals[source_ix]
    colors = original_colors[source_ix]

    features = local_features(points, normals, colors)
    tree = cKDTree(points)
    rows, seed_xyz, seed_y, seed_ix, seed_dist = read_seeds(points, tree)
    seed_x = features[seed_ix]

    # Spatially grouped validation prevents nearby clicks from landing in both folds.
    groups = DBSCAN(eps=0.60, min_samples=1).fit_predict(seed_xyz)
    n_splits = min(5, len(np.unique(groups)))
    model = ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=1, max_features=0.8,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    if n_splits >= 2:
        cv = GroupKFold(n_splits=n_splits)
        cv_pred = cross_val_predict(model, seed_x, seed_y, groups=groups, cv=cv, n_jobs=1)
    else:
        cv_pred = np.full_like(seed_y, "sin_validacion")
    # Expand each click only along its nearby surface, using normal agreement to avoid
    # crossing from a pipe to an adjacent beam at a junction.
    expanded_labels = {}
    for seed_id, (seed_index, seed_class) in enumerate(zip(seed_ix, seed_y)):
        neighborhood = tree.query_ball_point(points[seed_index], r=SEED_EXPANSION_RADIUS)
        neighborhood = np.asarray(neighborhood, dtype=int)
        same_surface = np.abs(normals[neighborhood] @ normals[seed_index]) >= SEED_NORMAL_AGREEMENT
        for point_id in neighborhood[same_surface]:
            old = expanded_labels.get(int(point_id))
            dist = float(np.linalg.norm(points[point_id] - points[seed_index]))
            if old is None or dist < old[1]:
                expanded_labels[int(point_id)] = (seed_class, dist)
    train_ix = np.fromiter(expanded_labels.keys(), dtype=int)
    train_y = np.array([expanded_labels[int(i)][0] for i in train_ix])
    model.fit(features[train_ix], train_y)

    candidates = np.any(np.all(base_rgb[:, None, :] == CANDIDATE_RGB[None, :, :], axis=2), axis=1)
    probabilities = model.predict_proba(features[candidates])
    best = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(best)), best]
    accepted = confidence >= 0.80
    candidate_indices = np.flatnonzero(candidates)
    out_rgb = base_rgb.copy()
    class_to_rgb = {name: np.array(RGB_BY_CLASS[name], dtype=np.uint8) for name in CLASSES}
    predicted_rgb = np.stack([class_to_rgb[model.classes_[j]] for j in best[accepted]])
    accepted_indices = candidate_indices[accepted]
    out_rgb[accepted_indices] = predicted_rgb
    changed = accepted_indices[np.any(predicted_rgb != base_rgb[accepted_indices], axis=1)]
    propagated = train_ix[~np.all(out_rgb[train_ix] == np.stack([
        class_to_rgb[c] for c in train_y
    ]), axis=1)]
    out_rgb[train_ix] = np.stack([class_to_rgb[c] for c in train_y])
    # Preserve the manually confirmed seed points as the labels that supervised training.
    out_rgb[seed_ix] = np.stack([class_to_rgb[c] for c in seed_y])
    changed_count = int(np.sum(np.any(out_rgb != base_rgb, axis=1)))
    out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    out.colors = o3d.utility.Vector3dVector(out_rgb.astype(float) / 255)
    o3d.io.write_point_cloud(str(OUTPUT), out, write_ascii=False)

    cv_report = classification_report(seed_y, cv_pred, labels=CLASSES, output_dict=True, zero_division=0)
    counts = {name: int(np.sum(np.all(out_rgb == rgb, axis=1))) for name, rgb in RGB_BY_CLASS.items()}
    stats = {
        "seed_count": len(rows), "seed_counts": {c: int(np.sum(seed_y == c)) for c in CLASSES},
        "expanded_training_points": int(len(train_ix)),
        "seed_expansion_radius_m": SEED_EXPANSION_RADIUS,
        "seed_normal_agreement": SEED_NORMAL_AGREEMENT,
        "spatial_groups": int(len(np.unique(groups))),
        "spatial_grouped_cv_accuracy": float(accuracy_score(seed_y, cv_pred)) if n_splits >= 2 else None,
        "spatial_grouped_cv_report": cv_report,
        "spatial_grouped_cv_confusion_matrix": confusion_matrix(seed_y, cv_pred, labels=CLASSES).tolist(),
        "cv_class_order": list(CLASSES),
        "max_seed_snap_distance_m": float(seed_dist.max()),
        "candidate_points": int(candidates.sum()), "accepted_predictions": int(len(accepted_indices)),
        "candidate_classes": list(CANDIDATE_CLASSES),
        "reclassified_points": changed_count,
        "locally_propagated_points": int(len(propagated)),
        "confidence_threshold": 0.80, "output_points_per_class": counts,
        "feature_names": ["abs_normal_xyz", "rgb", "local_covariance_eigenvalue_ratios",
                          "neighbor_abs_normal_mean", "neighbor_abs_normal_std",
                          "neighbor_rgb_mean", "neighbor_rgb_std", "knn_radius"],
        "note": "Sparse human seed labels; grouped CV is only a small-sample diagnostic, not full-scene ground truth.",
        "output": str(OUTPUT),
    }
    STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("Seed labels:", stats["seed_counts"])
    print("Spatially grouped seed CV accuracy:", round(stats["spatial_grouped_cv_accuracy"], 3))
    print("Candidate points:", stats["candidate_points"], "reclassified:", stats["reclassified_points"])
    print("Output:", OUTPUT)
    print("Stats:", STATS)


if __name__ == "__main__":
    main()
