# 1. DEPENDENCIAS E IMPORTACIONES =============================================

from pathlib import Path
import os, json, time, warnings
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
from scipy.ndimage import binary_closing, label as ndi_label, find_objects, maximum_filter, convolve
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# 2. CONFIGURACIÓN DEL PIPELINE ===============================================

_HERE = Path(__file__).resolve().parent
CONFIG = {
    "PLY_PATH": os.environ.get("PLY_PATH", str(_HERE / "fábrica.ply")),
    "OUTPUT_DIR": os.environ.get("OUT_DIR", str(_HERE / "results")),
    "MAX_POINTS": 823_510,
    "SEED": 42, "VOXEL": 0.05, "USE_FILE_NORMALS": True, "USE_RGB": True,
    "UP_AXIS": 2, "HIST_BIN": 0.05, "NORMAL_UP": 0.90,
    "FLOOR_BAND": 0.15, "CEILING_BAND": 0.30,
    "PLANE_DIST": 0.03, "PLANE_ITERS": 1000, "MAX_PLANES": 15,
    "MIN_PLANE_POINTS": 1500, "PLANE_PIECE_EPS": 0.15,
    "KNN_K": 20, "KNN_SAMPLE": 300, "ETA": 2.0, "DBSCAN_MIN": 5,
    "GIANT_FRACTION": 0.50, "GIANT_ETA_FACTOR": 0.70,
    "CYLINDER_MAX_POINTS": 5000, "CYLINDER_TRIALS": 120,
    "CYLINDER_RADIUS_MIN": 0.03, "CYLINDER_RADIUS_MAX": 0.60,
    "CYLINDER_INLIER_RATIO": 0.50, "CYLINDER_MAX_ANGULAR_GAP": 155.0, "TINY_POINTS": 25,

    "WALL_PLANARITY": 0.72, "WALL_VERTICALITY": 0.75,
    "ELONGATED_LINEARITY": 0.78, "COLUMN_VERTICALITY": 0.80,
    "PERSON_HEIGHT_MIN": 1.40, "PERSON_HEIGHT_MAX": 2.10,
    "PERSON_FOOTPRINT_MAX": 1.20, "PERSON_BASE_MAX": 0.35,
    "MACHINE_FOOTPRINT_MIN": 1.50, "MACHINE_HEIGHT_MAX": 3.50,
    "MACHINE_BASE_MAX": 0.50,
    # Comparación visual; ROI manual opcional en coordenadas XYZ (metros).
    "COMPARE_MAX_POINTS": 180_000, "COMPARE_PERSON_ROI": None,
    "COMPARE_CROP_MARGIN": 0.60,
    "FLOOR_PLANE_MIN_SIDE": 1.50, "FLOOR_PLANE_MIN_AREA": 5.0,
    "FLOOR_PLANE_GRID": 0.30, "FLOOR_PLANE_MIN_OCCUPANCY": 0.12,
    "FLOOR_PLANE_QUADRANT_SUPPORT": 1,
    "FLOOR_PLANE_MAX_Z_SPREAD": 0.35,
    "FLOOR_PIPE_REASSIGN_RADIUS": 0.15, "FLOOR_PIPE_REASSIGN_SHARE": 0.50,
    "FLOOR_LEVEL_GRID": 0.30, "FLOOR_LEVEL_MIN_OCCUPANCY": 0.20,
    "FLOOR_CEILING_CLEARANCE": 1.20, "FILL_VERTICAL_TOL": 0.10,
    "PIPE_CHECK_LINEARITY": 0.60,

    "RESIDUAL_METHOD": "dbscan", "ALT_CLUSTER_COUNT": 12,

    "USE_RG": True, "RG_ANGLE": 12.0, "RG_RADIUS_FACTOR": 2.2, "RG_RADIUS_MIN": 0.08, "RG_MIN": 40,
    "REFINE_MIN_POINTS": 300, "RG_MIN_COVER": 0.30,

    "USE_COLOR": True, "MS_QUANTILE": 0.12, "MS_VOXEL": 0.10, "MS_FIT_MAX": 30_000,
    "COLOR_CUT": 0.35, "COLOR_LIGHT_WEIGHT": 0.5,

    "FILL_UNASSIGNED": True, "FILL_RADIUS": 0.25,

    # [COLUMNAS] detección por continuidad vertical (ocupación XY a lo largo de la altura)
    "USE_COLUMN_STAGE": True,
    "COL_CELL": 0.10, "COL_ZBIN": 0.25,          # celda XY (m) y tamaño de bin vertical (m)
    "COL_Z_LOW": 0.30, "COL_Z_HIGH": 0.50,       # se ignora la franja pegada al piso y al techo
    "COL_MIN_OCC": 0.45,                         # fracción de la altura útil que debe estar ocupada en esa celda XY
    "COL_CLOSE": 2,                              # celdas para cerrar huecos (une las alas de una columna en H)
    "COL_MIN_THICK": 0.15, "COL_MIN_SIDE": 0.35, "COL_MAX_SIDE": 1.80,   # huella permitida (m): descarta tubos finos y paredes
    "COL_MIN_FILL": 0.20, "COL_PAD": 1,          # relleno mínimo de la huella y margen (celdas) para incluir alas
}

CONFIG_OVERRIDE = {"ETA": 1.5, "DBSCAN_MIN": 3}
CONFIG.update(CONFIG_OVERRIDE)
CONFIG.update(json.loads(os.environ.get("CFG_JSON", "{}")))

SEED = CONFIG["SEED"]
np.random.seed(SEED)
CLASS_NAMES = {0:"piso", 1:"techo", 3:"columna", 4:"tuberia",
               5:"viga", 6:"persona", 7:"maquina_vehiculo", 8:"otro"}
# Paleta categórica de alto contraste: colores saturados separados por tono y brillo.
COLORS = np.array([[0xE6/255,0xB4/255,0x22/255],  # piso
                   [0x28/255,0x78/255,0xB5/255],  # techo
                   [.90,.05,.05],                 # clase 2 sin uso
                   [0x65/255,0xA3/255,0x43/255],  # columna
                   [0x16/255,0xA6/255,0xA1/255],  # tuberia
                   [0xD6/255,0x5A/255,0x31/255],  # viga
                   [0xE0/255,0x52/255,0x9C/255],  # persona
                   [0x79/255,0x50/255,0xB5/255],  # maquina/vehiculo
                   [0x77/255,0x77/255,0x77/255]]) # otro
TIMINGS = []

# 3. UTILIDADES: CARGA Y VOXELIZACIÓN =========================================

def timed(name, count, fn):
    t = time.perf_counter(); out = fn(); dt = time.perf_counter()-t
    TIMINGS.append({"etapa":name,"segundos":dt,"puntos":int(count),"puntos_segundo":float(count/dt) if dt else 0})
    return out

def load_cloud(path):
    """Carga un PLY XYZ y obtiene opcionalmente sus atributos RGB/normales."""
    cloud = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(cloud.points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("No se pudieron leer puntos XYZ del archivo PLY.")
    rgb = np.asarray(cloud.colors) if cloud.has_colors() else None
    normals = np.asarray(cloud.normals) if cloud.has_normals() else None
    return xyz, rgb, normals

def voxelize(xyz, voxel):
    """Submuestrea y conserva los índices originales asociados a cada vóxel."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    bounds = pc.get_axis_aligned_bounding_box()
    down, _, trace = pc.voxel_down_sample_and_trace(voxel, bounds.min_bound, bounds.max_bound, False)
    centers = np.asarray(down.points)
    original = [np.asarray(ids, dtype=np.int64) for ids in trace]
    return centers, original

def class_colors(labels):
    color=np.tile(np.array([.2,.2,.2]),(len(labels),1)); ok=labels>=0
    color[ok]=COLORS[labels[ok]]
    return color

# 4. ATRIBUTOS POR VÓXEL: NORMALES Y COLOR ====================================

def estimate_normals(xyz, provided, config):
    """Usa las normales válidas del archivo y estima desde XYZ las ausentes o inválidas."""
    n = np.asarray(provided, dtype=float) if (config["USE_FILE_NORMALS"] and provided is not None and len(provided)==len(xyz)) else None
    if n is not None:
        good = np.isfinite(n).all(axis=1) & (np.linalg.norm(n,axis=1)>1e-8)
        if good.all(): return n / np.linalg.norm(n,axis=1,keepdims=True)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(3*config["VOXEL"],.12), max_nn=30))
    est = np.asarray(pc.normals)
    if n is None: return est / np.maximum(np.linalg.norm(est,axis=1,keepdims=True),1e-12)
    out = est.copy(); out[good] = n[good] / np.linalg.norm(n[good],axis=1,keepdims=True)
    return out

def check_up_axis(xyz, normals, config):
    axis = config["UP_AXIS"]
    med = np.nanmedian(np.abs(normals),axis=0)
    if axis != 2 or med[2] < 0.35:
        warnings.warn(f"El eje UP_AXIS={axis} puede no ser vertical (mediana |normal|={med.round(2)}). Revisa el eje y rota XYZ si hace falta.")

def voxel_attributes(n_vox, mapping, rgb, normals):
    """[NORMAS][COLOR] Promedia RGB y normales del archivo por vóxel (las normales se alinean al signo de la primera)."""
    lens = np.array([len(m) for m in mapping]); flat = np.concatenate(mapping); vid = np.repeat(np.arange(n_vox), lens)
    cnt = np.maximum(lens, 1).astype(float); vrgb = vnor = None
    if rgb is not None:
        vrgb = np.stack([np.bincount(vid, rgb[flat, j], n_vox) / cnt for j in range(3)], 1)
    if normals is not None:
        first = np.r_[0, np.cumsum(lens)[:-1]]; ref = normals[flat[first]]
        sgn = np.sign(np.einsum("ij,ij->i", normals[flat], ref[vid])); sgn[sgn == 0] = 1
        vnor = np.stack([np.bincount(vid, normals[flat, j] * sgn, n_vox) / cnt for j in range(3)], 1)
        vnor /= np.maximum(np.linalg.norm(vnor, axis=1, keepdims=True), 1e-12)
    return vrgb, vnor

# 5. MÓDULO DE COLOR: MEAN-SHIFT ==============================================

def color_module(xyz_vox, vrgb, config, rng):
    """[COLOR] Mean-Shift (compañero 3): se ajusta con una muestra submuestreada por vóxel y se predice para todos."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_vox)); pc.colors = o3d.utility.Vector3dVector(vrgb)
    sample = np.asarray(pc.voxel_down_sample(config["MS_VOXEL"]).colors)
    if len(sample) > config["MS_FIT_MAX"]: sample = sample[rng.choice(len(sample), config["MS_FIT_MAX"], replace=False)]
    bw = estimate_bandwidth(sample, quantile=config["MS_QUANTILE"], n_samples=2000, random_state=config["SEED"])
    ms = MeanShift(bandwidth=max(bw, 1e-3), bin_seeding=True, cluster_all=True, n_jobs=-1).fit(sample)
    C = ms.cluster_centers_; g = C.mean(1); chroma = C - g[:, None]

    cdist = np.sqrt(((chroma[:, None] - chroma[None]) ** 2).sum(-1) + (config["COLOR_LIGHT_WEIGHT"] * (g[:, None] - g[None])) ** 2)
    return ms.predict(vrgb), C, cdist

