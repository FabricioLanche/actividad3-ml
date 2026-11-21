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

# 6. REGION GROWING POR NORMALES Y COLOR ======================================

def region_growing_normales(points, normals, max_angle_deg=12.0, radius=0.08, min_cluster_size=40, cid=None, cdist=None, color_cut=None):
    """[NORMAS] Region growing por continuidad angular (compañero 2): arista si arccos(|n1.n2|)<=ángulo.
    [COLOR] Si se da cid/cdist, la arista además exige que la distancia de color entre sus clases sea <= color_cut."""
    n = len(points)
    if n == 0: return np.array([], dtype=int)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    pairs = cKDTree(points).query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0: return np.full(n, -1, dtype=int)
    ok = np.abs(np.sum(normals[pairs[:, 0]] * normals[pairs[:, 1]], axis=1)) >= np.cos(np.radians(max_angle_deg))
    if cid is not None and cdist is not None and color_cut is not None:
        ok &= cdist[cid[pairs[:, 0]], cid[pairs[:, 1]]] <= color_cut
    valid = pairs[ok]
    rows = np.concatenate([valid[:, 0], valid[:, 1]]); cols = np.concatenate([valid[:, 1], valid[:, 0]])
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    n_comp, raw = connected_components(adj, directed=False)
    counts = np.bincount(raw); good = np.where(counts >= min_cluster_size)[0]
    remap = np.full(n_comp, -1, dtype=int); remap[good] = np.arange(len(good))
    return remap[raw]

def split_with_normals_and_color(points, normals, config, cid, cdist):
    """[NORMAS][COLOR] Region growing sobre un cluster. Devuelve etiquetas de segmento (-1 = remanente) o None si las
    normales son demasiado ruidosas (cobertura < RG_MIN_COVER). Los fragmentos < RG_MIN se adhieren al segmento más cercano."""
    sub = region_growing_normales(points, normals, config["RG_ANGLE"], config["_RG_RADIUS"], config["RG_MIN"],
                                  cid, cdist, config["COLOR_CUT"] if cid is not None else None)
    good = sub >= 0
    if not good.any() or good.mean() < config["RG_MIN_COVER"]: return None
    if (~good).any():
        d_, nn_ = cKDTree(points[good]).query(points[~good], distance_upper_bound=2 * config["_RG_RADIUS"], workers=-1)
        hit = np.isfinite(d_); loc = np.flatnonzero(~good); sub = sub.copy()
        sub[loc[hit]] = sub[np.flatnonzero(good)[nn_[hit]]]
    return sub

def resolve_unclassified(ix, vox, vn, cid, cdist, floor_z, cfg, rng, vlabels, instance, next_inst):
    """[NORMAS][COLOR] Segunda oportunidad: un cluster que las reglas dejaron como "otro" se subdivide con region growing
    (normales + color) y cada segmento se reclasifica con las MISMAS reglas. Solo puede convertir "otro" en otra clase;
    lo ya clasificado por el compañero 1 no se toca."""
    sub = split_with_normals_and_color(vox[ix], vn[ix], cfg, None if cid is None else cid[ix], cdist)
    if sub is None: return next_inst
    for sl in np.unique(sub[sub >= 0]):
        jx = ix[sub == sl]
        if len(jx) < cfg["TINY_POINTS"]: continue
        cl = classify_features(cluster_features(vox[jx], vn[jx], floor_z, cfg, rng), cfg)
        if cl == 8: continue
        vlabels[jx] = cl; instance[jx] = next_inst; next_inst += 1
    return next_inst

def refine_pipe_segments(ix, vox, vn, cid, cdist, floor_z, cfg, rng, vlabels, instance, next_inst):
    """Busca subsegmentos cilíndricos dentro de grupos inicialmente clasificados como viga/otro."""
    sub = split_with_normals_and_color(vox[ix], vn[ix], cfg, None if cid is None else cid[ix], cdist)
    if sub is None: return next_inst
    for sl in np.unique(sub[sub >= 0]):
        jx = ix[sub == sl]
        if len(jx) < cfg["TINY_POINTS"]: continue
        f = cluster_features(vox[jx], vn[jx], floor_z, cfg, rng)
        if f["cilindro"]["ok"] and f["verticalidad"] < cfg["COLUMN_VERTICALITY"]:
            vlabels[jx] = 4; instance[jx] = next_inst; next_inst += 1
    return next_inst

# 7. DETECCIÓN DE PISO Y TECHO ================================================

def detect_floor_ceiling(xyz, normals, config):
    """Detecta piso(s) por picos de altura y cobertura XY; el techo es el pico superior."""
    axis=config["UP_AXIS"]; z=xyz[:,axis]; nz=np.abs(normals[:,axis])
    cand=nz>=config["NORMAL_UP"]
    if cand.sum()<max(100,int(.005*len(xyz))): cand=np.ones(len(xyz),bool)
    bw=config["HIST_BIN"]
    edges=np.arange(z.min()-2*bw,z.max()+3*bw,bw)
    hist,_=np.histogram(z[cand],bins=edges)
    smooth=np.convolve(hist,np.ones(3),mode="same")
    peaks,_=find_peaks(smooth, distance=max(1,int(.30/bw)))
    if len(peaks)<2: peaks=np.argsort(smooth)[-min(2,len(smooth)):]
    peaks=sorted(peaks,key=lambda i:smooth[i],reverse=True)
    centers=(edges[:-1]+edges[1:])/2

    chosen=sorted(peaks[:min(6,len(peaks))],key=lambda i:centers[i])
    ceil_z=float(centers[chosen[-1]])
    horizontal=nz>=config["NORMAL_UP"]
    ax=[i for i in range(3) if i!=axis]; floor_levels=[]
    for pi in peaks:
        level=float(centers[pi])
        if level>=ceil_z-config["CEILING_BAND"]: continue
        near=(np.abs(z-level)<=config["FLOOR_BAND"]) & horizontal
        if near.sum()<max(100,int(.003*len(xyz))): continue
        xy=xyz[near][:,ax]; cell=config["FLOOR_LEVEL_GRID"]
        ij=np.floor((xy-xy.min(0))/cell).astype(int); shape=ij.max(0)+1
        occupied=np.zeros(shape,dtype=bool); occupied[ij[:,0],ij[:,1]]=True
        if occupied.mean()>=config["FLOOR_LEVEL_MIN_OCCUPANCY"]:
            floor_levels.append(level)
    if not floor_levels:
        floor_levels=[float(centers[chosen[0]])]
    floor_z=float(min(floor_levels))
    # Solo el nivel más bajo se asigna por altura global. Los pisos elevados
    # deben pasar RANSAC y las pruebas de área/cobertura para no absorber tuberías.
    floor=(np.abs(z-floor_z)<=config["FLOOR_BAND"]) & horizontal
    ceiling=(np.abs(z-ceil_z)<=config["CEILING_BAND"]) & horizontal & ~floor
    if floor_z>=ceil_z: warnings.warn("No se detectó piso por debajo del techo; revisar ejes y bandas.")
    return floor,ceiling,floor_z,ceil_z,np.asarray(floor_levels,float)

# 7b. DETECCIÓN DE COLUMNAS POR CONTINUIDAD VERTICAL ===========================

def detect_columns(vox, excluded, floor_z, ceil_z, config):
    """[COLUMNAS] Una columna es una huella XY pequeña ocupada a lo largo de (casi) toda la altura del recinto.
    Las vigas y tuberías horizontales ocupan pocos niveles z, las paredes tienen huellas largas y los tubos
    verticales finos tienen huellas menores a COL_MIN_SIDE; la regla 'elongado + vertical' del clasificador
    falla cuando la columna está pegada a vigas/tuberías porque el cluster deja de ser lineal.
    Devuelve el id de columna por vóxel (-1 = ninguno)."""
    up = config["UP_AXIS"]; ax = [i for i in range(3) if i != up]
    cell, zb = config["COL_CELL"], config["COL_ZBIN"]
    z = vox[:, up]; lo = floor_z + config["COL_Z_LOW"]; hi = ceil_z - config["COL_Z_HIGH"]
    out = np.full(len(vox), -1, int)
    use = (~excluded) & (z > lo) & (z < hi)
    if use.sum() < 100 or hi <= lo: return out, 0
    o = vox[:, ax].min(0)
    ij = np.floor((vox[:, ax] - o) / cell).astype(int) + 2
    sh = ij.max(0) + 3
    zk = np.clip(np.floor((z - floor_z) / zb).astype(int), 0, None)
    occ = np.zeros((sh[0], sh[1], zk.max() + 1), bool)
    occ[ij[use, 0], ij[use, 1], zk[use]] = True
    cover = occ.sum(2) / ((hi - lo) / zb)
    mask = binary_closing(cover >= config["COL_MIN_OCC"], iterations=config["COL_CLOSE"])
    lab, n = ndi_label(mask, structure=np.ones((3, 3)))
    cellmap = np.full(lab.shape, -1, int); kept = 0
    for c, sl in enumerate(find_objects(lab), start=1):
        sub = lab[sl] == c
        ex, ey = (sl[0].stop - sl[0].start) * cell, (sl[1].stop - sl[1].start) * cell
        if min(ex, ey) < config["COL_MIN_THICK"] or not (config["COL_MIN_SIDE"] <= max(ex, ey) <= config["COL_MAX_SIDE"]): continue
        if sub.mean() < config["COL_MIN_FILL"]: continue
        cellmap[sl][sub] = kept; kept += 1
    if kept == 0: return out, 0
    if config["COL_PAD"] > 0: cellmap = maximum_filter(cellmap, size=2 * config["COL_PAD"] + 1)
    cid = cellmap[ij[:, 0], ij[:, 1]]
    out[(~excluded) & (cid >= 0)] = cid[(~excluded) & (cid >= 0)]
    return out, kept

# 8. EXTRACCIÓN DE PLANOS CON RANSAC ==========================================

def dbscan_labels(points, eps, min_samples):
    if len(points)==0: return np.empty(0,dtype=int)
    return DBSCAN(eps=float(eps),min_samples=int(min_samples),n_jobs=-1).fit_predict(points)

def extract_planes(xyz, normals, excluded, config):
    """Extrae planos RANSAC iterativos y separa cada plano en piezas conectadas."""
    remaining=np.flatnonzero(~excluded); plane_ids=np.full(len(xyz),-1,int); next_id=0
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    for _ in range(config["MAX_PLANES"]):
        if len(remaining)<config["MIN_PLANE_POINTS"]: break
        sub=cloud.select_by_index(remaining.tolist())
        model,inliers=sub.segment_plane(config["PLANE_DIST"],3,config["PLANE_ITERS"],.999)
        if len(inliers)<config["MIN_PLANE_POINTS"]: break
        ids=remaining[np.asarray(inliers,dtype=int)]
        local=dbscan_labels(xyz[ids],config["PLANE_PIECE_EPS"],max(5,config["DBSCAN_MIN"]))
        for lab in np.unique(local):
            piece=ids[local==lab]
            if len(piece)>=config["TINY_POINTS"]:
                plane_ids[piece]=next_id; next_id+=1
        mask=np.ones(len(remaining),bool); mask[np.asarray(inliers,dtype=int)]=False
        remaining=remaining[mask]
    return plane_ids

# 9. CLUSTERIZACIÓN DEL RESIDUO ===============================================

def adaptive_eps(points, config, rng):
    if len(points)<3: return config["ETA"]*config["VOXEL"]
    take=rng.choice(len(points),min(config["KNN_SAMPLE"],len(points)),replace=False)
    k=min(config["KNN_K"]+1,len(points))
    d,_=cKDTree(points).query(points[take],k=k,workers=-1)
    kth=d[:,-1] if d.ndim==2 else d
    med=float(np.median(kth[np.isfinite(kth)]))
    return max(config["VOXEL"],config["ETA"]*med)

def cluster_residual(points, normals, config, rng):
    """Agrupa residuo; los bloques gigantes se subdividen por orientación y escala."""
    method=config["RESIDUAL_METHOD"].lower()
    eps=adaptive_eps(points,config,rng)
    if method=="dbscan": labels=dbscan_labels(points,eps,config["DBSCAN_MIN"])
    elif method=="kmeans": labels=KMeans(n_clusters=min(config["ALT_CLUSTER_COUNT"],len(points)),random_state=SEED,n_init=5).fit_predict(points)
    elif method=="gmm": labels=GaussianMixture(n_components=min(config["ALT_CLUSTER_COUNT"],len(points)),random_state=SEED).fit_predict(points)
    elif method=="hdbscan":
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError as e: raise RuntimeError("HDBSCAN no está disponible en esta versión de scikit-learn.") from e
        labels=HDBSCAN(min_cluster_size=max(10,config["DBSCAN_MIN"])).fit_predict(points)
    else: raise ValueError("RESIDUAL_METHOD debe ser dbscan, hdbscan, kmeans o gmm.")
    valid=labels>=0
    if valid.any():
        ids,counts=np.unique(labels[valid],return_counts=True); giant=ids[np.argmax(counts)]
        if counts.max()/len(points)>config["GIANT_FRACTION"]:
            gidx=np.flatnonzero(labels==giant); nz=np.abs(normals[gidx,config["UP_AXIS"]])
            fam=np.where(nz>.85,0,np.where(nz<.25,1,2))
            new=labels.copy(); next_id=int(labels.max())+1
            for family in range(3):
                ix=gidx[fam==family]
                if len(ix)<config["DBSCAN_MIN"]: continue
                sub=dbscan_labels(points[ix],eps*config["GIANT_ETA_FACTOR"],config["DBSCAN_MIN"])
                for sl in np.unique(sub):
                    if sl<0: continue
                    si=ix[sub==sl]
                    if len(si)>=config["TINY_POINTS"]: new[si]=next_id; next_id+=1
                    else: new[si]=-1
            labels=new
    return labels,eps

