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

# 10. CARACTERÍSTICAS GEOMÉTRICAS Y REGLAS ====================================

def cylinder_check(points,normals,config,rng):
    """Ajuste circular RANSAC en plano perpendicular al eje PCA principal."""
    n=len(points)
    if n<30: return {"ok":False,"ratio":0.,"radius":None}
    ids=rng.choice(n,min(n,config["CYLINDER_MAX_POINTS"]),replace=False)
    p=points[ids]; nn=normals[ids]
    n=len(p)
    _,_,vh=np.linalg.svd(p-p.mean(axis=0),full_matrices=False); axis=vh[0]
    ref=np.array([1.,0,0]) if abs(axis[0])<.9 else np.array([0.,1,0])
    u=np.cross(axis,ref); u/=np.linalg.norm(u); v=np.cross(axis,u)
    q=np.column_stack(((p-p.mean(axis=0))@u,(p-p.mean(axis=0))@v))
    qn=np.column_stack((nn@u,nn@v)); best=0; best_r=None
    trials=min(config["CYLINDER_TRIALS"],max(20,n//10))
    for _ in range(trials):
        ii=rng.choice(n,3,replace=False); a,b,c=q[ii]
        mat=2*np.array([b-a,c-a]); det=np.linalg.det(mat)
        if abs(det)<1e-8: continue
        center=np.linalg.solve(mat,np.array([b@b-a@a,c@c-a@a])); radius=np.linalg.norm(a-center)
        if not config["CYLINDER_RADIUS_MIN"]<=radius<=config["CYLINDER_RADIUS_MAX"]: continue
        radial=q-center; rr=np.linalg.norm(radial,axis=1)
        agree=np.abs(np.sum((radial/np.maximum(rr[:,None],1e-9))*qn,axis=1))>.65
        inside=(np.abs(rr-radius)<max(.025,config["VOXEL"]*1.5)) & agree
        count=int(inside.sum())
        if count>best:
            angles=np.sort(np.mod(np.arctan2(radial[inside,1],radial[inside,0]),2*np.pi))
            if len(angles)<4: continue
            max_gap=float(np.degrees(np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))))
            if max_gap>config["CYLINDER_MAX_ANGULAR_GAP"]: continue
            best=count; best_r=float(radius)
    ratio=best/n
    return {"ok":ratio>=config["CYLINDER_INLIER_RATIO"],"ratio":ratio,"radius":best_r}

def cluster_features(points,normals,floor_z,config,rng):
    center=points.mean(axis=0); cov=np.cov(points-center,rowvar=False) if len(points)>2 else np.eye(3)*1e-9
    eig,vec=np.linalg.eigh(cov); order=np.argsort(eig)[::-1]; eig=np.maximum(eig[order],1e-12); vec=vec[:,order]
    linear=float((eig[0]-eig[1])/eig[0]); planar=float((eig[1]-eig[2])/eig[0]); spherical=float(eig[2]/eig[0])
    extent=np.ptp(points,axis=0); up=config["UP_AXIS"]
    other=[i for i in range(3) if i!=up]; footprint=float(extent[other[0]]*extent[other[1]])
    base=float(points[:,up].min()-floor_z); height=float(extent[up]); vertical=float(abs(vec[up,0]))
    elongated=linear>=config["ELONGATED_LINEARITY"]
    cyl=cylinder_check(points,normals,config,rng) if linear>=config["PIPE_CHECK_LINEARITY"] else {"ok":False,"ratio":0.,"radius":None}
    return {"linearidad":linear,"planaridad":planar,"esfericidad":spherical,"verticalidad":vertical,
            "extensiones":extent,"huella":footprint,"altura":height,"base_piso":base,
            "elongado":elongated,"cilindro":cyl}

def classify_features(f,config):
    if f["planaridad"]>=config["WALL_PLANARITY"] and f["verticalidad"]>=config["WALL_VERTICALITY"]: return 8
    if f["elongado"] and f["verticalidad"]>=config["COLUMN_VERTICALITY"]: return 3
    if f["cilindro"]["ok"] and f["verticalidad"]<config["COLUMN_VERTICALITY"]: return 4
    if f["elongado"]: return 5
    if config["PERSON_HEIGHT_MIN"]<=f["altura"]<=config["PERSON_HEIGHT_MAX"] and f["huella"]<=config["PERSON_FOOTPRINT_MAX"] and f["base_piso"]<=config["PERSON_BASE_MAX"]: return 6
    if f["huella"]>=config["MACHINE_FOOTPRINT_MIN"] and f["altura"]<=config["MACHINE_HEIGHT_MAX"] and f["base_piso"]<=config["MACHINE_BASE_MAX"]: return 7
    return 8

# 11. PIPELINE COMPLETO DE CLASIFICACIÓN ======================================

def predict(xyz,rgb=None,normals=None,config=None):
    """Devuelve etiquetas, ID de instancia y metadatos del pipeline por punto original."""
    cfg=dict(CONFIG); cfg.update(config or {}); rng=np.random.default_rng(cfg["SEED"])
    o3d.utility.random.seed(int(cfg["SEED"]))
    xyz=np.asarray(xyz,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3: raise ValueError("xyz debe tener forma (N,3).")
    started=time.perf_counter(); vox,mapping=timed("0_voxel",len(xyz),lambda:voxelize(xyz,cfg["VOXEL"]))
    vrgb,vfile=timed("atributos_voxel",len(vox),lambda:voxel_attributes(len(vox),mapping,
                     rgb if cfg["USE_COLOR"] else None, normals if cfg["USE_FILE_NORMALS"] else None))
    vn=timed("normales",len(vox),lambda:estimate_normals(vox,vfile,cfg))
    sp=cKDTree(vox).query(vox[rng.choice(len(vox),min(5000,len(vox)),replace=False)],k=2,workers=-1)[0][:,1]
    cfg["_RG_RADIUS"]=max(cfg["RG_RADIUS_MIN"],cfg["RG_RADIUS_FACTOR"]*float(np.median(sp)))
    cid=centers=cdist=None
    if cfg["USE_COLOR"] and vrgb is not None and cfg["USE_RG"]:
        (cid,centers,cdist)=timed("color_meanshift",len(vox),lambda:color_module(vox,vrgb,cfg,rng))
    check_up_axis(vox,vn,cfg)
    floor,ceiling,fz,cz,floor_levels=timed("1_piso_techo",len(vox),lambda:detect_floor_ceiling(vox,vn,cfg))
    excluded=floor|ceiling
    vlabels=np.full(len(vox),-1,int); instance=np.full(len(vox),-1,int); next_inst=0; n_cols=0
    if cfg["USE_COLUMN_STAGE"]:   # [COLUMNAS] se detectan antes de RANSAC/clusters para que ningún paso las absorba
        col_id,n_cols=timed("1b_columnas",int((~excluded).sum()),lambda:detect_columns(vox,excluded,fz,cz,cfg))
        is_col=col_id>=0; vlabels[is_col]=3; instance[is_col]=col_id[is_col]; next_inst=n_cols
        excluded=excluded|is_col
    planes=timed("2_planos_ransac",int((~excluded).sum()),lambda:extract_planes(vox,vn,excluded,cfg))
    vlabels[floor]=0; vlabels[ceiling]=1
    for pid in np.unique(planes[planes>=0]):
        ix=np.flatnonzero(planes==pid)
        # Los planos horizontales intermedios son pisos de otros niveles/edificios.
        # Los planos verticales permanecen como "otro" (no se reporta clase pared).
        horizontal=float(np.median(np.abs(vn[ix,cfg["UP_AXIS"]]))) >= cfg["NORMAL_UP"]
        plane_z=float(np.median(vox[ix,cfg["UP_AXIS"]]))
        if horizontal:
            zq=np.percentile(vox[ix,cfg["UP_AXIS"]],[5,95])
            if zq[1]-zq[0]>cfg["FLOOR_PLANE_MAX_Z_SPREAD"]:
                # La mediana puede coincidir con un nivel de piso aunque el segmento suba varios metros.
                planes[ix]=-1
                continue
            ax=[i for i in range(3) if i!=cfg["UP_AXIS"]]
            xy=vox[ix][:,ax]; side=np.ptp(xy,axis=0)
            cell=cfg["FLOOR_PLANE_GRID"]
            ij=np.floor((xy-xy.min(0))/cell).astype(int)
            grid=ij.max(0)+1; occupied=np.zeros(grid,dtype=bool); occupied[ij[:,0],ij[:,1]]=True
            coverage=float(occupied.mean())
            broad=(min(side)>=cfg["FLOOR_PLANE_MIN_SIDE"] and np.prod(side)>=cfg["FLOOR_PLANE_MIN_AREA"]
                   and coverage>=cfg["FLOOR_PLANE_MIN_OCCUPANCY"])
            if not broad:
                # Un parche horizontal angosto puede ser la cara superior de una tubería/viga.
                # Devuélvelo al agrupamiento geométrico para que se evalúe su sección.
                planes[ix]=-1
                continue
            if plane_z>=cz-cfg["CEILING_BAND"]:
                vlabels[ix]=1
            elif np.any(np.abs(floor_levels-plane_z)<=1.5*cfg["FLOOR_BAND"]):
                # Un piso tiene soporte XY en los cuatro cuadrantes alrededor de cada celda.
                # Una corona de tubería es una franja y no debe heredar la etiqueta piso.
                quadrant_counts=[]
                for rs,cs in ((slice(0,2),slice(0,2)),(slice(0,2),slice(3,5)),
                              (slice(3,5),slice(0,2)),(slice(3,5),slice(3,5))):
                    kernel=np.zeros((5,5),dtype=np.int8); kernel[rs,cs]=1
                    quadrant_counts.append(convolve(occupied.astype(np.int8),kernel,mode="constant"))
                supported=np.logical_and.reduce([
                    q[ij[:,0],ij[:,1]]>=cfg["FLOOR_PLANE_QUADRANT_SUPPORT"] for q in quadrant_counts])
                floor_ix=ix[supported]; pipe_ix=ix[~supported]
                if len(floor_ix):
                    vlabels[floor_ix]=0; instance[floor_ix]=next_inst; next_inst+=1
                if len(pipe_ix): planes[pipe_ix]=-1
                continue
            else:
                # Un plano horizontal fuera de los niveles de piso verificados puede ser una tubería/viga.
                planes[ix]=-1
                continue
        else:
            vlabels[ix]=8
        instance[ix]=next_inst; next_inst+=1
    residual=np.flatnonzero((~excluded)&(planes<0))
    cluster_labels,eps=timed("3_residuo_cluster",len(residual),lambda:cluster_residual(vox[residual],vn[residual],cfg,rng))
    clusters=np.unique(cluster_labels[cluster_labels>=0])
    for lab in clusters:
        loc=np.flatnonzero(cluster_labels==lab); ix=residual[loc]
        if len(ix)<cfg["TINY_POINTS"]: continue
        cl=classify_features(cluster_features(vox[ix],vn[ix],fz,cfg,rng),cfg)
        vlabels[ix]=cl; instance[ix]=next_inst; next_inst+=1
        if cl==8 and cfg["USE_RG"] and len(ix)>=cfg["REFINE_MIN_POINTS"]:
            next_inst=resolve_unclassified(ix,vox,vn,cid,cdist,fz,cfg,rng,vlabels,instance,next_inst)
        if cl in (5,8) and cfg["USE_RG"] and len(ix)>=cfg["REFINE_MIN_POINTS"]:
            next_inst=refine_pipe_segments(ix,vox,vn,cid,cdist,fz,cfg,rng,vlabels,instance,next_inst)
    if cfg["FILL_UNASSIGNED"]:
        un=np.flatnonzero(vlabels<0); ok_=np.flatnonzero(vlabels>=0)
        if len(un) and len(ok_):
            d_,nn_=cKDTree(vox[ok_]).query(vox[un],distance_upper_bound=cfg["FILL_RADIUS"],workers=-1)
            hit=np.isfinite(d_); dest=un[hit]; src_=ok_[nn_[hit]]
            vertical_ok=(~np.isin(vlabels[src_],[0,1]) |
                         (np.abs(vox[dest,cfg["UP_AXIS"]]-vox[src_,cfg["UP_AXIS"]])<=cfg["FILL_VERTICAL_TOL"]))
            dest=dest[vertical_ok]; src_=src_[vertical_ok]
            vlabels[dest]=vlabels[src_]; instance[dest]=instance[src_]
        # Completa puntos aislados sin propagar piso/techo a otra altura.
        un=np.flatnonzero(vlabels<0); ok_=np.flatnonzero(vlabels>=0)
        if len(un) and len(ok_):
            usable=ok_[~np.isin(vlabels[ok_],[0,1])]
            if len(usable):
                _,nn_=cKDTree(vox[usable]).query(vox[un],k=1,workers=-1)
                vlabels[un]=vlabels[usable[nn_]]; instance[un]=instance[usable[nn_]]
            else:
                vlabels[un]=8; instance[un]=next_inst; next_inst+=1

    outlab=np.full(len(xyz),-1,int); outinst=np.full(len(xyz),-1,int)
    mapped=np.zeros(len(xyz),bool)
    for vi,orig in enumerate(mapping):
        orig=orig[(orig>=0)&(orig<len(xyz))]
        outlab[orig]=vlabels[vi]; outinst[orig]=instance[vi]; mapped[orig]=True
    missing=np.flatnonzero(~mapped)
    if len(missing):
        _,nn=cKDTree(vox).query(xyz[missing],k=1,workers=-1)
        outlab[missing]=vlabels[nn]; outinst[missing]=instance[nn]
    # Limpia manchas elevadas ya expandidas al PLY cuando están rodeadas por tubería.
    raised=np.flatnonzero((outlab==0)&(xyz[:,cfg["UP_AXIS"]]>fz+2*cfg["FLOOR_BAND"]))
    pipes=np.flatnonzero(outlab==4); reassigned=0
    if len(raised) and len(pipes):
        local=cKDTree(xyz).query_ball_point(xyz[raised],r=cfg["FLOOR_PIPE_REASSIGN_RADIUS"])
        n_pipe=np.fromiter((np.sum(outlab[np.asarray(ids,dtype=int)]==4) for ids in local),dtype=int,count=len(local))
        n_all=np.fromiter((len(ids) for ids in local),dtype=int,count=len(local))
        promote=(n_pipe>=3)&(n_pipe/np.maximum(n_all,1)>=cfg["FLOOR_PIPE_REASSIGN_SHARE"])
        ix=raised[promote]
        if len(ix):
            _,nn=cKDTree(xyz[pipes]).query(xyz[ix],k=1,workers=-1)
            outlab[ix]=4; outinst[ix]=outinst[pipes[nn]]; reassigned=int(len(ix))
    TIMINGS.append({"etapa":"total","segundos":time.perf_counter()-started,"puntos":len(xyz),"puntos_segundo":len(xyz)/max(time.perf_counter()-started,1e-9)})
    return {"labels":outlab,"instance_id":outinst,"floor_z":fz,"floor_levels":floor_levels.tolist(),"ceiling_z":cz,"eps":eps,"color_centers":centers,"rg_radius":cfg.get("_RG_RADIUS"),"n_columns":n_cols,"floor_pipe_reassigned":reassigned}


# 12. MÉTRICAS Y EXPORTACIÓN ==================================================

def self_consistency(xyz,labels,config):
    rng=np.random.default_rng(config["SEED"]); n=min(50000,len(xyz)); ix=rng.choice(len(xyz),n,replace=False)
    k=min(11,len(xyz)); _,nn=cKDTree(xyz).query(xyz[ix],k=k,workers=-1)
    neigh=labels[nn[:,1:]]; own=labels[ix]
    disagree=np.mean(np.mean(neigh!=own[:,None],axis=1)>.5) if neigh.size else 0.
    sizes=np.unique(labels[labels>=0],return_counts=True)[1]
    return {"porcentaje_vecinos_mayoria_otra_clase":float(disagree*100),"porcentaje_sin_asignar":float(np.mean(labels<0)*100),
            "segmentos_pequenos":int(np.sum(sizes<config["TINY_POINTS"])),"floor_below_ceiling":bool(RESULT["floor_z"]<RESULT["ceiling_z"]),
            "floor_far_above_level":bool(np.any((labels==0)&(xyz[:,config["UP_AXIS"]]>RESULT["floor_z"]+config["FLOOR_BAND"]*2)))}

def export_results(xyz,result,config):
    out=Path(config["OUTPUT_DIR"]); out.mkdir(parents=True,exist_ok=True)
    labels=result["labels"]; inst=result["instance_id"]
    color=class_colors(labels)
    pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)); pc.colors=o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(str(out/"ply_by_class.ply"),pc,write_ascii=False)
    counts={CLASS_NAMES[c]:int(np.sum(labels==c)) for c in CLASS_NAMES}
    sizes=np.unique(inst[inst>=0],return_counts=True)[1] if np.any(inst>=0) else np.array([],int)
    stats={"points":len(xyz),"points_per_class":counts,"percent_unassigned":float(np.mean(labels<0)*100),
           "cluster_size_distribution":{"min":int(sizes.min()) if sizes.size else 0,
             "median":float(np.median(sizes)) if sizes.size else 0,"max":int(sizes.max()) if sizes.size else 0},
           "tiny_segments":int(np.sum(sizes<config["TINY_POINTS"])),
           "columns_detected":result.get("n_columns",0),"floor_z":result["floor_z"],"floor_levels":result.get("floor_levels",[result["floor_z"]]),"ceiling_z":result["ceiling_z"],"floor_pipe_reassigned":result.get("floor_pipe_reassigned",0),"adaptive_eps":result["eps"],"self_consistency":self_consistency(xyz,labels,config),"timings":TIMINGS,
           "config":{"VOXEL":config["VOXEL"],"ETA":config["ETA"],"DBSCAN_MIN":config["DBSCAN_MIN"],
                     "GIANT_FRACTION":config["GIANT_FRACTION"],"GIANT_ETA_FACTOR":config["GIANT_ETA_FACTOR"],
                     "RESIDUAL_METHOD":config["RESIDUAL_METHOD"]},
           "validation":"Reglas geométricas; no validado contra ground truth."}

    (out/"stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    return out,stats


# 13. CARGA Y EJECUCIÓN =======================================================

def get_ply(cfg):
    p = Path(cfg["PLY_PATH"])
    if not p.exists(): raise FileNotFoundError(f"No se encontró {p}")
    return str(p)

def subsample(xyz, rgb, normals, max_points, seed):
    if max_points is None or max_points >= len(xyz): return xyz, rgb, normals
    ix = np.sort(np.random.default_rng(seed).choice(len(xyz), max_points, replace=False))
    return xyz[ix], None if rgb is None else rgb[ix], None if normals is None else normals[ix]

_xyz0, _rgb0, _nrm0 = load_cloud(get_ply(CONFIG))
print(f"Nube original: {len(_xyz0):,} puntos | RGB: {_rgb0 is not None} | normales: {_nrm0 is not None}")
_xyz, _rgb, _nrm = subsample(_xyz0, _rgb0, _nrm0, CONFIG["MAX_POINTS"], CONFIG["SEED"])
print(f"Puntos usados: {len(_xyz):,}")


TIMINGS.clear()
RESULT = predict(_xyz, _rgb if CONFIG["USE_RGB"] else None, _nrm)
OUT, STATS = export_results(_xyz, RESULT, CONFIG)
print("Resultados:", OUT)
print("Piso:", round(RESULT["floor_z"], 3), "| techo:", round(RESULT["ceiling_z"], 3), "| eps DBSCAN:", round(RESULT["eps"], 3),
      "| radio region growing:", round(RESULT["rg_radius"], 3))
if RESULT["color_centers"] is not None:
    print("Centros de color (Mean-Shift):\n", np.round(RESULT["color_centers"], 2))
print("Columnas detectadas (etapa de continuidad vertical):", RESULT["n_columns"])
print("Etapa                         segundos       puntos/s")
for row in TIMINGS: print(f"{row['etapa']:<28} {row['segundos']:>8.2f} {row['puntos_segundo']:>14,.0f}")
print("\nPuntos por clase:")
for k, v in STATS["points_per_class"].items(): print(f"  {k:<18}{v:>10,}  {100*v/len(_xyz):5.1f}%")

# 14. COMPARACIÓN VISUAL ORIGINAL / CLASIFICACIÓN =============================

def _camera_basis(view_direction, up_axis):
    up=np.eye(3)[up_axis].astype(float)
    direction=np.asarray(view_direction,dtype=float); direction/=np.linalg.norm(direction)
    right=np.cross(direction,up)
    if np.linalg.norm(right)<1e-8: right=np.eye(3)[(up_axis+1)%3]
    right/=np.linalg.norm(right)
    camera_up=np.cross(right,direction); camera_up/=np.linalg.norm(camera_up)
    return right,camera_up,direction

def _project(points,basis):
    right,camera_up,depth=basis
    return points@right,points@camera_up,points@depth

def _auto_crop(points,labels,instances,class_id,margin):
    candidates=np.flatnonzero(labels==class_id)
    if not len(candidates): return None
    ids=instances[candidates]; ids=ids[ids>=0]
    if len(ids):
        inst_id,count=np.unique(ids,return_counts=True)
        candidates=candidates[instances[candidates]==int(inst_id[np.argmax(count)])]
    lo=points[candidates].min(0); hi=points[candidates].max(0)
    pad=np.maximum(margin,np.maximum(hi-lo,0.25)*0.20)
    return np.column_stack((lo-pad,hi+pad))

def _manual_roi_mask(points,roi):
    if roi is None: return None
    try: bounds=np.asarray([roi[axis] for axis in ("x","y","z")],dtype=float)
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('COMPARE_PERSON_ROI debe tener formato {"x":[min,max],"y":[min,max],"z":[min,max]}.') from exc
    if bounds.shape!=(3,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0]>=bounds[:,1]):
        raise ValueError("COMPARE_PERSON_ROI requiere límites XYZ finitos, crecientes y en metros.")
    return np.all((points>=bounds[:,0])&(points<=bounds[:,1]),axis=1)

def _view_specs(up_axis):
    horizontal=[i for i in range(3) if i!=up_axis]; x_axis,y_axis=horizontal
    specs=[]
    for angle,name in zip((45,135,225,315),("Esquina XY +/+","Esquina XY -/+","Esquina XY -/-","Esquina XY +/-")):
        az=np.radians(angle); elev=np.radians(30)
        direction=(np.cos(az)*np.eye(3)[x_axis]+np.sin(az)*np.eye(3)[y_axis])*np.cos(elev)+np.eye(3)[up_axis]*np.sin(elev)
        specs.append((name,direction))
    for angle,name in zip((0,90,180,270),("Lateral +X","Lateral +Y","Lateral -X","Lateral -Y")):
        az=np.radians(angle); direction=np.cos(az)*np.eye(3)[x_axis]+np.sin(az)*np.eye(3)[y_axis]
        specs.append((name,direction))
    specs.extend((("Superior",np.eye(3)[up_axis]),("Inferior",-np.eye(3)[up_axis])))
    return specs

def render_comparison(xyz,rgb,labels,instances,path,config):
    """Compara original y clases en diez cámaras comunes y dos acercamientos."""
    if len(xyz)!=len(labels) or len(xyz)!=len(instances):
        raise ValueError("XYZ, etiquetas e instancias deben contener el mismo número de puntos.")
    rng=np.random.default_rng(config["SEED"])
    common=rng.choice(len(xyz),min(config["COMPARE_MAX_POINTS"],len(xyz)),replace=False)
    points=xyz[common]
    original=(np.tile([0.68,0.68,0.68],(len(common),1)) if rgb is None else np.clip(rgb[common],0,1))
    classified=class_colors(labels[common]); up=config["UP_AXIS"]

    roi=config.get("COMPARE_PERSON_ROI"); manual=_manual_roi_mask(xyz,roi)
    if manual is not None:
        person_ix=np.flatnonzero(manual); person_title="Personas — recorte XYZ manual"
    else:
        box=_auto_crop(xyz,labels,instances,5,config["COMPARE_CROP_MARGIN"])
        person_ix=np.empty(0,int) if box is None else np.flatnonzero(np.all((xyz>=box[:,0])&(xyz<=box[:,1]),axis=1))
        person_title="Personas — mayor instancia detectada"
    pipe_box=_auto_crop(xyz,labels,instances,3,config["COMPARE_CROP_MARGIN"])
    pipe_ix=np.empty(0,int) if pipe_box is None else np.flatnonzero(np.all((xyz>=pipe_box[:,0])&(xyz<=pipe_box[:,1]),axis=1))

    fig,axs=plt.subplots(6,4,figsize=(18,25),facecolor="white",squeeze=False)
    for view_ix,(name,direction) in enumerate(_view_specs(up)):
        basis=_camera_basis(direction,up); px,py,depth=_project(points,basis); order=np.argsort(depth)
        for panel,colors,title in ((0,original,"Original (RGB)"),(1,classified,"Clasificación")):
            ax=axs[view_ix//2,(view_ix%2)*2+panel]
            ax.scatter(px[order],py[order],c=colors[order],s=0.28,marker=".",linewidths=0,rasterized=True)
            ax.set_aspect("equal",adjustable="box"); ax.axis("off"); ax.set_title(title,fontsize=8)
            padx=max(float(np.ptp(px))*0.025,0.02); pady=max(float(np.ptp(py))*0.025,0.02)
            ax.set_xlim(float(px.min()-padx),float(px.max()+padx)); ax.set_ylim(float(py.min()-pady),float(py.max()+pady))
        axs[view_ix//2,(view_ix%2)*2].text(0.02,0.98,name,transform=axs[view_ix//2,(view_ix%2)*2].transAxes,
                                             va="top",ha="left",fontsize=9,bbox={"facecolor":"white","alpha":0.8,"edgecolor":"none","pad":2})

    crops=((person_title,person_ix),("Tuberías — mayor instancia detectada",pipe_ix))
    for crop_ix,(title,indices) in enumerate(crops):
        left=crop_ix*2; ax_l,ax_r=axs[5,left],axs[5,left+1]
        if len(indices):
            p=xyz[indices]; c0=np.tile([0.68,0.68,0.68],(len(indices),1)) if rgb is None else np.clip(rgb[indices],0,1)
            c1=class_colors(labels[indices]); basis=_camera_basis(np.eye(3)[up],up)
            px,py,depth=_project(p,basis); order=np.argsort(depth)
            for ax,colors,label in ((ax_l,c0,"Original (RGB)"),(ax_r,c1,"Clasificación")):
                ax.scatter(px[order],py[order],c=colors[order],s=1.2,marker=".",linewidths=0,rasterized=True)
                ax.set_aspect("equal",adjustable="box"); ax.axis("off"); ax.set_title(label,fontsize=9)
                padx=max(float(np.ptp(px))*0.06,0.02); pady=max(float(np.ptp(py))*0.06,0.02)
                ax.set_xlim(float(px.min()-padx),float(px.max()+padx)); ax.set_ylim(float(py.min()-pady),float(py.max()+pady))
        else:
            for ax,label in ((ax_l,"Original (RGB)"),(ax_r,"Clasificación")):
                ax.axis("off"); ax.set_title(label,fontsize=9); ax.text(0.5,0.5,"Sin puntos para mostrar",ha="center",va="center",transform=ax.transAxes)
        ax_l.text(0.02,0.98,title,transform=ax_l.transAxes,va="top",ha="left",fontsize=9,
                  bbox={"facecolor":"white","alpha":0.85,"edgecolor":"none","pad":2})
        if crop_ix==0 and roi is None and not len(indices):
            ax_r.text(0.05,0.80,"No se detectaron personas.\nConfigura COMPARE_PERSON_ROI para inspeccionar\nla esquina manualmente.",transform=ax_r.transAxes,va="top",fontsize=8)
    handles=[plt.Line2D([0],[0],marker="o",ls="",color=COLORS[c],
                        label=f"{CLASS_NAMES[c]} ({np.sum(labels==c):,} puntos)")
             for c in CLASS_NAMES if np.any(labels==c)]
    if np.any(labels<0):
        handles.append(plt.Line2D([0],[0],marker="o",ls="",color=(.2,.2,.2),
                                  label=f"sin asignar ({np.sum(labels<0):,} puntos)"))
    fig.legend(handles=handles,loc="lower center",ncol=4,fontsize=8,frameon=False)
    fig.suptitle(f"Comparación PLY original / clasificación — {len(xyz):,} puntos; muestra común {len(common):,}",fontsize=15)
    fig.tight_layout(rect=(0,0.025,1,0.975),h_pad=0.55,w_pad=0.15)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)

COMPARISON_PATH=Path(OUT)/"clasificacion.png"
render_comparison(_xyz,_rgb,RESULT["labels"],RESULT["instance_id"],COMPARISON_PATH,CONFIG)
print("Comparación original/clasificación guardada en",COMPARISON_PATH)
