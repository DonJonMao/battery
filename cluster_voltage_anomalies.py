import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable
import warnings
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-battery")
warnings.filterwarnings("ignore", message="Glyph .* missing from font", category=UserWarning)


@dataclass(frozen=True)
class CategoryConfig:
    name: str
    point_type_pattern: str
    min_points: int
    resample_points: int
    max_shift_frac: float
    dbscan_min_samples: int
    far_quantile: float
    knn_quantile: float
    max_plots: int
    min_window_points: int
    hard_max_voltage: float | None
    hard_min_voltage: float | None
    similarity_threshold: float
    similarity_min_peers: int


PLOT_MAX_SAMPLE = 60
NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def load_voltage_rows(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.xlsx"))
    if not files:
        return pd.DataFrame()
    rows = []
    for file in files:
        try:
            xls = pd.ExcelFile(file)
        except Exception:
            continue
        sheet_names = [s for s in xls.sheet_names if s.upper().startswith("DC")]
        if not sheet_names:
            sheet_names = [xls.sheet_names[0]]
        for sheet in sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet)
            except Exception:
                continue
            df = df.copy()
            df["__file"] = file.name
            df["__sheet"] = sheet
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    df_all = pd.concat(rows, ignore_index=True)
    df_all = df_all.rename(columns=str.strip)
    return df_all


def excel_col_to_index(cell_ref: str) -> int:
    letters = []
    for ch in cell_ref:
        if ch.isalpha():
            letters.append(ch.upper())
        else:
            break
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def load_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    out = []
    for si in root.findall(f"{NS_MAIN}si"):
        parts = []
        for t in si.iter(f"{NS_MAIN}t"):
            parts.append(t.text or "")
        out.append("".join(parts))
    return out


def read_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        node = cell.find(f"{NS_MAIN}is/{NS_MAIN}t")
        return node.text if node is not None else ""
    value_node = cell.find(f"{NS_MAIN}v")
    if value_node is None:
        return ""
    text = value_node.text or ""
    if cell_type == "s":
        try:
            return shared_strings[int(text)]
        except Exception:
            return text
    return text


def read_sheet_xml(zf: ZipFile, target: str, shared_strings: list[str]) -> pd.DataFrame:
    xml_path = target if target.startswith("xl/") else f"xl/{target}"
    root = ET.fromstring(zf.read(xml_path))
    sheet_data = root.find(f"{NS_MAIN}sheetData")
    if sheet_data is None:
        return pd.DataFrame()

    rows = []
    max_cols = 0
    for row in sheet_data.findall(f"{NS_MAIN}row"):
        values: dict[int, str] = {}
        for cell in row.findall(f"{NS_MAIN}c"):
            ref = cell.attrib.get("r", "")
            if not ref:
                continue
            col_idx = excel_col_to_index(ref)
            values[col_idx] = read_cell_value(cell, shared_strings)
            max_cols = max(max_cols, col_idx + 1)
        if not values:
            continue
        row_values = [""] * max_cols
        for idx, value in values.items():
            if idx >= len(row_values):
                row_values.extend([""] * (idx + 1 - len(row_values)))
            row_values[idx] = value
        rows.append(row_values)
    if not rows:
        return pd.DataFrame()

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = [str(x).strip() for x in rows[0]]
    data = rows[1:]
    if not header:
        header = [f"col_{i}" for i in range(width)]
    return pd.DataFrame(data, columns=header)


def list_sheet_targets(zf: ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    sheets = workbook.find(f"{NS_MAIN}sheets")
    if sheets is None:
        return []
    out = []
    for sheet in sheets:
        name = sheet.attrib.get("name", "")
        rel_id = sheet.attrib.get(f"{NS_REL}id", "")
        target = rel_map.get(rel_id, "")
        if name and target:
            out.append((name, target))
    return out


def load_voltage_rows_xml(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.xlsx"))
    if not files:
        return pd.DataFrame()
    rows = []
    for file in files:
        try:
            with ZipFile(file) as zf:
                shared_strings = load_shared_strings(zf)
                sheet_targets = list_sheet_targets(zf)
                sheet_names = [name for name, _ in sheet_targets if name.upper().startswith("DC")]
                if not sheet_names and sheet_targets:
                    sheet_names = [sheet_targets[0][0]]
                target_map = dict(sheet_targets)
                for sheet_name in sheet_names:
                    df = read_sheet_xml(zf, target_map[sheet_name], shared_strings)
                    if df.empty:
                        continue
                    df = df.copy()
                    df["__file"] = file.name
                    df["__sheet"] = sheet_name
                    rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df_all = pd.concat(rows, ignore_index=True)
    df_all = df_all.rename(columns=str.strip)
    return df_all


def load_voltage_rows_any(input_dir: Path) -> pd.DataFrame:
    df_all = load_voltage_rows(input_dir)
    if not df_all.empty:
        return df_all
    return load_voltage_rows_xml(input_dir)


def to_seconds(t: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(t):
        return (t - t.min()).dt.total_seconds().to_numpy()
    t_parsed = pd.to_datetime(t, errors="coerce")
    if t_parsed.notna().any():
        return (t_parsed - t_parsed.min()).dt.total_seconds().to_numpy()
    return pd.to_numeric(t, errors="coerce").to_numpy()


def smooth_series(v: np.ndarray) -> np.ndarray:
    n = len(v)
    if n < 5:
        return v
    win = max(3, min(11, (n // 10) * 2 + 1))
    return pd.Series(v).rolling(win, center=True, min_periods=1).median().to_numpy()


def find_discharge_window(v: np.ndarray, min_len: int) -> tuple[int, int] | None:
    n = len(v)
    if n < min_len:
        return None
    v_smooth = smooth_series(v)
    v_max = v_smooth.max()
    v_min = v_smooth.min()
    drop = v_max - v_min
    if drop <= 1e-6:
        return None
    start = np.argmax(v_smooth <= v_max - 0.05 * drop)
    if v_smooth[start] > v_max - 0.05 * drop:
        return None
    end = np.argmax(v_smooth <= v_max - 0.95 * drop)
    if v_smooth[end] > v_max - 0.95 * drop:
        end = n - 1
    if end <= start:
        end = n - 1
    if end - start + 1 < min_len:
        return None
    return start, end


def build_series(df: pd.DataFrame, cfg: CategoryConfig) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    ids = []
    v_raw = []
    meta_rows = []
    for series_id, g in df.groupby("series_id", dropna=False):
        g = g.sort_values("上报时间")
        g = g.copy()
        g["监控值"] = pd.to_numeric(g["监控值"], errors="coerce")
        g = g.groupby("上报时间", dropna=False, as_index=False).agg({"监控值": "mean"})
        if len(g) < cfg.min_points:
            continue
        t = to_seconds(g["上报时间"])
        v = pd.to_numeric(g["监控值"], errors="coerce").to_numpy()
        if np.isnan(v).all():
            continue
        mask = (~np.isnan(v)) & (~np.isnan(t))
        t = t[mask]
        v = v[mask]
        if len(v) < cfg.min_points:
            continue
        window = find_discharge_window(v, cfg.min_window_points)
        window_found = window is not None
        if window is None:
            s, e = 0, len(v) - 1
        else:
            s, e = window
        t = t[s : e + 1]
        v = v[s : e + 1]
        if len(v) < cfg.min_points:
            continue
        if t[-1] == t[0]:
            continue
        t_norm = (t - t[0]) / (t[-1] - t[0])
        t_new = np.linspace(0, 1, cfg.resample_points)
        v_new = np.interp(t_new, t_norm, v)
        v_raw.append(v_new)
        ids.append(series_id)
        meta_rows.append(
            {
                "series_id": series_id,
                "n_points": len(v),
                "duration_s": t[-1] - t[0],
                "window_found": window_found,
                "v_min": float(np.min(v)),
                "v_max": float(np.max(v)),
                "v_range": float(np.max(v) - np.min(v)),
            }
        )
    if not v_raw:
        return np.empty((0, cfg.resample_points)), np.empty((0, cfg.resample_points)), [], pd.DataFrame()
    meta = pd.DataFrame(meta_rows)
    return np.vstack(v_raw), np.vstack(v_raw), ids, meta


def align_series(v: np.ndarray, max_shift: int) -> tuple[np.ndarray, np.ndarray]:
    n, m = v.shape
    aligned = np.zeros_like(v)
    shifts = np.zeros(n, dtype=int)
    ref = np.median(v, axis=0)
    for i in range(n):
        best_shift = 0
        best_mse = np.inf
        for s in range(-max_shift, max_shift + 1):
            if s < 0:
                cur = np.pad(v[i, -s:], (0, -s), mode="edge")
            elif s > 0:
                cur = np.pad(v[i, :-s], (s, 0), mode="edge")
            else:
                cur = v[i]
            mse = np.mean((cur - ref) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_shift = s
        shifts[i] = best_shift
        if best_shift < 0:
            aligned[i] = np.pad(v[i, -best_shift:], (0, -best_shift), mode="edge")
        elif best_shift > 0:
            aligned[i] = np.pad(v[i, :-best_shift], (best_shift, 0), mode="edge")
        else:
            aligned[i] = v[i]
    return aligned, shifts


def feature_matrix(v_raw: np.ndarray, v_aligned: np.ndarray, shifts: np.ndarray) -> tuple[np.ndarray, list[str]]:
    v_mean = v_raw.mean(axis=1)
    v_range = v_raw.max(axis=1) - v_raw.min(axis=1)
    dv = np.diff(v_aligned, axis=1)
    d2 = np.diff(v_aligned, n=2, axis=1)
    d2_std = d2.std(axis=1)
    zigzag = np.mean(np.abs(dv), axis=1)
    shift_abs = np.abs(shifts)
    X = np.vstack([v_mean, v_range, d2_std, zigzag, shift_abs]).T
    names = ["v_mean", "v_range", "d2_std", "zigzag", "shift_abs"]
    return X, names


def dbscan_auto(X: np.ndarray, min_samples: int) -> tuple[float, np.ndarray, float, int]:
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    kdist = np.sort(dists[:, -1])
    eps = float(np.quantile(kdist, 0.9))
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    noise_ratio = float(np.mean(labels == -1))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return eps, labels, noise_ratio, n_clusters


def score_outliers(X: np.ndarray, labels: np.ndarray, k_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(X)))
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    knn_score = dists.mean(axis=1)
    cluster_dist = np.zeros(len(X))
    for lab in set(labels):
        mask = labels == lab
        if lab == -1 or mask.sum() == 0:
            cluster_dist[mask] = knn_score[mask]
            continue
        center = X[mask].mean(axis=0)
        cluster_dist[mask] = np.linalg.norm(X[mask] - center, axis=1)
    return knn_score, cluster_dist


def detect_anomalies(
    X: np.ndarray,
    labels: np.ndarray,
    knn_score: np.ndarray,
    cluster_dist: np.ndarray,
    far_quantile: float,
    knn_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    is_far = cluster_dist >= np.quantile(cluster_dist, far_quantile)
    is_knn_high = knn_score >= np.quantile(knn_score, knn_quantile)
    anomaly = (labels == -1) | (is_far & is_knn_high)
    return anomaly, is_far, is_knn_high


def plot_overlay(
    out_dir: Path,
    category_name: str,
    anomaly_row: pd.Series,
    v_raw_by_id: dict[str, np.ndarray],
    v_aligned_by_id: dict[str, np.ndarray],
    meta: pd.DataFrame,
    ids: list[str],
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    series_id = anomaly_row["series_id"]
    target_raw = v_raw_by_id[series_id]
    target_aligned = v_aligned_by_id[series_id]
    same_group = (meta["设备"] == anomaly_row["设备"]) & (meta["组别"] == anomaly_row["组别"]) & (meta["机房"] == anomaly_row["机房"])
    peer_ids = meta.loc[same_group, "series_id"].tolist()
    peer_ids = [pid for pid in peer_ids if pid != series_id]
    if len(peer_ids) > PLOT_MAX_SAMPLE:
        peer_ids = list(np.random.default_rng(42).choice(peer_ids, size=PLOT_MAX_SAMPLE, replace=False))
    x = np.linspace(0, 1, len(target_raw))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    for pid in peer_ids:
        axes[0].plot(x, v_raw_by_id[pid], color="0.7", linewidth=0.8, alpha=0.6)
        axes[1].plot(x, v_aligned_by_id[pid], color="0.7", linewidth=0.8, alpha=0.6)
    if peer_ids:
        med_raw = np.median(np.vstack([v_raw_by_id[pid] for pid in peer_ids]), axis=0)
        med_aligned = np.median(np.vstack([v_aligned_by_id[pid] for pid in peer_ids]), axis=0)
        axes[0].plot(x, med_raw, color="black", linewidth=2)
        axes[1].plot(x, med_aligned, color="black", linewidth=2)
    axes[0].plot(x, target_raw, color="red", linewidth=2)
    axes[1].plot(x, target_aligned, color="red", linewidth=2)
    axes[0].set_title(f"{category_name} 绝对电压")
    axes[1].set_title(f"{category_name} 形状对齐")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    title = f"{category_name}__{anomaly_row['__file']}__{anomaly_row['__sheet']}__{anomaly_row['机房']}__{anomaly_row['设备']}__{anomaly_row['组别']}__{anomaly_row['监控量']}"
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{title}.png")
    plt.close(fig)


def run_category(df: pd.DataFrame, cfg: CategoryConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_cat = df[df["测点类型"].astype(str).str.contains(cfg.point_type_pattern, na=False)].copy()
    if df_cat.empty:
        return pd.DataFrame(), pd.DataFrame()
    df_cat = df_cat.dropna(subset=["监控量", "上报时间", "监控值"])
    df_cat["series_id"] = (
        df_cat["机房"].astype(str)
        + "|"
        + df_cat["设备"].astype(str)
        + "|"
        + df_cat["组别"].astype(str)
        + "|"
        + df_cat["监控量"].astype(str)
        + "|"
        + df_cat["__file"].astype(str)
        + "|"
        + df_cat["__sheet"].astype(str)
        + "|"
        + df_cat["测点编码"].astype(str)
    )
    v_raw, v_aligned, ids, meta_small = build_series(df_cat, cfg)
    if len(ids) == 0:
        return pd.DataFrame(), pd.DataFrame()
    max_shift = max(1, int(cfg.resample_points * cfg.max_shift_frac))
    v_aligned, shifts = align_series(v_raw, max_shift=max_shift)
    X, feature_names = feature_matrix(v_raw, v_aligned, shifts)
    keep = np.isfinite(X).all(axis=1)
    if not np.all(keep):
        X = X[keep]
        v_raw = v_raw[keep]
        v_aligned = v_aligned[keep]
        shifts = shifts[keep]
        ids = [ids[i] for i in range(len(ids)) if keep[i]]
    if len(ids) == 0:
        return pd.DataFrame(), pd.DataFrame()
    meta_small = meta_small[meta_small["series_id"].isin(ids)]
    X = RobustScaler().fit_transform(X)
    X = PCA(n_components=min(5, X.shape[1])).fit_transform(X)
    eps, labels, noise_ratio, n_clusters = dbscan_auto(X, min_samples=cfg.dbscan_min_samples)
    knn_score, cluster_dist = score_outliers(X, labels, k_neighbors=8)
    anomaly, is_far, is_knn_high = detect_anomalies(
        X,
        labels,
        knn_score,
        cluster_dist,
        far_quantile=cfg.far_quantile,
        knn_quantile=cfg.knn_quantile,
    )
    meta = df_cat.drop_duplicates(subset=["series_id"]).copy()
    meta = meta[meta["series_id"].isin(ids)]
    meta = meta.merge(meta_small, on="series_id", how="left")
    meta = meta.set_index("series_id").loc[ids].reset_index()
    out = meta.copy()
    out["label"] = labels
    out["anomaly_original"] = anomaly
    out["is_far"] = is_far
    out["is_knn_high"] = is_knn_high
    out["knn_score"] = knn_score
    out["cluster_dist"] = cluster_dist
    out["best_shift"] = shifts
    idx = {n: i for i, n in enumerate(feature_names)}
    for n in feature_names:
        out[n] = X[:, idx[n]]
    out["dbscan_eps"] = eps
    out["dbscan_noise_ratio"] = noise_ratio
    out["dbscan_clusters"] = n_clusters
    out = out.reset_index(drop=True)
    v_raw_by_id = {ids[i]: v_raw[i] for i in range(len(ids))}
    v_aligned_by_id = {ids[i]: v_aligned[i] for i in range(len(ids))}
    hard_anomaly = np.zeros(len(out), dtype=bool)
    if cfg.hard_max_voltage is not None and cfg.hard_min_voltage is not None:
        hard_anomaly = (out["v_max"] > cfg.hard_max_voltage) & (out["v_min"] < cfg.hard_min_voltage)
    similarity = np.full(len(out), np.nan, dtype=float)
    if cfg.similarity_threshold > 0 and cfg.similarity_min_peers > 1:
        index_by_id = {sid: i for i, sid in enumerate(out["series_id"])}
        for _, group in out.groupby(["机房", "设备", "组别"], dropna=False):
            if len(group) < cfg.similarity_min_peers:
                continue
            group_ids = group["series_id"].tolist()
            curves = np.vstack([v_aligned_by_id[sid] for sid in group_ids])
            group_med = np.median(curves, axis=0)
            group_std = float(np.std(group_med))
            for sid in group_ids:
                idx = index_by_id[sid]
                series = v_aligned_by_id[sid]
                if group_std <= 1e-9 or float(np.std(series)) <= 1e-9:
                    continue
                similarity[idx] = float(np.corrcoef(series, group_med)[0, 1])
    similar_ok = similarity >= cfg.similarity_threshold
    anomaly_final = hard_anomaly | (out["anomaly_original"].to_numpy() & ~similar_ok)
    out["hard_anomaly"] = hard_anomaly
    out["similarity_group"] = similarity
    out["similarity_keep"] = similar_ok
    out["anomaly"] = anomaly_final
    out = out.sort_values(["anomaly", "knn_score"], ascending=[False, False]).reset_index(drop=True)
    plot_base = out_dir / "plots" / cfg.name
    anomalies = out[out["anomaly"]]
    plot_n = len(anomalies) if cfg.max_plots < 0 else int(cfg.max_plots)
    plotted = min(len(anomalies), max(plot_n, 0))
    for _, row in anomalies.head(plotted).iterrows():
        plot_overlay(
            out_dir=plot_base,
            category_name=cfg.name,
            anomaly_row=row,
            v_raw_by_id=v_raw_by_id,
            v_aligned_by_id=v_aligned_by_id,
            meta=out,
            ids=ids,
        )
    print(f"{cfg.name}：已绘制异常图 {plotted}/{len(anomalies)} 到 {plot_base}")
    group_summary = (
        out.groupby(["__file", "__sheet", "机房", "设备"], dropna=False)
        .agg(
            total_series=("监控量", "nunique"),
            anomaly_series=("anomaly", "sum"),
            max_knn=("knn_score", "max"),
        )
        .reset_index()
    )
    group_summary["anomaly_ratio"] = group_summary["anomaly_series"] / group_summary["total_series"]
    group_summary = group_summary.sort_values(["anomaly_ratio", "max_knn"], ascending=[False, False])
    return out, group_summary


def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "监控值" not in df.columns and "上报值" in df.columns:
        df["监控值"] = df["上报值"]
    if "__sheet" not in df.columns:
        df["__sheet"] = ""
    if "组别" not in df.columns:
        grp = df["监控量"].astype(str).str.extract(r"(第\d+组)")
        df["组别"] = grp[0].fillna("未知组")
    if "测点编码" not in df.columns:
        df["测点编码"] = df["__sheet"].astype(str)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="阶段一：对电池放电曲线做异常检测，并为异常样本生成对比图。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, default="/mnt/nvme/projects/battery/开关电源")
    parser.add_argument("--output-dir", type=str, default="/mnt/nvme/projects/battery/output")
    parser.add_argument("--resample-points", type=int, default=60)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--min-window-points", type=int, default=6)
    parser.add_argument("--max-shift-frac", type=float, default=0.12)
    parser.add_argument("--dbscan-min-samples", type=int, default=6)
    parser.add_argument("--far-quantile", type=float, default=0.99)
    parser.add_argument("--knn-quantile", type=float, default=0.99)
    parser.add_argument(
        "--max-plots",
        type=int,
        default=-1,
        help="每个类别最多生成多少张异常图；小于 0 表示对全部异常样本绘图。",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all = load_voltage_rows_any(input_dir)
    if df_all.empty:
        print(f"未找到可用数据：{input_dir}")
        return 2
    df_all = normalize_columns(df_all)
    df_all = ensure_columns(
        df_all,
        ["机房", "设备", "测点类型", "监控量", "上报时间", "监控值"],
    )
    cfg_single = CategoryConfig(
        name="单体XX电压",
        point_type_pattern="单体",
        min_points=args.min_points,
        resample_points=args.resample_points,
        max_shift_frac=args.max_shift_frac,
        dbscan_min_samples=args.dbscan_min_samples,
        far_quantile=args.far_quantile,
        knn_quantile=args.knn_quantile,
        max_plots=args.max_plots,
        min_window_points=args.min_window_points,
        hard_max_voltage=2.6,
        hard_min_voltage=1.7,
        similarity_threshold=0.92,
        similarity_min_peers=3,
    )
    cfg_total = CategoryConfig(
        name="电池组总电压",
        point_type_pattern="总电压",
        min_points=max(args.min_points, 10),
        resample_points=args.resample_points,
        max_shift_frac=args.max_shift_frac,
        dbscan_min_samples=args.dbscan_min_samples,
        far_quantile=args.far_quantile,
        knn_quantile=args.knn_quantile,
        max_plots=args.max_plots,
        min_window_points=max(args.min_window_points, 10),
        hard_max_voltage=None,
        hard_min_voltage=None,
        similarity_threshold=0.92,
        similarity_min_peers=3,
    )
    single_out, single_group = run_category(df_all, cfg_single, out_dir)
    total_out, total_group = run_category(df_all, cfg_total, out_dir)
    if not single_out.empty:
        single_out.to_csv(out_dir / "anomalies_single_cells.csv", index=False, encoding="utf-8-sig")
        single_group.to_csv(out_dir / "anomalies_single_battery_groups.csv", index=False, encoding="utf-8-sig")
        print(f"单体XX电压：异常单体={int(single_out['anomaly'].sum())}/{len(single_out)}")
        print(f"单体XX电压：结果已输出到 {out_dir}")
    else:
        print("单体XX电压：无可用序列")
    if not total_out.empty:
        total_out.to_csv(out_dir / "anomalies_pack_voltage.csv", index=False, encoding="utf-8-sig")
        total_group.to_csv(out_dir / "anomalies_pack_voltage_groups.csv", index=False, encoding="utf-8-sig")
        print(f"电池组总电压：异常曲线={int(total_out['anomaly'].sum())}/{len(total_out)}")
        print(f"电池组总电压：结果已输出到 {out_dir}")
    else:
        print("电池组总电压：无可用序列")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
