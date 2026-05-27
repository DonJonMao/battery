import argparse
from dataclasses import asdict
import os
from pathlib import Path
from typing import Iterable
import warnings
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-battery")
warnings.filterwarnings("ignore", message="Glyph .* missing from font", category=UserWarning)

from classify_battery_plot_shapes import CLASS_LABELS, ShapeMetrics, classify_shape
from cluster_voltage_anomalies import align_series, find_discharge_window, normalize_columns, ensure_columns


NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PLOT_MAX_SAMPLE = 60


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


def build_series_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["机房"].astype(str)
        + "|"
        + df["设备"].astype(str)
        + "|"
        + df["组别"].astype(str)
        + "|"
        + df["监控量"].astype(str)
        + "|"
        + df["__file"].astype(str)
        + "|"
        + df["__sheet"].astype(str)
        + "|"
        + df["测点编码"].astype(str)
    )


def to_seconds_safe(t: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(t):
        return (t - t.min()).dt.total_seconds().to_numpy()
    t_parsed = pd.to_datetime(t, errors="coerce")
    if t_parsed.notna().any():
        return (t_parsed - t_parsed.min()).dt.total_seconds().to_numpy()
    return pd.to_numeric(t, errors="coerce").to_numpy()


def resample_curve(g: pd.DataFrame, resample_points: int, min_points: int, min_window_points: int) -> tuple[np.ndarray, dict[str, object]] | None:
    g = g.sort_values("上报时间")
    g = g.copy()
    g["监控值"] = pd.to_numeric(g["监控值"], errors="coerce")
    g = g.groupby("上报时间", dropna=False, as_index=False).agg({"监控值": "mean"})
    if len(g) < min_points:
        return None
    t = to_seconds_safe(g["上报时间"])
    v = pd.to_numeric(g["监控值"], errors="coerce").to_numpy()
    mask = (~np.isnan(v)) & (~np.isnan(t))
    t = t[mask]
    v = v[mask]
    if len(v) < min_points:
        return None
    window = find_discharge_window(v, min_window_points)
    window_found = window is not None
    if window is None:
        s, e = 0, len(v) - 1
    else:
        s, e = window
    t = t[s : e + 1]
    v = v[s : e + 1]
    if len(v) < min_points or t[-1] == t[0]:
        return None
    t_norm = (t - t[0]) / (t[-1] - t[0])
    t_new = np.linspace(0, 1, resample_points)
    v_new = np.interp(t_new, t_norm, v)
    meta = {
        "n_points": int(len(v)),
        "duration_s": float(t[-1] - t[0]),
        "window_found": bool(window_found),
        "v_min": float(np.min(v)),
        "v_max": float(np.max(v)),
        "v_range": float(np.max(v) - np.min(v)),
    }
    return v_new, meta


def build_all_single_cell_series(
    df_all: pd.DataFrame,
    resample_points: int,
    min_points: int,
    min_window_points: int,
    max_shift_frac: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    df_cat = df_all[df_all["测点类型"].astype(str).str.contains("单体", na=False)].copy()
    df_cat = df_cat.dropna(subset=["监控量", "上报时间", "监控值"])
    df_cat["series_id"] = build_series_id(df_cat)

    ids: list[str] = []
    raw_curves: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []
    for series_id, g in df_cat.groupby("series_id", dropna=False):
        result = resample_curve(g, resample_points=resample_points, min_points=min_points, min_window_points=min_window_points)
        if result is None:
            continue
        curve, meta = result
        ids.append(series_id)
        raw_curves.append(curve)
        meta_rows.append({"series_id": series_id, **meta})

    if not ids:
        return pd.DataFrame(), {}, {}

    v_raw = np.vstack(raw_curves)
    max_shift = max(1, int(resample_points * max_shift_frac))
    v_aligned, shifts = align_series(v_raw, max_shift=max_shift)

    meta = df_cat.drop_duplicates(subset=["series_id"]).copy()
    meta = meta[meta["series_id"].isin(ids)]
    meta = meta.merge(pd.DataFrame(meta_rows), on="series_id", how="left")
    meta = meta.set_index("series_id").loc[ids].reset_index()
    meta["best_shift"] = shifts

    v_raw_by_id = {ids[i]: v_raw[i] for i in range(len(ids))}
    v_aligned_by_id = {ids[i]: v_aligned[i] for i in range(len(ids))}
    return meta, v_raw_by_id, v_aligned_by_id


def normalize_curve(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    lo = float(np.min(v))
    hi = float(np.max(v))
    if hi - lo <= 1e-9:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def smooth_curve(v: np.ndarray) -> np.ndarray:
    if len(v) < 5:
        return v
    win = max(5, min(21, (len(v) // 20) * 2 + 1))
    kernel = np.ones(win, dtype=float) / win
    pad = win // 2
    padded = np.pad(v, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def compute_shape_metrics(v_raw: np.ndarray) -> ShapeMetrics:
    v = smooth_curve(normalize_curve(v_raw))
    n = len(v)
    edge = max(5, n // 12)
    tail = max(5, n // 20)
    valley_idx = int(np.argmin(v))
    start_level = float(np.median(v[:edge]))
    end_level = float(np.median(v[-edge:]))
    valley_level = float(v[valley_idx])
    drop = start_level - valley_level
    recover = end_level - valley_level
    pre = v[: valley_idx + 1]
    post = v[valley_idx:]
    pre_down_ratio = float(np.mean(np.diff(pre) < -0.002)) if len(pre) > 1 else 0.0
    post_up_ratio = float(np.mean(np.diff(post) > 0.002)) if len(post) > 1 else 0.0
    dv = np.diff(v)
    active = dv[np.abs(dv) > 0.002]
    sign = np.sign(active)
    sign_change_ratio = float(np.mean(sign[1:] != sign[:-1])) if len(sign) > 1 else 0.0
    dynamic_range = float(np.quantile(v, 0.9) - np.quantile(v, 0.1))
    end_drop = float(np.max(v[:-tail]) - end_level) if n > tail else 0.0
    recover_ratio = recover / (drop + 1e-9) if drop > 1e-9 else 0.0
    return ShapeMetrics(
        valley_x=valley_idx / n,
        start_level=start_level,
        valley_level=valley_level,
        end_level=end_level,
        drop=drop,
        recover=recover,
        recover_ratio=recover_ratio,
        pre_down_ratio=pre_down_ratio,
        post_up_ratio=post_up_ratio,
        sign_change_ratio=sign_change_ratio,
        dynamic_range=dynamic_range,
        end_drop=end_drop,
    )


def plot_raw_overlay(
    out_dir: Path,
    shape_class: str,
    row: pd.Series,
    v_raw_by_id: dict[str, np.ndarray],
    v_aligned_by_id: dict[str, np.ndarray],
    meta: pd.DataFrame,
) -> Path:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    series_id = row["series_id"]
    same_group = (
        (meta["设备"] == row["设备"])
        & (meta["组别"] == row["组别"])
        & (meta["机房"] == row["机房"])
    )
    peer_ids = meta.loc[same_group, "series_id"].tolist()
    peer_ids = [pid for pid in peer_ids if pid != series_id]
    if len(peer_ids) > PLOT_MAX_SAMPLE:
        peer_ids = list(np.random.default_rng(42).choice(peer_ids, size=PLOT_MAX_SAMPLE, replace=False))

    x = np.linspace(0, 1, len(v_raw_by_id[series_id]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    for pid in peer_ids:
        axes[0].plot(x, v_raw_by_id[pid], color="0.75", linewidth=0.8, alpha=0.6)
        axes[1].plot(x, v_aligned_by_id[pid], color="0.75", linewidth=0.8, alpha=0.6)
    if peer_ids:
        med_raw = np.median(np.vstack([v_raw_by_id[pid] for pid in peer_ids]), axis=0)
        med_aligned = np.median(np.vstack([v_aligned_by_id[pid] for pid in peer_ids]), axis=0)
        axes[0].plot(x, med_raw, color="black", linewidth=2)
        axes[1].plot(x, med_aligned, color="black", linewidth=2)
    axes[0].plot(x, v_raw_by_id[series_id], color="red", linewidth=2)
    axes[1].plot(x, v_aligned_by_id[series_id], color="red", linewidth=2)
    axes[0].set_title("Raw Voltage")
    axes[1].set_title("Aligned Shape")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    title = f"单体XX电压__{row['__file']}__{row['__sheet']}__{row['机房']}__{row['设备']}__{row['组别']}__{row['监控量']}"
    fig.suptitle(f"class={shape_class} | see filename for series metadata", fontsize=9)
    fig.tight_layout()
    path = out_dir / f"{title}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def load_anomaly_targets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "anomaly" not in df.columns:
        raise ValueError(f"异常清单缺少 anomaly 列: {path}")
    anomaly_mask = df["anomaly"].astype(str).str.lower() == "true"
    df = df[anomaly_mask].copy()
    return df.reset_index(drop=True)


def summarize_counts(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["shape_class", "shape_class_cn"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("shape_class")
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="备选阶段二：直接读取原始单体曲线，对异常清单做形状分类并输出对比图。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, default="/mnt/nvme/projects/battery/UPS/型号1")
    parser.add_argument(
        "--anomaly-csv",
        type=str,
        default="/mnt/nvme/projects/battery/output_ups_型号1/anomalies_single_cells.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/nvme/projects/battery/output_ups_型号1_原始三分类/单体XX电压",
    )
    parser.add_argument("--resample-points", type=int, default=60)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--min-window-points", type=int, default=6)
    parser.add_argument("--max-shift-frac", type=float, default=0.12)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_voltage_rows_xml(input_dir)
    if df_all.empty:
        print(f"未找到可用原始数据: {input_dir}")
        return 2
    df_all = normalize_columns(df_all)
    df_all = ensure_columns(df_all, ["机房", "设备", "测点类型", "监控量", "上报时间", "监控值"])

    meta, v_raw_by_id, v_aligned_by_id = build_all_single_cell_series(
        df_all,
        resample_points=args.resample_points,
        min_points=args.min_points,
        min_window_points=args.min_window_points,
        max_shift_frac=args.max_shift_frac,
    )
    if meta.empty:
        print("未构建出可用单体曲线")
        return 2

    anomaly_df = load_anomaly_targets(Path(args.anomaly_csv))
    available_ids = set(meta["series_id"])
    anomaly_df = anomaly_df[anomaly_df["series_id"].isin(available_ids)].copy()
    if anomaly_df.empty:
        print(f"异常清单与原始曲线无交集: {args.anomaly_csv}")
        return 2

    records = []
    for _, row in anomaly_df.iterrows():
        series_id = row["series_id"]
        raw_curve = v_raw_by_id[series_id]
        metrics = compute_shape_metrics(raw_curve)
        shape_class, reason = classify_shape(metrics)
        class_dir = output_dir / CLASS_LABELS[shape_class]
        meta_row = meta.loc[meta["series_id"] == series_id].iloc[0]
        plot_path = plot_raw_overlay(
            out_dir=class_dir,
            shape_class=shape_class,
            row=meta_row,
            v_raw_by_id=v_raw_by_id,
            v_aligned_by_id=v_aligned_by_id,
            meta=meta,
        )
        record = meta_row.to_dict()
        record.update(asdict(metrics))
        record["shape_class"] = shape_class
        record["shape_class_cn"] = CLASS_LABELS[shape_class]
        record["reason"] = reason
        record["generated_plot"] = str(plot_path)
        records.append(record)

    out_df = pd.DataFrame(records)
    out_df["shape_class"] = pd.Categorical(
        out_df["shape_class"],
        categories=["discharge_like", "invalid_or_bad_collection", "uncertain"],
        ordered=True,
    )
    out_df = out_df.sort_values(["shape_class", "__file", "设备", "监控量"]).reset_index(drop=True)

    csv_path = output_dir / "raw_shape_classification.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = summarize_counts(out_df)
    summary_path = output_dir / "raw_shape_classification_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for _, row in summary.iterrows():
        print(f"{row['shape_class_cn']}: {int(row['count'])}")
    print(f"分类明细: {csv_path}")
    print(f"汇总: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
