import argparse
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


RED_R_MIN = 180
RED_G_MAX = 120
RED_B_MAX = 120
MIN_SEGMENT_WIDTH = 50

CLASS_LABELS = {
    "discharge_like": "放电型异常",
    "invalid_or_bad_collection": "疑似操作或采集异常",
    "uncertain": "不确定",
}


@dataclass(frozen=True)
class ShapeMetrics:
    valley_x: float
    start_level: float
    valley_level: float
    end_level: float
    drop: float
    recover: float
    recover_ratio: float
    pre_down_ratio: float
    post_up_ratio: float
    sign_change_ratio: float
    dynamic_range: float
    end_drop: float


def red_mask(image: np.ndarray) -> np.ndarray:
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    return (r >= RED_R_MIN) & (g <= RED_G_MAX) & (b <= RED_B_MAX) & ((r - np.maximum(g, b)) >= 60)


def find_panel_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    xs = np.where(mask)[1]
    if xs.size == 0:
        return []
    counts = np.bincount(xs, minlength=mask.shape[1])
    segments: list[tuple[int, int]] = []
    in_segment = False
    start = 0
    for idx, count in enumerate(counts):
        if count > 0 and not in_segment:
            start = idx
            in_segment = True
            continue
        if count == 0 and in_segment:
            if idx - start > MIN_SEGMENT_WIDTH:
                segments.append((start, idx - 1))
            in_segment = False
    if in_segment and mask.shape[1] - start > MIN_SEGMENT_WIDTH:
        segments.append((start, mask.shape[1] - 1))
    return segments


def extract_curve_from_plot(path: Path) -> np.ndarray:
    image = np.array(Image.open(path).convert("RGB"))
    mask = red_mask(image)
    segments = find_panel_segments(mask)
    if len(segments) < 1:
        raise ValueError(f"未找到红色曲线面板: {path}")
    left_x0, left_x1 = segments[0]
    curve_y = []
    for x in range(left_x0, left_x1 + 1):
        ys = np.where(mask[:, x])[0]
        curve_y.append(float(np.median(ys)) if len(ys) else np.nan)
    y = np.array(curve_y, dtype=float)
    valid_idx = np.where(~np.isnan(y))[0]
    if len(valid_idx) < 2:
        raise ValueError(f"红色曲线像素不足: {path}")
    y = np.interp(np.arange(len(y)), valid_idx, y[valid_idx])
    voltage_like = -y
    voltage_like = (voltage_like - voltage_like.min()) / (voltage_like.max() - voltage_like.min() + 1e-9)
    return voltage_like


def smooth_curve(v: np.ndarray) -> np.ndarray:
    if len(v) < 5:
        return v
    win = max(5, min(21, (len(v) // 20) * 2 + 1))
    kernel = np.ones(win, dtype=float) / win
    pad = win // 2
    padded = np.pad(v, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def compute_metrics(v: np.ndarray) -> ShapeMetrics:
    v_smooth = smooth_curve(v)
    n = len(v_smooth)
    edge = max(5, n // 12)
    tail = max(5, n // 20)
    valley_idx = int(np.argmin(v_smooth))
    start_level = float(np.median(v_smooth[:edge]))
    end_level = float(np.median(v_smooth[-edge:]))
    valley_level = float(v_smooth[valley_idx])
    drop = start_level - valley_level
    recover = end_level - valley_level
    pre = v_smooth[: valley_idx + 1]
    post = v_smooth[valley_idx:]
    pre_down_ratio = float(np.mean(np.diff(pre) < -0.002)) if len(pre) > 1 else 0.0
    post_up_ratio = float(np.mean(np.diff(post) > 0.002)) if len(post) > 1 else 0.0
    dv = np.diff(v_smooth)
    active = dv[np.abs(dv) > 0.002]
    sign = np.sign(active)
    sign_change_ratio = float(np.mean(sign[1:] != sign[:-1])) if len(sign) > 1 else 0.0
    dynamic_range = float(np.quantile(v_smooth, 0.9) - np.quantile(v_smooth, 0.1))
    end_drop = float(np.max(v_smooth[:-tail]) - end_level) if n > tail else 0.0
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


def classify_shape(m: ShapeMetrics) -> tuple[str, str]:
    if m.dynamic_range < 0.12:
        return "invalid_or_bad_collection", "整体动态范围过小，更像平线或轻微波动"
    if m.valley_x >= 0.92:
        return "invalid_or_bad_collection", "最低点贴近尾部，缺少谷底后的回升段"
    if m.valley_x <= 0.03:
        return "invalid_or_bad_collection", "最低点贴近起点，像截断或采集窗口异常"
    if m.end_drop >= 0.72:
        return "invalid_or_bad_collection", "末端突降明显，像结束阶段误操作或截断"

    valley_inside = 0.07 <= m.valley_x <= 0.88
    has_drop = m.drop >= 0.18
    has_recover = m.recover >= 0.18
    recovery_visible = (m.post_up_ratio >= 0.09) or (m.recover_ratio >= 0.55)
    stable_shape = m.sign_change_ratio <= 0.08
    not_tail_collapse = m.end_drop <= 0.55
    if valley_inside and has_drop and has_recover and recovery_visible and stable_shape and not_tail_collapse:
        return "discharge_like", "存在内部谷底，且谷底后有持续回升，形态接近放电曲线"

    if m.valley_x >= 0.78 and m.post_up_ratio < 0.05:
        return "invalid_or_bad_collection", "谷底过晚且几乎没有回升，更像无效放电或采集异常"
    if m.sign_change_ratio >= 0.07 and m.dynamic_range < 0.25:
        return "invalid_or_bad_collection", "锯齿波动偏强且整体幅度不大，更像噪声或采样问题"

    return "uncertain", "形状介于两类之间，建议人工复核"


def parse_plot_name(path: Path) -> dict[str, str]:
    parts = path.stem.split("__")
    result = {
        "plot_name": path.name,
        "source_file": "",
        "source_sheet": "",
        "room": "",
        "device": "",
        "group_name": "",
        "measure_name": "",
    }
    if len(parts) >= 7:
        result.update(
            {
                "source_file": parts[1],
                "source_sheet": parts[2],
                "room": parts[3],
                "device": parts[4],
                "group_name": parts[5],
                "measure_name": parts[6],
            }
        )
    return result


def classify_plot(path: Path) -> dict[str, object]:
    curve = extract_curve_from_plot(path)
    metrics = compute_metrics(curve)
    shape_class, reason = classify_shape(metrics)
    row = parse_plot_name(path)
    row.update(asdict(metrics))
    row["shape_class"] = shape_class
    row["shape_class_cn"] = CLASS_LABELS[shape_class]
    row["reason"] = reason
    row["src_path"] = str(path)
    return row


def copy_to_bucket(src: Path, out_root: Path, shape_class: str) -> Path:
    bucket = out_root / CLASS_LABELS[shape_class]
    bucket.mkdir(parents=True, exist_ok=True)
    dst = bucket / src.name
    shutil.copy2(src, dst)
    return dst


def build_dataframe(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.glob("*.png")):
        try:
            rows.append(classify_plot(path))
        except Exception as exc:
            rows.append(
                {
                    "plot_name": path.name,
                    "src_path": str(path),
                    "shape_class": "uncertain",
                    "shape_class_cn": CLASS_LABELS["uncertain"],
                    "reason": f"处理失败: {exc}",
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "shape_class" in df.columns:
        df["shape_class"] = pd.Categorical(
            df["shape_class"],
            categories=["discharge_like", "invalid_or_bad_collection", "uncertain"],
            ordered=True,
        )
        df = df.sort_values(["shape_class", "source_file", "device", "measure_name"], na_position="last").reset_index(drop=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="阶段二：读取阶段一生成的异常曲线 PNG，并按曲线形状做三分类。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/nvme/projects/battery/output_ups_型号1/plots/单体XX电压",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/nvme/projects/battery/output_ups_型号1/plots_三分类/单体XX电压",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="shape_classification.csv",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(input_dir)
    if df.empty:
        print(f"未找到可分类图片: {input_dir}")
        return 2

    copied_paths = []
    for _, row in df.iterrows():
        src = Path(row["src_path"])
        dst = copy_to_bucket(src, output_dir, str(row["shape_class"]))
        copied_paths.append(str(dst))
    df["copied_path"] = copied_paths

    csv_path = output_dir / args.csv_name
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby(["shape_class", "shape_class_cn"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("shape_class")
    )
    summary_path = output_dir / "shape_class_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for _, row in summary.iterrows():
        print(f"{row['shape_class_cn']}: {int(row['count'])}")
    print(f"分类明细: {csv_path}")
    print(f"汇总: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
