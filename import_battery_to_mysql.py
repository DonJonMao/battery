#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 battery 项目的原始 Excel 数据与现有分析结果导入本地 MySQL。

导入内容包括：
1. 原始序列元数据：battery_series
2. 原始测点数据：battery_measurement
3. 阶段一异常检测结果：battery_anomaly_result
4. 阶段二形状分类结果：battery_shape_classification

说明：
- 不依赖 Docker，直接连接本机 MySQL。
- 默认幂等导入；重复执行会按唯一键做 upsert。
- 阶段一聚合类 CSV 不单独入表，通过视图 battery_anomaly_group_summary_v 复现。
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd
import pymysql


NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
RAW_REQUIRED_COLUMNS = ["机房", "设备", "测点类型", "监控量", "上报时间", "监控值"]
SHAPE_CLASS_LABELS = {
    "discharge_like": "放电型异常",
    "invalid_or_bad_collection": "疑似操作或采集异常",
    "uncertain": "不确定",
}


def text_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def to_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    except Exception:
        return None
    if pd.isna(number):
        return None
    return float(number)


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def to_bool(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y"}:
        return 1
    if text in {"0", "false", "f", "no", "n"}:
        return 0
    return None


def to_datetime(value: Any):
    if pd.isna(value):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def sha1_text(*parts: Any) -> str:
    joined = "||".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


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
    values: list[str] = []
    for si in root.findall(f"{NS_MAIN}si"):
        parts = []
        for t in si.iter(f"{NS_MAIN}t"):
            parts.append(t.text or "")
        values.append("".join(parts))
    return values


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

    rows: list[list[str]] = []
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

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = [str(value).strip() for value in rows[0]]
    data = rows[1:]
    if not header:
        header = [f"col_{idx}" for idx in range(width)]
    return pd.DataFrame(data, columns=header)


def list_sheet_targets(zf: ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    sheets = workbook.find(f"{NS_MAIN}sheets")
    if sheets is None:
        return []
    targets: list[tuple[str, str]] = []
    for sheet in sheets:
        name = sheet.attrib.get("name", "")
        rel_id = sheet.attrib.get(f"{NS_REL}id", "")
        target = rel_map.get(rel_id, "")
        if name and target:
            targets.append((name, target))
    return targets


def load_voltage_rows(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.xlsx"))
    rows = []
    for file in files:
        try:
            xls = pd.ExcelFile(file)
        except Exception:
            continue
        sheet_names = [name for name in xls.sheet_names if name.upper().startswith("DC")]
        if not sheet_names:
            sheet_names = [xls.sheet_names[0]]
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
            except Exception:
                continue
            df = df.copy()
            df["__file"] = file.name
            df["__sheet"] = sheet_name
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).rename(columns=str.strip)


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
    return pd.concat(rows, ignore_index=True).rename(columns=str.strip)


def load_voltage_rows_any(input_dir: Path) -> pd.DataFrame:
    df = load_voltage_rows(input_dir)
    if not df.empty:
        return df
    return load_voltage_rows_xml(input_dir)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "监控值" not in normalized.columns and "上报值" in normalized.columns:
        normalized["监控值"] = normalized["上报值"]
    if "__sheet" not in normalized.columns:
        normalized["__sheet"] = ""
    if "组别" not in normalized.columns:
        grp = normalized["监控量"].astype(str).str.extract(r"(第\d+组)")
        normalized["组别"] = grp[0].fillna("未知组")
    if "测点编码" not in normalized.columns:
        normalized["测点编码"] = normalized["__sheet"].astype(str)
    return normalized


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    return df


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


def project_relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def dataset_meta_from_dir(project_root: Path, dataset_dir: Path) -> tuple[str, str, str | None]:
    raw_dataset = project_relative(dataset_dir, project_root)
    parts = Path(raw_dataset).parts
    source_category = parts[0] if parts else raw_dataset
    source_subtype = "/".join(parts[1:]) if len(parts) > 1 else None
    return raw_dataset, source_category, source_subtype


def normalize_city(city: str | None) -> str | None:
    if not city:
        return None
    normalized = city.strip()
    if normalized.endswith("市"):
        normalized = normalized[:-1]
    return normalized or None


def build_unique_lookup(rows: list[tuple[Any, Any]]) -> dict[Any, Any | None]:
    mapping: dict[Any, Any | None] = {}
    for key, value in rows:
        if key not in mapping:
            mapping[key] = value
        elif mapping[key] != value:
            mapping[key] = None
    return mapping


def load_machine_room_lookup(conn) -> tuple[dict[tuple[str, str], int | None], dict[str, int | None]]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT id, room_name, city FROM machine_room")
        rows = cursor.fetchall()

    room_city_pairs = []
    room_only_pairs = []
    for row in rows:
        room_name = text_or_none(row.get("room_name"))
        city = normalize_city(text_or_none(row.get("city")))
        room_id = row.get("id")
        if room_name:
            room_only_pairs.append((room_name, room_id))
            if city:
                room_city_pairs.append(((room_name, city), room_id))
    return build_unique_lookup(room_city_pairs), build_unique_lookup(room_only_pairs)


def match_machine_room_id(
    room_name: str | None,
    city: str | None,
    room_city_lookup: dict[tuple[str, str], int | None],
    room_lookup: dict[str, int | None],
) -> int | None:
    if not room_name:
        return None
    city_norm = normalize_city(city)
    if city_norm:
        matched = room_city_lookup.get((room_name, city_norm))
        if matched:
            return matched
    matched = room_lookup.get(room_name)
    return matched or None


def discover_raw_dataset_dirs(project_root: Path) -> list[Path]:
    dataset_dirs = set()
    for root_name in ("UPS", "开关电源"):
        root_dir = project_root / root_name
        if not root_dir.exists():
            continue
        for file_path in root_dir.rglob("*.xlsx"):
            dataset_dirs.add(file_path.parent)
    return sorted(dataset_dirs)


def discover_files(project_root: Path, filename: str) -> list[Path]:
    return sorted(project_root.rglob(filename))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def upsert_rows(conn, table: str, rows: list[dict[str, Any]], chunk_size: int = 1000) -> int:
    if not rows:
        return 0

    columns = list(rows[0].keys())
    placeholders = ",".join(["%s"] * len(columns))
    update_columns = [column for column in columns if column not in {"id", "created_at", "imported_at"}]
    sql = (
        f"INSERT INTO `{table}` ({','.join(f'`{column}`' for column in columns)}) "
        f"VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {','.join(f'`{column}`=VALUES(`{column}`)' for column in update_columns)}"
    )

    total = 0
    for start in range(0, len(rows), chunk_size):
        batch = rows[start : start + chunk_size]
        values = [tuple(row.get(column) for column in columns) for row in batch]
        with conn.cursor() as cursor:
            cursor.executemany(sql, values)
        conn.commit()
        total += len(batch)
    return total


def execute_sql_file(conn, sql_path: Path) -> None:
    statement_lines: list[str] = []
    with sql_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue
            statement_lines.append(line)
            if stripped.endswith(";"):
                statement = "".join(statement_lines).strip()
                if statement:
                    with conn.cursor() as cursor:
                        cursor.execute(statement)
                statement_lines = []
    conn.commit()


def build_series_rows(
    df: pd.DataFrame,
    raw_dataset: str,
    source_category: str,
    source_subtype: str | None,
    room_city_lookup: dict[tuple[str, str], int | None],
    room_lookup: dict[str, int | None],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[tuple[str, str, str, str, str, str], str | None]]:
    working = df.copy()
    working["report_time_dt"] = pd.to_datetime(working["上报时间"], errors="coerce")
    working["report_value_num"] = pd.to_numeric(working["上报值"], errors="coerce") if "上报值" in working.columns else pd.NA
    working["monitor_value_num"] = pd.to_numeric(working["监控值"], errors="coerce")
    working["series_id"] = build_series_id(working)

    series_rows: list[dict[str, Any]] = []
    series_info_by_id: dict[str, dict[str, Any]] = {}
    resolver_pairs: list[tuple[tuple[str, str, str, str, str, str], str]] = []

    sort_columns = ["series_id", "report_time_dt", "上报时间"]
    working = working.sort_values(sort_columns, na_position="last")
    for series_id, group in working.groupby("series_id", dropna=False):
        first_row = group.iloc[0]
        latest_row = group.iloc[-1]
        room_name = text_or_none(first_row.get("机房"))
        city = text_or_none(first_row.get("市"))
        machine_room_id = match_machine_room_id(room_name, city, room_city_lookup, room_lookup)

        valid_times = group["report_time_dt"].dropna()
        first_report_time = valid_times.min().to_pydatetime() if not valid_times.empty else None
        last_report_time = valid_times.max().to_pydatetime() if not valid_times.empty else None

        row = {
            "raw_dataset": raw_dataset,
            "source_category": source_category,
            "source_subtype": source_subtype,
            "series_id": series_id,
            "machine_room_id": machine_room_id,
            "province": text_or_none(first_row.get("省")),
            "city": city,
            "site_name": text_or_none(first_row.get("站点")),
            "building_name": text_or_none(first_row.get("楼栋")),
            "room_name": room_name,
            "device_type": text_or_none(first_row.get("设备类型")),
            "device_name": text_or_none(first_row.get("设备")),
            "point_type": text_or_none(first_row.get("测点类型")),
            "measure_name": text_or_none(first_row.get("监控量")),
            "group_name": text_or_none(first_row.get("组别")),
            "point_code": text_or_none(first_row.get("测点编码")),
            "unit": text_or_none(first_row.get("单位")),
            "source_file": text_or_none(first_row.get("__file")),
            "source_sheet": text_or_none(first_row.get("__sheet")),
            "first_report_time": first_report_time,
            "last_report_time": last_report_time,
            "sample_count": int(len(group)),
            "latest_report_value": to_float(latest_row.get("上报值")),
            "latest_monitor_value": to_float(latest_row.get("监控值")),
        }
        series_rows.append(row)
        series_info_by_id[series_id] = row

        resolver_key = (
            row["source_file"] or "",
            row["source_sheet"] or "",
            row["room_name"] or "",
            row["device_name"] or "",
            row["group_name"] or "",
            row["measure_name"] or "",
        )
        resolver_pairs.append((resolver_key, series_id))

    resolver = build_unique_lookup(resolver_pairs)
    return series_rows, series_info_by_id, resolver


def build_measurement_rows(
    df: pd.DataFrame,
    raw_dataset: str,
    source_category: str,
    source_subtype: str | None,
    series_info_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    working = df.copy()
    working["series_id"] = build_series_id(working)
    rows: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        series_id = row["series_id"]
        series_info = series_info_by_id.get(series_id, {})
        report_time_raw = text_or_none(row.get("上报时间"))
        report_value_raw = text_or_none(row.get("上报值"))
        monitor_value_raw = text_or_none(row.get("监控值"))
        row_hash = sha1_text(
            raw_dataset,
            series_id,
            text_or_none(row.get("__file")),
            text_or_none(row.get("__sheet")),
            report_time_raw,
            report_value_raw,
            monitor_value_raw,
            text_or_none(row.get("单位")),
        )
        rows.append(
            {
                "raw_dataset": raw_dataset,
                "source_category": source_category,
                "source_subtype": source_subtype,
                "series_id": series_id,
                "row_hash": row_hash,
                "machine_room_id": series_info.get("machine_room_id"),
                "source_file": text_or_none(row.get("__file")),
                "source_sheet": text_or_none(row.get("__sheet")),
                "report_time": to_datetime(row.get("上报时间")),
                "report_time_raw": report_time_raw,
                "report_value": to_float(row.get("上报值")),
                "report_value_raw": report_value_raw,
                "monitor_value": to_float(row.get("监控值")),
                "monitor_value_raw": monitor_value_raw,
                "unit": text_or_none(row.get("单位")),
            }
        )
    return rows


def merge_series_meta(base: dict[str, Any], series_info: dict[str, Any] | None) -> dict[str, Any]:
    if not series_info:
        return base
    merged = dict(base)
    for key, value in series_info.items():
        if merged.get(key) in {None, ""} and value not in {None, ""}:
            merged[key] = value
    return merged


def build_anomaly_rows(
    csv_path: Path,
    project_root: Path,
    series_info_by_id: dict[str, dict[str, Any]],
    room_city_lookup: dict[tuple[str, str], int | None],
    room_lookup: dict[str, int | None],
) -> list[dict[str, Any]]:
    df = read_csv(csv_path)
    if df.empty:
        return []
    result_dataset = project_relative(csv_path.parent, project_root)
    result_kind = "single_cell" if csv_path.name == "anomalies_single_cells.csv" else "pack_voltage"

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        series_id = text_or_none(row.get("series_id"))
        series_info = series_info_by_id.get(series_id or "")
        room_name = text_or_none(row.get("机房"))
        city = text_or_none(row.get("市"))
        machine_room_id = match_machine_room_id(room_name, city, room_city_lookup, room_lookup)
        if series_info and series_info.get("machine_room_id"):
            machine_room_id = series_info["machine_room_id"]

        base = {
            "raw_dataset": series_info.get("raw_dataset") if series_info else None,
            "source_category": series_info.get("source_category") if series_info else None,
            "source_subtype": series_info.get("source_subtype") if series_info else None,
            "series_id": series_id,
            "machine_room_id": machine_room_id,
            "province": text_or_none(row.get("省")),
            "city": city,
            "site_name": text_or_none(row.get("站点")),
            "building_name": text_or_none(row.get("楼栋")),
            "room_name": room_name,
            "device_type": text_or_none(row.get("设备类型")),
            "device_name": text_or_none(row.get("设备")),
            "point_type": text_or_none(row.get("测点类型")),
            "measure_name": text_or_none(row.get("监控量")),
            "group_name": text_or_none(row.get("组别")),
            "point_code": text_or_none(row.get("测点编码")),
            "source_file": text_or_none(row.get("__file")),
            "source_sheet": text_or_none(row.get("__sheet")),
            "unit": text_or_none(row.get("单位")),
        }
        base = merge_series_meta(base, series_info)

        rows.append(
            {
                "result_dataset": result_dataset,
                "result_kind": result_kind,
                "record_hash": sha1_text(result_dataset, result_kind, series_id),
                "raw_dataset": base.get("raw_dataset"),
                "source_category": base.get("source_category"),
                "source_subtype": base.get("source_subtype"),
                "series_id": base.get("series_id"),
                "machine_room_id": base.get("machine_room_id"),
                "province": base.get("province"),
                "city": base.get("city"),
                "site_name": base.get("site_name"),
                "building_name": base.get("building_name"),
                "room_name": base.get("room_name"),
                "device_type": base.get("device_type"),
                "device_name": base.get("device_name"),
                "point_type": base.get("point_type"),
                "measure_name": base.get("measure_name"),
                "group_name": base.get("group_name"),
                "point_code": base.get("point_code"),
                "source_file": base.get("source_file"),
                "source_sheet": base.get("source_sheet"),
                "report_time": to_datetime(row.get("上报时间")),
                "report_value": to_float(row.get("上报值")),
                "unit": base.get("unit"),
                "monitor_value": to_float(row.get("监控值")),
                "n_points": to_int(row.get("n_points")),
                "duration_s": to_float(row.get("duration_s")),
                "window_found": to_bool(row.get("window_found")),
                "v_min": to_float(row.get("v_min")),
                "v_max": to_float(row.get("v_max")),
                "v_range": to_float(row.get("v_range")),
                "label": to_int(row.get("label")),
                "anomaly_original": to_bool(row.get("anomaly_original")),
                "is_far": to_bool(row.get("is_far")),
                "is_knn_high": to_bool(row.get("is_knn_high")),
                "knn_score": to_float(row.get("knn_score")),
                "cluster_dist": to_float(row.get("cluster_dist")),
                "best_shift": to_int(row.get("best_shift")),
                "v_mean": to_float(row.get("v_mean")),
                "d2_std": to_float(row.get("d2_std")),
                "zigzag": to_float(row.get("zigzag")),
                "shift_abs": to_float(row.get("shift_abs")),
                "dbscan_eps": to_float(row.get("dbscan_eps")),
                "dbscan_noise_ratio": to_float(row.get("dbscan_noise_ratio")),
                "dbscan_clusters": to_int(row.get("dbscan_clusters")),
                "hard_anomaly": to_bool(row.get("hard_anomaly")),
                "similarity_group": to_float(row.get("similarity_group")),
                "similarity_keep": to_bool(row.get("similarity_keep")),
                "anomaly_flag": to_bool(row.get("anomaly")) or 0,
                "source_csv_path": project_relative(csv_path, project_root),
            }
        )
    return rows


def resolve_series_id_for_shape_row(
    row: pd.Series,
    resolver: dict[tuple[str, str, str, str, str, str], str | None],
) -> str | None:
    if text_or_none(row.get("series_id")):
        return text_or_none(row.get("series_id"))
    key = (
        text_or_none(row.get("source_file")) or "",
        text_or_none(row.get("source_sheet")) or "",
        text_or_none(row.get("room")) or "",
        text_or_none(row.get("device")) or "",
        text_or_none(row.get("group_name")) or "",
        text_or_none(row.get("measure_name")) or "",
    )
    return resolver.get(key)


def build_shape_rows(
    csv_path: Path,
    project_root: Path,
    shape_source: str,
    series_info_by_id: dict[str, dict[str, Any]],
    resolver: dict[tuple[str, str, str, str, str, str], str | None],
    room_city_lookup: dict[tuple[str, str], int | None],
    room_lookup: dict[str, int | None],
) -> list[dict[str, Any]]:
    df = read_csv(csv_path)
    if df.empty:
        return []

    result_dataset = project_relative(csv_path.parent, project_root)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        series_id = resolve_series_id_for_shape_row(row, resolver)
        series_info = series_info_by_id.get(series_id or "")

        room_name = text_or_none(row.get("机房")) or text_or_none(row.get("room"))
        city = text_or_none(row.get("市")) or (series_info.get("city") if series_info else None)
        machine_room_id = match_machine_room_id(room_name, city, room_city_lookup, room_lookup)
        if series_info and series_info.get("machine_room_id"):
            machine_room_id = series_info["machine_room_id"]

        base = {
            "raw_dataset": series_info.get("raw_dataset") if series_info else None,
            "source_category": series_info.get("source_category") if series_info else None,
            "source_subtype": series_info.get("source_subtype") if series_info else None,
            "series_id": series_id,
            "machine_room_id": machine_room_id,
            "province": text_or_none(row.get("省")),
            "city": city,
            "site_name": text_or_none(row.get("站点")),
            "building_name": text_or_none(row.get("楼栋")),
            "room_name": room_name,
            "device_type": text_or_none(row.get("设备类型")),
            "device_name": text_or_none(row.get("设备")) or text_or_none(row.get("device")),
            "point_type": text_or_none(row.get("测点类型")) or (series_info.get("point_type") if series_info else None),
            "measure_name": text_or_none(row.get("监控量")) or text_or_none(row.get("measure_name")),
            "group_name": text_or_none(row.get("组别")) or text_or_none(row.get("group_name")),
            "point_code": text_or_none(row.get("测点编码")) or (series_info.get("point_code") if series_info else None),
            "source_file": text_or_none(row.get("__file")) or text_or_none(row.get("source_file")),
            "source_sheet": text_or_none(row.get("__sheet")) or text_or_none(row.get("source_sheet")),
            "unit": text_or_none(row.get("单位")) or (series_info.get("unit") if series_info else None),
        }
        base = merge_series_meta(base, series_info)

        record_key = series_id or sha1_text(
            result_dataset,
            shape_source,
            base.get("source_file"),
            base.get("source_sheet"),
            base.get("room_name"),
            base.get("device_name"),
            base.get("group_name"),
            base.get("measure_name"),
        )

        rows.append(
            {
                "result_dataset": result_dataset,
                "shape_source": shape_source,
                "record_hash": sha1_text(result_dataset, shape_source, record_key),
                "raw_dataset": base.get("raw_dataset"),
                "source_category": base.get("source_category"),
                "source_subtype": base.get("source_subtype"),
                "series_id": base.get("series_id"),
                "machine_room_id": base.get("machine_room_id"),
                "province": base.get("province"),
                "city": base.get("city"),
                "site_name": base.get("site_name"),
                "building_name": base.get("building_name"),
                "room_name": base.get("room_name"),
                "device_type": base.get("device_type"),
                "device_name": base.get("device_name"),
                "point_type": base.get("point_type"),
                "measure_name": base.get("measure_name"),
                "group_name": base.get("group_name"),
                "point_code": base.get("point_code"),
                "source_file": base.get("source_file"),
                "source_sheet": base.get("source_sheet"),
                "report_time": to_datetime(row.get("上报时间")),
                "report_value": to_float(row.get("上报值")),
                "unit": base.get("unit"),
                "monitor_value": to_float(row.get("监控值")),
                "n_points": to_int(row.get("n_points")),
                "duration_s": to_float(row.get("duration_s")),
                "window_found": to_bool(row.get("window_found")),
                "v_min": to_float(row.get("v_min")),
                "v_max": to_float(row.get("v_max")),
                "v_range": to_float(row.get("v_range")),
                "best_shift": to_int(row.get("best_shift")),
                "valley_x": to_float(row.get("valley_x")),
                "start_level": to_float(row.get("start_level")),
                "valley_level": to_float(row.get("valley_level")),
                "end_level": to_float(row.get("end_level")),
                "drop_value": to_float(row.get("drop")),
                "recover_value": to_float(row.get("recover")),
                "recover_ratio": to_float(row.get("recover_ratio")),
                "pre_down_ratio": to_float(row.get("pre_down_ratio")),
                "post_up_ratio": to_float(row.get("post_up_ratio")),
                "sign_change_ratio": to_float(row.get("sign_change_ratio")),
                "dynamic_range": to_float(row.get("dynamic_range")),
                "end_drop": to_float(row.get("end_drop")),
                "shape_class": text_or_none(row.get("shape_class")) or "uncertain",
                "shape_class_cn": text_or_none(row.get("shape_class_cn"))
                or SHAPE_CLASS_LABELS.get(text_or_none(row.get("shape_class")) or "uncertain"),
                "reason": text_or_none(row.get("reason")),
                "generated_plot": text_or_none(row.get("generated_plot")),
                "src_path": text_or_none(row.get("src_path")),
                "copied_path": text_or_none(row.get("copied_path")),
                "source_csv_path": project_relative(csv_path, project_root),
            }
        )
    return rows


def connect_mysql(args):
    return pymysql.connect(
        host=args.host,
        user=args.user,
        password=args.password,
        database=args.database,
        port=args.port,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="将 battery 项目数据导入本地 MySQL。")
    parser.add_argument("--project-root", default="/mnt/nvme/projects/battery", help="battery 项目根目录")
    parser.add_argument("--schema-file", default="/mnt/nvme/projects/battery/battery_schema.sql", help="建表 SQL 路径")
    parser.add_argument("--host", default="localhost", help="MySQL 主机")
    parser.add_argument("--user", default="api_user", help="MySQL 用户")
    parser.add_argument("--password", default="", help="MySQL 密码")
    parser.add_argument("--database", default="cloud", help="MySQL 数据库")
    parser.add_argument("--port", type=int, default=3306, help="MySQL 端口")
    parser.add_argument("--init-schema", action="store_true", help="导入前执行建表 SQL")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    schema_file = Path(args.schema_file).resolve()

    conn = connect_mysql(args)
    try:
        if args.init_schema:
            execute_sql_file(conn, schema_file)

        room_city_lookup, room_lookup = load_machine_room_lookup(conn)

        raw_dataset_dirs = discover_raw_dataset_dirs(project_root)
        series_info_by_id: dict[str, dict[str, Any]] = {}
        resolver_pairs: list[tuple[tuple[str, str, str, str, str, str], str]] = []

        total_series = 0
        total_measurements = 0
        for dataset_dir in raw_dataset_dirs:
            raw_dataset, source_category, source_subtype = dataset_meta_from_dir(project_root, dataset_dir)
            df = load_voltage_rows_any(dataset_dir)
            if df.empty:
                continue
            df = normalize_columns(df)
            df = ensure_columns(df, RAW_REQUIRED_COLUMNS)

            series_rows, dataset_series_info, dataset_resolver = build_series_rows(
                df=df,
                raw_dataset=raw_dataset,
                source_category=source_category,
                source_subtype=source_subtype,
                room_city_lookup=room_city_lookup,
                room_lookup=room_lookup,
            )
            measurement_rows = build_measurement_rows(
                df=df,
                raw_dataset=raw_dataset,
                source_category=source_category,
                source_subtype=source_subtype,
                series_info_by_id=dataset_series_info,
            )

            total_series += upsert_rows(conn, "battery_series", series_rows)
            total_measurements += upsert_rows(conn, "battery_measurement", measurement_rows)
            series_info_by_id.update(dataset_series_info)
            resolver_pairs.extend((key, value) for key, value in dataset_resolver.items() if value)

        resolver = build_unique_lookup(resolver_pairs)

        anomaly_rows: list[dict[str, Any]] = []
        for csv_path in discover_files(project_root, "anomalies_single_cells.csv"):
            anomaly_rows.extend(
                build_anomaly_rows(
                    csv_path=csv_path,
                    project_root=project_root,
                    series_info_by_id=series_info_by_id,
                    room_city_lookup=room_city_lookup,
                    room_lookup=room_lookup,
                )
            )
        for csv_path in discover_files(project_root, "anomalies_pack_voltage.csv"):
            anomaly_rows.extend(
                build_anomaly_rows(
                    csv_path=csv_path,
                    project_root=project_root,
                    series_info_by_id=series_info_by_id,
                    room_city_lookup=room_city_lookup,
                    room_lookup=room_lookup,
                )
            )
        total_anomalies = upsert_rows(conn, "battery_anomaly_result", anomaly_rows)

        shape_rows: list[dict[str, Any]] = []
        for csv_path in discover_files(project_root, "raw_shape_classification.csv"):
            shape_rows.extend(
                build_shape_rows(
                    csv_path=csv_path,
                    project_root=project_root,
                    shape_source="raw_curve",
                    series_info_by_id=series_info_by_id,
                    resolver=resolver,
                    room_city_lookup=room_city_lookup,
                    room_lookup=room_lookup,
                )
            )
        for csv_path in discover_files(project_root, "shape_classification.csv"):
            shape_rows.extend(
                build_shape_rows(
                    csv_path=csv_path,
                    project_root=project_root,
                    shape_source="plot_png",
                    series_info_by_id=series_info_by_id,
                    resolver=resolver,
                    room_city_lookup=room_city_lookup,
                    room_lookup=room_lookup,
                )
            )
        total_shapes = upsert_rows(conn, "battery_shape_classification", shape_rows)

        print(f"原始数据集目录: {len(raw_dataset_dirs)}")
        print(f"battery_series upsert 行数: {total_series}")
        print(f"battery_measurement upsert 行数: {total_measurements}")
        print(f"battery_anomaly_result upsert 行数: {total_anomalies}")
        print(f"battery_shape_classification upsert 行数: {total_shapes}")
        print("导入完成。")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
