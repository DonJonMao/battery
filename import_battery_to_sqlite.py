#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 battery 项目的原始数据与分析结果导入本地 SQLite 数据库。

用途：
- 在当前机器上没有 MySQL 建表权限时，先生成一个可直接查询的本地数据集库。
- 保留与 drive 侧 machine_room 的机房映射能力，便于后续统一调用。
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from import_battery_to_mysql import (
    RAW_REQUIRED_COLUMNS,
    build_anomaly_rows,
    build_shape_rows,
    build_series_rows,
    build_measurement_rows,
    build_unique_lookup,
    dataset_meta_from_dir,
    discover_files,
    discover_raw_dataset_dirs,
    ensure_columns,
    load_voltage_rows_any,
    normalize_city,
    normalize_columns,
    text_or_none,
)


def load_machine_room_lookup_from_csv(csv_path: Path):
    if not csv_path.exists():
        return {}, {}
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    room_city_pairs = []
    room_only_pairs = []
    for _, row in df.iterrows():
        room_name = text_or_none(row.get("room_name"))
        city = normalize_city(text_or_none(row.get("city")))
        room_id = row.get("id")
        if room_name:
            room_only_pairs.append((room_name, room_id))
            if city:
                room_city_pairs.append(((room_name, city), room_id))
    return build_unique_lookup(room_city_pairs), build_unique_lookup(room_only_pairs)


def normalize_frame_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for column in df.columns:
        if pd.api.types.is_bool_dtype(df[column]):
            df[column] = df[column].astype(int)
    return df


def write_table(conn: sqlite3.Connection, table_name: str, rows: list[dict]) -> int:
    df = normalize_frame_rows(rows)
    if df.empty:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        return 0
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    return len(df)


def create_indexes(conn: sqlite3.Connection) -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_battery_series_series_id ON battery_series(series_id)",
        "CREATE INDEX IF NOT EXISTS idx_battery_series_room_name ON battery_series(room_name)",
        "CREATE INDEX IF NOT EXISTS idx_battery_series_machine_room_id ON battery_series(machine_room_id)",
        "CREATE INDEX IF NOT EXISTS idx_battery_measurement_series_id ON battery_measurement(series_id)",
        "CREATE INDEX IF NOT EXISTS idx_battery_measurement_report_time ON battery_measurement(report_time)",
        "CREATE INDEX IF NOT EXISTS idx_battery_anomaly_series_id ON battery_anomaly_result(series_id)",
        "CREATE INDEX IF NOT EXISTS idx_battery_anomaly_machine_room_id ON battery_anomaly_result(machine_room_id)",
        "CREATE INDEX IF NOT EXISTS idx_battery_anomaly_flag ON battery_anomaly_result(anomaly_flag)",
        "CREATE INDEX IF NOT EXISTS idx_battery_shape_series_id ON battery_shape_classification(series_id)",
        "CREATE INDEX IF NOT EXISTS idx_battery_shape_machine_room_id ON battery_shape_classification(machine_room_id)",
        "CREATE INDEX IF NOT EXISTS idx_battery_shape_class ON battery_shape_classification(shape_class)",
    ]
    for statement in statements:
        conn.execute(statement)
    conn.commit()


def create_views(conn: sqlite3.Connection) -> None:
    conn.execute("DROP VIEW IF EXISTS battery_anomaly_group_summary_v")
    conn.execute(
        """
        CREATE VIEW battery_anomaly_group_summary_v AS
        SELECT
          result_dataset,
          result_kind,
          source_file,
          source_sheet,
          room_name,
          device_name,
          COUNT(DISTINCT series_id) AS total_series,
          SUM(CASE WHEN anomaly_flag = 1 THEN 1 ELSE 0 END) AS anomaly_series,
          MAX(knn_score) AS max_knn,
          CASE
            WHEN COUNT(DISTINCT series_id) = 0 THEN 0
            ELSE 1.0 * SUM(CASE WHEN anomaly_flag = 1 THEN 1 ELSE 0 END) / COUNT(DISTINCT series_id)
          END AS anomaly_ratio
        FROM battery_anomaly_result
        GROUP BY
          result_dataset,
          result_kind,
          source_file,
          source_sheet,
          room_name,
          device_name
        """
    )
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="将 battery 项目数据导入本地 SQLite。")
    parser.add_argument("--project-root", default="/mnt/nvme/projects/battery", help="battery 项目根目录")
    parser.add_argument("--sqlite-path", default="/mnt/nvme/projects/battery/battery_dataset.sqlite", help="SQLite 数据库文件路径")
    parser.add_argument(
        "--machine-room-csv",
        default="/mnt/nvme/projects/drive/scripts/machine_room.csv",
        help="drive 导出的机房快照 CSV，用于补 machine_room_id",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    sqlite_path = Path(args.sqlite_path).resolve()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    room_city_lookup, room_lookup = load_machine_room_lookup_from_csv(Path(args.machine_room_csv).resolve())

    raw_dataset_dirs = discover_raw_dataset_dirs(project_root)
    series_info_by_id = {}
    resolver_pairs = []

    all_series_rows = []
    all_measurement_rows = []
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

        all_series_rows.extend(series_rows)
        all_measurement_rows.extend(measurement_rows)
        series_info_by_id.update(dataset_series_info)
        resolver_pairs.extend((key, value) for key, value in dataset_resolver.items() if value)

    resolver = build_unique_lookup(resolver_pairs)

    all_anomaly_rows = []
    for csv_path in discover_files(project_root, "anomalies_single_cells.csv"):
        all_anomaly_rows.extend(
            build_anomaly_rows(
                csv_path=csv_path,
                project_root=project_root,
                series_info_by_id=series_info_by_id,
                room_city_lookup=room_city_lookup,
                room_lookup=room_lookup,
            )
        )
    for csv_path in discover_files(project_root, "anomalies_pack_voltage.csv"):
        all_anomaly_rows.extend(
            build_anomaly_rows(
                csv_path=csv_path,
                project_root=project_root,
                series_info_by_id=series_info_by_id,
                room_city_lookup=room_city_lookup,
                room_lookup=room_lookup,
            )
        )

    all_shape_rows = []
    for csv_path in discover_files(project_root, "raw_shape_classification.csv"):
        all_shape_rows.extend(
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
        all_shape_rows.extend(
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

    conn = sqlite3.connect(sqlite_path)
    try:
        series_count = write_table(conn, "battery_series", all_series_rows)
        measurement_count = write_table(conn, "battery_measurement", all_measurement_rows)
        anomaly_count = write_table(conn, "battery_anomaly_result", all_anomaly_rows)
        shape_count = write_table(conn, "battery_shape_classification", all_shape_rows)
        create_indexes(conn)
        create_views(conn)
    finally:
        conn.close()

    print(f"SQLite 数据库: {sqlite_path}")
    print(f"原始数据集目录: {len(raw_dataset_dirs)}")
    print(f"battery_series 行数: {series_count}")
    print(f"battery_measurement 行数: {measurement_count}")
    print(f"battery_anomaly_result 行数: {anomaly_count}")
    print(f"battery_shape_classification 行数: {shape_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
