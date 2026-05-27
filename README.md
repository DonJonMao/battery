# battery

这个项目当前分成两个阶段：

1. 阶段一：故障预测 / 异常检测  
   入口脚本：`cluster_voltage_anomalies.py`
2. 阶段二：基于图形学的曲线形状分类  
   入口脚本：`classify_battery_plot_shapes.py`

另外还提供了一个“备选阶段二”脚本：`classify_battery_raw_shapes.py`。它不依赖阶段一输出的 PNG，而是直接从原始单体电压曲线生成对比图并做形状分类。

## 目录说明

- `UPS/型号1`：UPS 型号 1 的原始 Excel 数据
- `开关电源`：开关电源数据
- `cluster_voltage_anomalies.py`：阶段一异常检测
- `classify_battery_plot_shapes.py`：阶段二，读取阶段一生成的 PNG 做三分类
- `classify_battery_raw_shapes.py`：备选阶段二，直接基于原始曲线做三分类

## 环境准备

建议使用 Python 3.10 及以上。

安装依赖：

```bash
cd /mnt/nvme/projects/battery
python -m pip install -U pandas numpy scikit-learn pillow matplotlib openpyxl
```

如果想看脚本参数说明，可以直接执行：

```bash
python cluster_voltage_anomalies.py --help
python classify_battery_plot_shapes.py --help
python classify_battery_raw_shapes.py --help
```

## 推荐启动方式

下面以 `UPS/型号1` 为例。

### 1. 阶段一：故障预测 / 异常检测

```bash
cd /mnt/nvme/projects/battery
python cluster_voltage_anomalies.py \
  --input-dir /mnt/nvme/projects/battery/UPS/型号1 \
  --output-dir /mnt/nvme/projects/battery/output_ups_型号1
```

输出内容主要包括：

- `output_ups_型号1/anomalies_single_cells.csv`：单体异常检测结果
- `output_ups_型号1/anomalies_single_battery_groups.csv`：按设备聚合后的异常概览
- `output_ups_型号1/plots/单体XX电压/*.png`：单体异常对比图
- `output_ups_型号1/anomalies_pack_voltage.csv`：电池组总电压异常检测结果
- `output_ups_型号1/plots/电池组总电压/*.png`：电池组总电压异常图

说明：

- 现在阶段一默认已经改成“全量绘图”。
- `--max-plots` 的默认值是 `-1`，表示对全部异常样本绘图。
- 如果你只想抽样调试，可以手动加上 `--max-plots 80` 之类的限制。

### 2. 阶段二：基于阶段一 PNG 的图形学分类

```bash
cd /mnt/nvme/projects/battery
python classify_battery_plot_shapes.py \
  --input-dir /mnt/nvme/projects/battery/output_ups_型号1/plots/单体XX电压 \
  --output-dir /mnt/nvme/projects/battery/output_ups_型号1/plots_三分类/单体XX电压
```

输出内容主要包括：

- `output_ups_型号1/plots_三分类/单体XX电压/shape_classification.csv`：每张图的分类明细
- `output_ups_型号1/plots_三分类/单体XX电压/shape_class_summary.csv`：三分类汇总
- `output_ups_型号1/plots_三分类/单体XX电压/放电型异常/*.png`
- `output_ups_型号1/plots_三分类/单体XX电压/疑似操作或采集异常/*.png`
- `output_ups_型号1/plots_三分类/单体XX电压/不确定/*.png`

这个脚本会遍历 `--input-dir` 下的全部 PNG，因此只要阶段一已经全量绘图，阶段二就会自动做全量分类。

## 阶段一/阶段二的实现细节（当前代码逻辑）

### 阶段一：`cluster_voltage_anomalies.py` 具体怎么做

阶段一不是简单阈值告警，而是“数据清洗 + 曲线对齐 + 特征聚类 + 多条件融合”的流程：

1. 数据读取与容错
   - 从 `--input-dir` 扫描全部 `.xlsx`。
   - 优先走 `pandas` 读取；如果失败，会回退到 XML 级读取（解压 xlsx 后直接读 `sheetData`），避免部分文件格式不规范导致整批失败。
   - 默认优先读取 `DC*` 工作表；没有 `DC` 时回退首个 sheet。

2. 字段标准化
   - 自动补齐或映射必要列：例如 `上报值 -> 监控值`。
   - 若缺少 `组别`，会从 `监控量` 中抽取“第X组”。
   - 最终强校验关键字段：`机房/设备/测点类型/监控量/上报时间/监控值`。

3. 按类别分两条检测支路
   - `单体XX电压`：`测点类型` 含“单体”。
   - `电池组总电压`：`测点类型` 含“总电压”。
   - 两类使用不同的参数配置（如最少点数、硬阈值策略）。

4. 序列构建与放电窗口截取
   - 以 `机房|设备|组别|监控量|文件|sheet|测点编码` 作为 `series_id` 聚合。
   - 每条序列按时间排序并做去重均值。
   - 使用平滑后曲线自动定位放电窗口（从 5% 下降到 95% 下降区间）；若定位失败则保留全段。
   - 将每条曲线重采样到统一长度（默认 60 点），保证后续可比较。

5. 时间对齐与特征工程
   - 在允许范围内做左右平移，选择与参考中位曲线 MSE 最小的对齐位置。这里的“参考中位曲线”是当前类别下全量序列的中位形态。
   - 提取 5 维核心特征：`v_mean / v_range / d2_std / zigzag / shift_abs`。
   - 特征经 `RobustScaler` 归一化，再做 PCA 降维后用于聚类。

6. 异常判定（两轮判定 + 融合规则）
   - 用 DBSCAN 自动估计 `eps` 并聚类，噪声点先标异常。
   - 计算 KNN 距离分数与簇中心距离分数。
   - 第一轮（全量粗筛）组合规则：`噪声点` 或 `远离中心且KNN高分` 记为候选异常。
   - 对“单体XX电压”再加硬阈值：`v_max > 2.6 且 v_min < 1.7` 直接判异常。（不放电时可以预测内阻和温度）
   - 第二轮（同组复核）再做同组相似度过滤：按 `机房+设备+组别` 计算组内中位曲线相关系数；若候选异常与同组中位曲线高度一致，则从异常中剔除，避免“整组一起偏移但并非单点异常”的误报。

7. 输出与绘图
   - 输出单体与总电压两套 CSV 结果和按设备聚合统计。
   - 对每个异常生成“绝对电压 + 对齐形状”双子图 PNG。
   - `--max-plots < 0` 表示全量画图（当前默认 `-1`，即全量）。

### 阶段二：`classify_battery_plot_shapes.py` 具体怎么做

阶段二是“从阶段一图片中反解曲线，再做规则化形状分类”：

1. 输入来源
   - 读取阶段一输出目录 `plots/单体XX电压` 下全部 PNG。
   - 每张 PNG 的文件名包含来源文件、sheet、机房、设备、组别等元信息，阶段二会解析并回填到结果表。

2. 曲线提取（图像到序列）
   - 用 RGB 阈值提取红色曲线像素（目标线是红色）。
   - 自动识别面板横向区间，默认取左侧主面板。
   - 按列取红像素中位数，插值补齐缺失点，得到一条连续曲线。
   - 将 y 轴反向并归一化到 `[0,1]`，得到“电压形状近似序列”。

3. 形状指标计算
   - 平滑后提取谷底位置 `valley_x`、起点/谷底/终点电平。
   - 计算核心指标：`drop / recover / recover_ratio / pre_down_ratio / post_up_ratio / sign_change_ratio / dynamic_range / end_drop`。

4. 三分类决策逻辑
   - `放电型异常`：存在内部谷底，且谷底后有持续回升，波形稳定。
   - `疑似操作或采集异常`：如动态范围过小、谷底贴边、末端突降、晚谷底无回升、锯齿噪声明显。
   - `不确定`：介于两类之间，建议人工复核。

5. 结果落盘
   - 输出每张图的分类明细 `shape_classification.csv`。
   - 输出汇总 `shape_class_summary.csv`。
   - 按类别复制图片到 3 个目录，便于人工抽查。

### 两阶段的衔接关系

- 阶段一产出“异常清单 + 异常图”；阶段二默认消费阶段一的异常图。
- 阶段二是否全量，本质取决于阶段一是否已经把异常样本全量绘图完成。
- 当前默认参数下，阶段一已是全量绘图模式（`--max-plots=-1`），因此阶段二可直接全量分类。

## 备选方式：直接从原始曲线做阶段二

如果你不想依赖阶段一生成的 PNG，可以直接基于异常清单和原始曲线做三分类：

```bash
cd /mnt/nvme/projects/battery
python classify_battery_raw_shapes.py \
  --input-dir /mnt/nvme/projects/battery/UPS/型号1 \
  --anomaly-csv /mnt/nvme/projects/battery/output_ups_型号1/anomalies_single_cells.csv \
  --output-dir /mnt/nvme/projects/battery/output_ups_型号1_原始三分类/单体XX电压
```

输出内容主要包括：

- `output_ups_型号1_原始三分类/单体XX电压/raw_shape_classification.csv`
- `output_ups_型号1_原始三分类/单体XX电压/raw_shape_classification_summary.csv`
- 三个类别目录下的原始曲线对比图

## 常见运行顺序

最常用的是这一套：

```bash
cd /mnt/nvme/projects/battery
python cluster_voltage_anomalies.py \
  --input-dir /mnt/nvme/projects/battery/UPS/型号1 \
  --output-dir /mnt/nvme/projects/battery/output_ups_型号1

python classify_battery_plot_shapes.py \
  --input-dir /mnt/nvme/projects/battery/output_ups_型号1/plots/单体XX电压 \
  --output-dir /mnt/nvme/projects/battery/output_ups_型号1/plots_三分类/单体XX电压
```

## 重新跑数据时的建议

- 如果你希望输出目录完全干净，重新运行前可以手动删除旧的 `output_*` 目录。
- 如果只是补齐之前没画出来的异常图，直接重新执行阶段一和阶段二即可。
- 阶段二是否全量，取决于阶段一 `plots/单体XX电压` 目录里是否已经有全量 PNG。

## 数据库存储

如果你希望把 `battery` 的原始数据和分析结果统一放进本机 MySQL，便于后续被 `drive` 直接查询，可以使用下面这套方式。

### 1. 建表

建表 SQL 在：

- `battery_schema.sql`

它会创建 4 张表：

- `battery_series`：原始序列元数据，一条曲线一行
- `battery_measurement`：原始测点时序数据
- `battery_anomaly_result`：阶段一异常检测结果
- `battery_shape_classification`：阶段二形状分类结果

另外还会创建一个视图：

- `battery_anomaly_group_summary_v`

这个视图用于复现原来 `anomalies_*_groups.csv` 的聚合口径，因此不再额外落重复汇总表。

### 2. 直接导入本机 MySQL

导入脚本在：

- `import_battery_to_mysql.py`

使用示例：

```bash
cd /mnt/nvme/projects/battery
python import_battery_to_mysql.py \
  --project-root /mnt/nvme/projects/battery \
  --host localhost \
  --user api_user \
  --database cloud \
  --init-schema
```

说明：

- 默认使用本机 MySQL，不使用 Docker。
- 默认密码为空；如果你的本地库有密码，补充 `--password '你的密码'` 即可。
- 脚本会自动扫描：
  - `UPS/**.xlsx`
  - `开关电源/**.xlsx`
  - `output*/anomalies_single_cells.csv`
  - `output*/anomalies_pack_voltage.csv`
  - `**/raw_shape_classification.csv`
  - `**/shape_classification.csv`
- 脚本按唯一键做 upsert，可以重复执行，不会简单重复插入。

### 3. 环境依赖

除了前面分析脚本用到的依赖外，入库脚本还需要：

```bash
python -m pip install -U pandas pymysql openpyxl
```

### 4. 设计说明

当前数据库设计故意把“原始时序”和“算法结果”拆开：

- 原始时序保真，后面可以重新训练、重新打标签、重新算特征
- 阶段一和阶段二结果可重复导入、可保留数据集来源
- `series_id` 作为稳定主键，把原始数据、异常结果、形状分类串起来
- 导入时会尝试按 `机房名称 + 城市` 去匹配 `drive` 项目里的 `machine_room.id`

这样后面要做“机房级风险画像”“预测驱动的调度策略”“直接从 `drive` 接口查询 battery 风险”都会简单很多。

### 5. 没有 MySQL 写权限时的本地落库方案

如果当前机器上的 MySQL 账号只有读权限，或者你不想额外申请数据库权限，可以先直接落到本地 SQLite：

- 导入脚本：`import_battery_to_sqlite.py`
- 生成文件：`battery_dataset.sqlite`

使用示例：

```bash
cd /mnt/nvme/projects/battery
python import_battery_to_sqlite.py \
  --project-root /mnt/nvme/projects/battery \
  --sqlite-path /mnt/nvme/projects/battery/battery_dataset.sqlite \
  --machine-room-csv /mnt/nvme/projects/drive/scripts/machine_room.csv
```

说明：

- 这会把 UPS、开关电源、阶段一异常结果、阶段二形状分类结果一起写入一个本地 SQLite 数据库文件。
- 仍然会尝试利用 `machine_room.csv` 去补 `machine_room_id`，这样后续和 `drive` 侧关联更容易。
- 表结构与 MySQL 方案保持一致的业务含义：
  - `battery_series`
  - `battery_measurement`
  - `battery_anomaly_result`
  - `battery_shape_classification`
  - `battery_anomaly_group_summary_v`
