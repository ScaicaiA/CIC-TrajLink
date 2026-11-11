# CIC-TrajLink

四城 Top10 主实验的完整代码与数据。默认依次运行 Istanbul、Tokyo、Moscow、NYC，训练种子为 **42、43、44**，共 **12 项**；输出各次结果及三种子的均值、样本标准差。

流程：Transformer 轨迹编码与点关系 MLP → 双路候选召回 → 14 维边特征与概率校准 → 轨迹分组。训练和推理实现位于 `cic/`。

## 固定参数 kp、kc

`kp` 是 POI 路径的候选近邻数，`kc` 是 Transformer 编码路径的候选近邻数。每个城市、训练种子直接使用下表的固定值；表格单元为 **(kp, kc)**。

| 城市 | seed 42 | seed 43 | seed 44 |
| --- | --- | --- | --- |
| Istanbul | **(500, 100)** | **(400, 200)** | **(300, 200)** |
| Tokyo | **(400, 700)** | **(400, 700)** | **(600, 600)** |
| Moscow | **(400, 900)** | **(300, 1000)** | **(200, 1000)** |
| NYC | **(100, 200)** | **(100, 200)** | **(100, 200)** |

可执行配置为 [`config/four_cities.json`](config/four_cities.json) 中各城市的 `fixed_k`。两路取正相似度近邻，排除自身，合并为去重的无向候选边；有效近邻数最多为当前轨迹数减一。

## 数据与实验设置

四城数据完整包含在 `data/*.csv.gz` 中，使用无损 gzip 压缩。首次运行自动恢复为原 CSV 并检查 SHA-256；不需要另行下载数据。CSV 的五列为 `uuid, start_time, end_time, point_count, points_json`，保留原记录顺序与内容。

| 城市 | 总用户数 | 总轨迹数 | 神经网络训练用户 | 校准用户 | 验证用户 | 测试用户 | 测试轨迹数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Istanbul | 1282 | 12820 | 692 | 77 | 256 | 257 | 2570 |
| Tokyo | 2090 | 20900 | 1129 | 125 | 418 | 418 | 4180 |
| Moscow | 1971 | 19710 | 1065 | 118 | 394 | 394 | 3940 |
| NYC | 479 | 4790 | 258 | 29 | 96 | 96 | 960 |

每用户 10 条轨迹。先按用户划分 60% / 20% / 20%，再将原训练用户按 90% / 10% 分为神经网络训练与独立校准两部分。划分种子固定为 42，三个训练种子共用相同的用户划分及轨迹顺序。

- Transformer：30 epochs，batch_users=16，学习率 0.001，weight_decay=0.0001，temperature=0.1。
- 点关系 MLP：20 epochs，batch_size=128，学习率 0.001，weight_decay=0.001；训练轨迹对 16384，验证轨迹对 4096，正例比例 0.25。
- 每条轨迹最多保留 32 个点；Transformer 输出 32 维编码。
- 独立校准部分拟合六维支持模型与 14 维 HGB 校准器；HGB 的 max_iter=100、max_leaf_nodes=7、min_samples_leaf=50、learning_rate=0.1、l2_regularization=1。
- 最终分组使用已校准的候选边概率，保持原主实验的分组目标与细化规则。推理不接收真实用户数或每用户轨迹数。

## 安装

使用 Python 3.10+，完整实验建议在 Linux CUDA 服务器上运行。CPU 可用于检查流程。

```bash
cd CIC-TrajLink
python3 -m pip install -r requirements.txt
python3 verify_package.py
```

GPU 运行前检查 PyTorch 是否能使用 CUDA：

```bash
python3 -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'Install a CUDA-enabled PyTorch build compatible with your server'"
```

如果服务器已有可用的 CUDA PyTorch，满足版本范围即可，无需专门替换。`verify_package.py` 只验证代码和数据，不启动训练。

## 运行全部主实验

在项目根目录执行：

```bash
python3 -u run_main.py --device cuda --output runs/main
```

该命令从头训练每个城市、种子的 Transformer 与 MLP，拟合后端，完成测试推理和指标计算；所有 12 项串行使用同一设备。全部完成时，终端打印 `main_experiment_complete`，`runs/main/progress.json` 的 `status` 为 `complete`。

在服务器后台运行：

```bash
nohup python3 -u run_main.py --device cuda --output runs/main > main.log 2>&1 &
echo $! > main.pid
```

查看进度和日志：

```bash
cat runs/main/progress.json
tail -n 60 -f main.log
```

退出 `tail -f` 时按 Ctrl+C 只退出日志查看；后台实验继续运行。

中断后使用相同参数继续：

```bash
nohup python3 -u run_main.py --device cuda --output runs/main --resume >> main.log 2>&1 &
echo $! > main.pid
```

`--resume` 校验并跳过已完成项，复用完整训练结束的 Transformer。未完成阶段保存在 `incomplete_attempts/`，再从该阶段起点重跑；不支持恢复到训练阶段中间的某个 batch。已完成的模型与结果不会覆盖。同一输出目录不能同时启动两个任务。恢复时保持代码、配置、所选城市/种子、环境与数据路径一致。

只运行一个城市、种子：

```bash
python3 -u run_main.py --device cuda --cities NYC --seeds 42 --output runs/nyc_seed42
```

单种子运行只列逐项结果。需要 NYC 三种子汇总时，去掉 `--seeds 42` 并使用新的输出目录；均值与标准差只对完成全部三个种子的城市生成。要运行 CPU 版本，把 `--device cuda` 改为 `--device cpu`。

## 读取结果

| 文件 | 内容 |
| --- | --- |
| `runs/main/summary.txt` | 可直接阅读的逐项结果与三种子汇总 |
| `runs/main/summary.json` | 完成数量、完整实验标记、逐项指标与汇总 |
| `runs/main/per_seed_results.csv` | 每个城市、种子的结果 |
| `runs/main/mean_std_results.csv` | 已完成三个种子的城市均值与样本标准差 |
| `runs/main/progress.json` | 当前任务、完成数量或失败原因 |
| `runs/main/environment.json` | 实际运行环境和依赖版本 |
| `runs/main/logs/<城市>_seed<种子>.log` | 单项训练与推理日志 |
| `runs/main/seed_42/preparation/NYC/encoder/encoder.pt` | 示例：NYC seed 42 的 Transformer 权重 |
| `runs/main/seed_42/main/NYC/mlp/pointwise_mlp.pt` | 示例：对应 MLP 权重 |
| `runs/main/seed_42/main/NYC/calibration/` | 示例：支持模型、14 维校准器和校准记录 |
| `runs/main/seed_42/main/NYC/test/` | 示例：最终分组、候选边、概率及测试指标 |

G-Acc、Pairwise Precision / Recall / F1 的汇总单位是百分数；标准差采用 `ddof=1`。单项 `metrics.json` 中对应指标使用 0–1 比例。NMI、ARI 保持原数值尺度。推理计时包含特征预处理、编码、召回、打分、14 维特征与分组，不包含训练、数据加载、结果写盘或指标计算。

可以单独刷新汇总并与已有主实验结果核对：

```bash
python3 collect_results.py --root runs/main
python3 compare_reference.py --run runs/main
```

`reference/main_results.json` 保存已有主实验的参考指标；运行程序不会把参考值当作新结果写入输出目录。对应的三种子参考值如下：

| 城市 | G-Acc (%) | Pairwise F1 (%) |
| --- | ---: | ---: |
| Istanbul | 79.455 ± 0.206 | 77.788 ± 1.340 |
| Tokyo | 71.738 ± 0.251 | 70.592 ± 0.294 |
| Moscow | 77.462 ± 0.766 | 75.641 ± 0.489 |
| NYC | 86.736 ± 1.114 | 85.261 ± 1.805 |

固定数据、配置与种子用于复现实验；不同硬件、PyTorch 或数值库版本可能导致数值差异，不保证逐位一致。实际结果以本次运行生成的文件为准。

## 代码检查

```bash
python3 -m unittest discover -s tests -v
python3 tests/integration_smoke.py --output runs/smoke --cities Istanbul Tokyo Moscow NYC
```

第二条命令在每城 60 个用户的真实数据子集上，以 2 epochs 跑通 12 项训练、校准、推理、汇总与恢复流程，输出明确标记为 `engineering_smoke`。它用于验证程序可运行，不代表正式实验指标。发布检查记录见 `verification.json`。

## 上传仓库

将本目录的内容上传为仓库根目录即可。代码、固定配置、说明、测试、参考指标和四城压缩数据均已包含。运行时解压的 CSV、模型权重、日志与 `runs/` 已列入 `.gitignore`。
