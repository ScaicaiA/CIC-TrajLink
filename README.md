# CIC-TrajLink

四城 Top10 主实验的完整代码与数据。

## 数据与实验设置

四城数据完整包含在 `data/*.csv.gz` 中，使用无损 gzip 压缩。首次运行自动恢复为原 CSV 并检查 SHA-256；不需要另行下载数据。CSV 的五列为 `uuid, start_time, end_time, point_count, points_json`，保留原记录顺序与内容。

| 城市 | 总用户数 | 总轨迹数 | 神经网络训练用户 | 校准用户 | 验证用户 | 测试用户 | 测试轨迹数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Istanbul | 1282 | 12820 | 692 | 77 | 256 | 257 | 2570 |
| Tokyo | 2090 | 20900 | 1129 | 125 | 418 | 418 | 4180 |
| Moscow | 1971 | 19710 | 1065 | 118 | 394 | 394 | 3940 |
| NYC | 479 | 4790 | 258 | 29 | 96 | 96 | 960 |

每用户 10 条轨迹。先按用户划分 60% / 20% / 20%，再将原训练用户按 90% / 10% 分为神经网络训练与独立校准两部分。划分种子固定为 42，三个训练种子共用相同的用户划分及轨迹顺序。

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

## 运行全部主实验

在项目根目录执行：

```bash
python3 -u run_main.py --device cuda --output runs/main
```

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

只运行一个城市、种子：

```bash
python3 -u run_main.py --device cuda --cities NYC --seeds 42 --output runs/nyc_seed42
```


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
