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
