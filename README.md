# ALP-AWB3

[English](README.en.md) | 中文

ALP-AWB3 是用于 **0–1 多维背包问题**的低延迟数学启发式。给定每件物品的收益、各类资源消耗和资源总容量，它返回满足全部容量约束的物品集合。

算法先用自适应线性规划构造可行解，再通过至多三个、每个至多 20 件物品的受限核联合重新选择。追加窗口交替组织移出与移入候选，采用“保留 10 件＋引入 10 件”的滚动方式；价格辅助位图加速资源可行的半表配对，半表与完整受限问题缓存复用重叠计算。当前配置包括首核在内最多三核，不使用第二次全局 LP 构造、GNN、Transformer、信息素或模型权重。

本仓库提供当前求解器及使用说明。实验数据、实验设置、实验环境记录和论文文件不在本仓库中。

## 安装

已验证的 Python 版本为 **3.12**。求解器只依赖 `requirements.txt` 中的 NumPy 和 HiGHS Python 接口（`highspy`），不需要 PyTorch、CPLEX 或 GPU。

```bash
git clone https://github.com/GRX666-yy/ALP-AWB3.git
cd ALP-AWB3
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

若系统使用 `python3` 命令，请相应替换上述 `python`。

## Python 接口

下面是一个六物品、两资源的人工使用示例，不是实验数据。

```python
from solve import solve

result = solve(
    values=[9, 8, 12, 6, 10, 7],
    weights=[[5, 2], [2, 5], [6, 6], [3, 1], [1, 4], [4, 3]],
    capacity=[10, 10],
    budget_s=0.05,
    seed=0,
    name="example",
)
print("profit:", result["value"])
print("selected item indices:", result["selected_ids"])
print("algorithm return time (s):", result["elapsed_s"])
```

输入格式：`values` 为长度 N 的收益数组；`weights` 为 **N×M，物品在行、资源在列**的消耗矩阵；`capacity` 为长度 M 的容量数组。N、M 必须为正，收益与容量须为有限正数，资源消耗须为有限非负数，预算须为有限非负数。超出单项容量的物品不会被选入可行解。

## 命令行接口

把下列内容保存为 UTF-8 编码的 `instance.json`：

```json
{
  "name": "example",
  "values": [9, 8, 12, 6, 10, 7],
  "weights": [[5, 2], [2, 5], [6, 6], [3, 1], [1, 4], [4, 3]],
  "capacity": [10, 10]
}
```

```bash
python -B solve.py --input instance.json --budget-ms 50 --seed 0 --output solution.json
```

省略 `--output` 时结果打印到终端；命令行预算单位为毫秒，Python 接口的 `budget_s` 单位为秒。可用 `python solve.py --help` 查看参数。输出文件的父目录应已存在。

主要返回字段：

|字段|含义|
|---|---|
|`value`|已接纳可行解的总收益|
|`selected_ids`|被选物品的**从 0 开始**的下标|
|`elapsed_s`|内部求解流程的实际返回时间，单位秒|
|`wrapper_elapsed_s`|数组转换、输入检查、求解和返回后校验的总时间|
|`budget_s`|改善接纳预算，单位秒|
|`events`|接纳的改善轨迹，含时间、收益与压缩选集|
|`stats`|构造、窗口、缓存和位图配对等运行统计|
|`model_version`|当前冻结配置标识 `88`|

`selected` 为内部压缩位图十六进制字符串；常规使用请读取 `selected_ids`。两种时间字段都不包含 Python 启动、模块导入和 JSON 文件读写。

## 预算与调用方式

预算控制**可行改善的接纳截止时间**，实际返回可能晚于预算。若极短预算内来不及接纳非零解，将返回收益为 0 的空可行集。算法采用有限邻域，不承诺全局最优。

内部依赖临时模块替换，同一进程中的调用应**顺序执行**。需要并行处理不同实例时，使用相互独立的进程，避免多线程同时调用。数值库线程数在求解器导入时设为 1。

## 代码结构与版本

```text
solve.py                数组接口和 JSON 命令行入口
requirements.txt        当前求解器的依赖版本
runtime/
  problem.py            实例、容量校验、接纳与贪心保底
  lp.py                 HiGHS 与自适应 LP 构造
  candidates.py         LP 特征和候选排序
  prefix.py, clock.py   E100 前缀和统一计时
  repair.py, ordering.py 首核接口与窗口次序
  engine.py, core.py    三核滚动与受限问题求解
  bitmap.py, common.py  位图配对与依赖装配
```

这是配置标识为 `mdkp-bitmap-three-core-v88` 的精简运行包。它保留当前使用路径，并将所需函数重新连接到本地模块；完整历史测量源码另行保存。更新说明文档不会改变求解器配置，也不会重新生成实验结果。
