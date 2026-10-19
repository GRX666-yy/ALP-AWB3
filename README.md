# ALP-AWB3

当前论文方法：ALP-AWB3，最多三核（含首核）、每窗至多20项、两个追加窗10留＋10新；自适应LP＋E100、固定引导交替窗口、原始半表/完整问题缓存和价格辅助位图。

此目录只保留当前调用路径所需代码。`solve.py`及当前`runtime/engine.py`、`core.py`、`bitmap.py`沿用原冻结文件；LP构造、候选排序等实际使用函数从原依赖中提取，函数体保持不变，重新连接本地模块，删除历史模型加载和未使用依赖。原始逐字节冻结发行与测量证据仍在原工作区，未被本目录替换。清理目录不会产生新的性能实验结论。

## 文件

- `solve.py`：唯一求解入口，支持数组接口及JSON命令行。
- `runtime/problem.py`：输入实例、容量检查、贪心保底及改善接纳。
- `runtime/lp.py`：HiGHS模型、自适应构造、有限扩展。
- `runtime/candidates.py`、`prefix.py`、`clock.py`：LP特征、候选次序、E100前缀和统一截止。
- `runtime/repair.py`、`ordering.py`：首核接口、半表枚举及交替次序。
- `runtime/engine.py`、`core.py`、`bitmap.py`、`common.py`：当前三核、配对后端及内部依赖装配。
- `requirements.txt`：已测环境使用的NumPy与HiGHS版本。无权重文件，不需PyTorch或CPLEX。

## 运行

建议Python 3.12。先在所选环境执行 `python -m pip install -r requirements.txt`。

```python
from solve import solve
result = solve(
    values=[9, 8, 12, 6, 10, 7],
    weights=[[5, 2], [2, 5], [6, 6], [3, 1], [1, 4], [4, 3]],
    capacity=[10, 10],
    budget_s=0.05,
    seed=0,
)
print(result['value'], result['selected_ids'])
```

JSON输入含`values`、`weights`（物品×资源）、`capacity`，可另填`name`。运行 `python -B solve.py --input your_input.json --budget-ms 50 --output your_result.json`。收益和容量须正，资源用量须非负，所有数值须有限。返回包含`value`、`selected_ids`、改善事件与统计。

预算是改善接纳截止，实际返回可能更晚。内部使用临时patch，须单进程顺序调用。当前主要证据是低维短预算，不保证全局最优或所有实例优于CPLEX。
