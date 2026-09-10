# 验证记录

日期：2026-09-09。本次在 Mac 上完成工程整理；没有连接远端 Ubuntu 或访问真实 nuScenes 数据。

| 检查 | 状态 |
|---|---|
| `pip install --no-build-isolation --no-deps -e .` | 通过，成功生成 editable 包并安装 CLI |
| `rogs --help` | 通过，无需加载 CUDA |
| `python tools/verify_algorithm.py` | 通过，19 个算法文件仅导入命名空间变化 |
| `python -m unittest discover -s tests -v` | 通过，16 项测试 |
| GS 构建副本生成 | 通过，RGB 3 通道、语义 7 通道，正确 GLM 路径 |
| `python tools/fetch_thirdparty.py --verify` | 通过，6 份第三方源码的提交和内容摘要已记录并校验 |
| Python 编译检查 | 通过，src/tools/tests |
| 真实数据索引/预处理 | 未运行；索引已用包含关键帧与 sweep 的合成元数据测试 |
| CUDA 编译、GS 前反向 | 未运行，须在远端执行 doctor smoke |
| 分割权重推理 | 未运行；权重下载器和独立配置已提供 |
| 同场景 baseline / 工程版数值对照 | 未运行；命令和结果对比器已提供 |
| 论文指标复现 | 未完成；还需核对原标签和环境 |

16 项测试覆盖算法 AST、公式变化检测、原始配置保留、环境/相对路径、sample/sweep 索引、精确场景筛选、缺失数据、目录穿越、循环链、语义扩展 CUDA 文件不变、Mapillary 原始类别对应，以及缓存全命中不加载模型、无效缓存提前阻断、不同模型不复用缓存、缺失帧筛选。缓存控制流测试使用模拟依赖，尚未完成真实 PNG/JPEG 和 GPU 推理的端到端测试。测试不声称替代真实 GPU 与数据验收。
