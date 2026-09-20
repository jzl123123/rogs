# 验证记录

更新日期：2026-09-15。本次在 Mac 上完成工程整理与 HUGSIM 地面后端集成；没有连接远端 Ubuntu 或访问真实 nuScenes 数据。

| 检查 | 状态 |
|---|---|
| `pip install --no-build-isolation --no-deps -e .` | 通过，成功生成 editable 包并安装 CLI |
| `rogs --help` | 通过，无需加载 CUDA |
| `python tools/verify_algorithm.py` | 通过，19 个算法文件仅导入命名空间变化 |
| `python -m unittest discover -s tests -v` | 通过，24 项测试 |
| GS 构建副本生成 | 通过，RGB 3 通道、语义 7 通道，正确 GLM 路径 |
| `python tools/fetch_thirdparty.py --verify` | 通过，9 份第三方源码的提交和内容摘要已记录并校验 |
| Python 编译检查 | 通过，src/tools/tests |
| 真实数据索引/预处理 | 未运行；索引已用包含关键帧与 sweep 的合成元数据测试 |
| CUDA 编译、GS 前反向 | 未运行，须在远端执行 doctor smoke |
| 分割权重推理 | 未运行；权重下载器和独立配置已提供 |
| 同场景 baseline / 工程版数值对照 | 未运行；命令和结果对比器已提供 |
| 论文指标复现 | 未完成；还需核对原标签和环境 |
| `python tools/verify_hugsim.py` | 通过，11 个 HUGSIM 文件的函数/类 AST 对齐，另核对共享 RoGS mask |
| HUGSIM gsplat 构建副本 | 通过，使用其锁文件指定 fork 和真实 GLM gitlink；未编译 CUDA |
| HUGSIM 共享数据准备 | 通过合成 nuScenes 元数据 + 实际 JPEG/PNG/NPY/PLY 的 CPU 端到端测试 |
| HUGSIM BEV 几何 | 世界坐标、图像行列方向、高度关系及像素预算检查通过；未做 GPU 渲染 |
| HUGSIM 真实 GPU 训练/导出 | 未执行；已提供独立安装、RGB/20 通道前后向 smoke、训练和导出入口 |

原 16 项测试继续覆盖 RoGS 算法、配置、索引及分割缓存控制流。新增 8 项覆盖 HUGSIM 算法对齐、默认配置、相机投影、共享标签映射、最近相机地面投影、稀疏源码提取、真实图像数据转换/缓存失效和 BEV 几何。分割模型推理仍未实测；新增实际图像测试针对已有标签的适配，不代表 GPU 分割或重建端到端通过。
