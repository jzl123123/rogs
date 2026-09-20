# HUGSIM 地面后端

本工程新增独立 HUGSIM 地面训练后端，原 RoGS 命令与算法继续保留。当前完成源码集成和 CPU 数据链路验证，尚未在 Ubuntu/NVIDIA 上编译、训练或验证 BEV 渲染；不声称已经复现 HUGSIM Table VI。

## 源码边界

- `thirdparty/HUGSIM`：官方提交 `62c690d39fd90020e68a196bd8bcc1c4d4191f2e`，保留根文件、训练/场景/工具代码、nuScenes 与相关预处理脚本的稀疏快照。
- `thirdparty/HUGSIM_splat`：采用 HUGSIM 自己的 pixi.lock 指定提交 `88f2a40c4e2f6bafde2beeaba6c43bbb0ccb1f5f`，支持 `smts` 与 `RGB+ED+S`；不可替换为普通 gsplat。
- `thirdparty/hugsim_glm`：读取上述渲染器的真实 gitlink，版本 `45008b225e28eb700fa0f7d3ff69b7c1db94fadf`，与 RoGS 的 GLM 分开。
- `src/rogs/hugsim`：地面模型、训练循环、增密/裁剪、损失及地面渲染函数来自上述快照，仅改导入命名空间。`verify_hugsim.py` 比较 11 个文件的函数/类 AST。
- 只提取地面使用的渲染与 PLY 函数，避免导入动态场景、tiny-cuda-nn 和闭环仿真；移除 GroundModel 中未使用的 simple_knn 导入。上游源码快照本身不修改，许可文件和来源注释保留。
- `adapter.py` 延迟读取 RGB/语义，避免一次性将整个场景图像装入内存；训练时的数据读取开销会与官方预加载方式不同。
- `data.py`、`runner.py`、`export.py` 属于工程适配，并非 HUGSIM 原论文算法文件。

## 共用与独立的口径

| 项目 | shared_rogs 模式 | native_hugsim 模式 |
|---|---|---|
| 图像索引 | 与 RoGS 相同的原始 nuScenes keyframes+sweeps，无需 ASAP | 使用现有 HUGSIM meta_data.json，保持原顺序 |
| 分割 | 复用原 Mapillary PNG，不重新推理 | 使用现有 Cityscapes 语义 NPY |
| 输入尺寸 | 使用 RoGS image_width/image_height，默认 800×450，再裁到 800×225 | 保持预处理文件中的实际尺寸 |
| 地面 mask | 默认精确使用 RoGS label2mask 和 CAM_BACK 底部排除规则 | 上游 labels≤1 |
| 地面语义 | 保持 HUGSIM 20 通道；Mapillary sidewalk→1，其余 RoGS 有效地面→0，无效→19 | 保持原始 20 通道语义 |
| 初始化 | RoGS 融合道路 PLY，采样后按 HUGSIM 最近前视相机平面投影 | 已有 ground_points3d.ply，保持原初始化 |
| 相机坐标 | 变换到第一帧前视相机坐标，记录 world_to_hugsim / inv_pose | 使用现有 camtoworld |
| 训练/验证划分 | 每路相机时间序列每第五帧留出，所有选定帧均记录 | 按上游 nuScenes idx%30≥24 留出 |
| 优化 | 独立 HUGSIM 默认参数：30000 次迭代、20 通道语义、L1+SSIM、局部高度约束、增密 | 同左 |
| 输出记录 | 共用 run_bundle，另存 HUGSIM 实际配置、数据来源和 checkpoint | 同左 |

共享 mask 中的路缘、地形等类别在 HUGSIM 中合并到有效地面类别 0；不能把 20 通道输出当作 RoGS 的 7 类语义地图。`mask_policy: road_sidewalk` 是可选的较窄标签映射，也不宣称等价于重新运行 InverseForm。

共享模式需要先生成 RoGS 的道路参考 PLY，默认最多采样 200000 点。相机高度默认按 HUGSIM nuScenes loader 的首帧 LiDAR 局部范围、0.01 m RANSAC 阈值估计；也可在 profile 中明确设置正数高度。它与 HUGSIM 原生的单目深度初始化不同。最近相机投影保留上游“排除最后一个前视相机”和距离相等时取首个索引的行为，距离计算分块以限制内存。

**评测限制：** RoGS 原配置用全部帧训练，共享 HUGSIM 则保留每第五帧；融合道路 PLY 的几何和颜色可能使用全场景观测，包括留出帧。因此这些结果是工程对照/重建诊断，不能当作严格无泄漏的新视角评测。正式比较应另行统一训练专用初始化、帧划分、语义来源、地面 mask 和评测区域。HUGSIM 自带结果包括 train/test 的 PSNR、SSIM、LPIPS；不要直接与 RoGS BEV 稀疏 LiDAR 指标比较。

## Ubuntu 安装

HUGSIM 当前官方环境为 Python 3.11、PyTorch 2.4.1/cu118；RoGS 参考栈较老，分别建环境以免覆盖已验证的 CUDA 扩展。工程源码、数据和分割缓存可跨环境共享。

```bash
conda env create -f environment-hugsim.yml
conda activate rogs-hugsim
python tools/install_hugsim.py
```

安装器只装地面所需包和专用 gsplat，不安装闭环模拟器、车辆模型、InverseForm 或 UniDepth。无需重新跑分割，但原生 HUGSIM 数据需要用户已有的原生预处理结果。PyTorch CUDA wheel 不包含 nvcc，主机仍需匹配的 CUDA Toolkit。TorchMetrics 的 LPIPS 首次运行还可能下载其度量网络权重；它不属于地面训练分割模型。

可单独生成构建目录或检查 GPU：

```bash
python tools/build_hugsim.py --prepare-only  # Mac 也可检查
python tools/build_hugsim.py                # 编译并运行 RGB/20 通道前后向 smoke
python -m rogs.hugsim.runner --smoke
```

## 跑共享数据

沿用 README 中的 `NUSC_ROOT`、`NUSC_SEG_ROOT`、`NUSC_GROUND_ROOT`。一次选一个场景。先在 RoGS 环境完成分割和道路 PLY，再切到 HUGSIM 环境：

```bash
# RoGS 环境；已有标签/PLY 时无需重复生成
rogs segment --config configs/nuscenes.yaml --model-profile configs/segmentation/mapillary_swin_l.yaml
rogs preprocess --config configs/nuscenes.yaml

# HUGSIM 环境
rogs hugsim-prepare --config configs/nuscenes.yaml --profile configs/hugsim/shared_nusc.yaml
rogs hugsim-train --config configs/nuscenes.yaml --profile configs/hugsim/shared_nusc.yaml --dry-run
rogs hugsim-train --config configs/nuscenes.yaml --profile configs/hugsim/shared_nusc.yaml
```

准备目录默认为 `data/hugsim_shared/<scene>`，包含 PNG 图像、NPY 语义、相机 JSON、初始 PLY、适配记录。原图和原分割缓存不会被修改。输入、协议或适配代码变化后拒绝复用旧目录，需要选择新的 prepared_root。图像用无损 PNG 保存，避免额外 JPEG 编码误差。

`--dry-run` 只核对数据并显示实际 HUGSIM 设置，不创建训练记录或启动 GPU。HUGSIM 参数放在 profile 的 `overrides`，不会把 RoGS 的训练参数强行移植过来。例如 GPU smoke 训练可设置：

```yaml
overrides:
  train:
    iterations: 20
    checkpoint_iterations: [20]
```

这仅用于检查能否跑通，不用于报告质量。原版 GroundModel 中有硬编码学习率，profile 中某些 opt 学习率字段并不会生效；保留原行为，不把这些字段包装为已生效的调参接口。

## 跑已有 HUGSIM 原生数据

```bash
export HUGSIM_SOURCE=/data/hugsim/scene-0038
rogs hugsim-prepare --config configs/nuscenes.yaml --profile configs/hugsim/native_nusc.yaml
rogs hugsim-train --config configs/nuscenes.yaml --profile configs/hugsim/native_nusc.yaml --dry-run
rogs hugsim-train --config configs/nuscenes.yaml --profile configs/hugsim/native_nusc.yaml
```

原生目录必须包含 `meta_data.json`、其中引用的 RGB、对应 `semantics/...npy`、`ground_points3d.ply`。本后端不需要与地面无关的 points3d.ply、动态物体标注或闭环文件。当前入口仍读取共享 RoGS 运行配置，需解析其中的三个数据路径环境变量，但原生模式不会访问这些 RoGS 数据目录。

## 地面地图导出

训练目录包含原生 `ckpts/ground_chkpnt*.pth`、`point_cloud_vis/.../ground.ply`、透视渲染图与 `ground/results.json`，另导出完整 `ground_gaussians.ply`。它不能用 RoGS checkpoint loader 加载。

使用同一 HUGSIM 渲染器生成世界坐标对齐的俯视地图：

```bash
python -m rogs.hugsim.export --run outputs/<HUGSIM运行目录> --resolution 0.05
```

输出 `bev/rgb.png`、`semantic.png`、`mask.png`、`height.npy`、`map.json`。`map.json` 记录米/像素、世界坐标、图像行列方向；默认 0.05 m/像素与 RoGS 一致。仅支持原始世界 Z 为竖直方向的 nuScenes/对应原生数据；要求 meta_data.json 提供 world→HUGSIM 的 inv_pose。mask 使用 alpha>0.5，属于明确的导出口径，不等于 RoGS 原生 mask。大于 1600 万像素时要求提高米/像素，避免无意分配过大 GPU 内存。尚未完成 GPU 导出实测。

## 验证与保留的上游问题

```bash
python tools/fetch_thirdparty.py --verify
python tools/verify_algorithm.py
python tools/verify_hugsim.py
python -m unittest discover -s tests -v
```

保留 HUGSIM 原训练循环和优化时序，包括最终迭代保存后不再 optimizer.step、局部高度项用 torch.std、度量对象的调用方式及 GroundModel 初始 opacity 行为。单点局部 patch 的 torch.std 可能产生 NaN；当前没有偷偷修改为另一种估计器。遇到这个问题应记录输入并单独评估修复版本。无有效地面像素的帧会在准备/检查阶段报错，避免空语义集合进入原版交叉熵；不会静默丢帧。
