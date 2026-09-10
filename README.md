# RoGS 工程版

将 [RoGS](https://github.com/fzhiheng/RoGS) 封装为可安装的 Python 工程，集中管理分割与 GS 源码依赖，保留官方算法行为。当前只整理 nuScenes 工作流，自采数据迁移后置。

本地 Mac 用于开发与静态检查；CUDA 编译、分割和建图在 Ubuntu + NVIDIA 上执行。**当前已完成源码级对齐检查，尚未做远端编译和数值复现；不能把工程封装完成当作论文指标已复现。**

## 目录

```text
RoGS/
  src/rogs/                 # namespaced 算法代码与 CLI/配置/数据检查
  configs/nuscenes.yaml     # 可运行配置，外部数据路径通过环境变量指定
  configs/reference/       # 原版 YAML，不修改
  configs/segmentation/    # 本地 Mask2Former 配置与权重入口
  thirdparty/
    rogs_upstream/          # 原版源码快照，用来做对照
    diff-gs-depth-alpha/    # 原版 RGB/depth/orthographic CUDA 渲染器
    glm/                   # GS 实际引用的子模块提交
    Mask2Former/           # 官方分割模型、配置与模型表
    detectron2/            # 分割框架 v0.6
    pytorch3d/             # V0.7.8 源码
    weights/               # 权重存放位置，不提交大文件
    sources.lock.json      # 精确提交与源码树 SHA256
    _build/                # 自动生成的构建副本，不修改上述源码
  tools/                   # 依赖校验、安装、扩展生成、算法/结果对比
  tests/                   # Mac 可运行的工程验证
  docs/                    # 环境说明、算法边界、上游 README
  outputs/                 # 运行配置、来源记录、日志和地图
```

## Mac 上检查工程

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python tools/verify_algorithm.py
python tools/fetch_thirdparty.py --verify
python -m unittest discover -s tests -v
python tools/build_extensions.py --prepare-only
rogs --help
```

`--prepare-only` 根据同一 GS 源码生成两个构建目录：RGB 为 3 通道，nuScenes 语义为 7 通道，并填入正确版本的 GLM；它不编译 CUDA。重复执行只重建 `thirdparty/_build/` 下的产物。

## Ubuntu 安装

先确认远端 GPU 型号、驱动、显存以及 `nvcc --version`。完整 CUDA Toolkit 是编译前提，PyTorch CUDA wheel 不能替代 Toolkit。环境中的 `CUDA_HOME` 应指向相应 Toolkit。

```bash
conda env create -f environment.yml
conda activate rogs
python tools/install.py --torch-profile upstream-cu117 --segmentation
```

这里的 `upstream-cu117` 保留 RoGS README 的 Torch 1.13.1/cu117。README 同时列出的 PyTorch3D 0.7.8 与其官方支持范围不一致；这是**待远端验证配方**，详情见 [环境说明](docs/ENVIRONMENT.md)。另提供 `supported-cu118`（Torch 2.1.2/cu118），也需要独立编译与对照，不能静默混用。

安装器按顺序安装包、校验源码、生成扩展、在构建副本里编译 PyTorch3D/GS/Detectron2/MSDeformAttn，最后运行：

```bash
rogs doctor --smoke --segmentation
```

这会检查 RGB/7 类语义的透视与正交渲染及反向传播。Mac 上不能完成该 CUDA 验证。若已有正确环境，可以直接运行 `python tools/build_extensions.py --segmentation`。

## nuScenes 数据配置

```bash
export NUSC_ROOT=/data/nuscenes
export NUSC_SEG_ROOT=/data/nuscenes_seg
export NUSC_GROUND_ROOT=/data/rogs_preprocessed/nuScenes_road_gt
rogs check --config configs/nuscenes.yaml --stage images
```

`NUSC_ROOT` 包含 `v1.0-trainval/`、`samples/`、`sweeps/`。图像会按原版同时读取关键帧与 sweeps，不能只准备 2 Hz 关键帧。默认只选 `scene-0655`；现有全量 trainval 足够，无需重新下载 mini。

`NUSC_SEG_ROOT` 是**原始 65 类 ID 的单通道 PNG**，例如 `samples/seg_CAM_FRONT/<原图同名>.png`、`sweeps/seg_CAM_FRONT/...`，不是彩色可视化，也不是已经压成 7 类的标签。

### A：用原版标签对齐

使用上游 README 链接的 [RoMe 预计算标签](https://github.com/DRosemei/RoMe)。保持原标签是首个参考基线的推荐路径。

```bash
rogs check --config configs/nuscenes.yaml --stage labels
rogs preprocess --config configs/nuscenes.yaml
rogs check --config configs/nuscenes.yaml --stage train
rogs train --config configs/nuscenes.yaml
```

预处理封装调用真实的 `process_nusc.py`，修正了上游文档中模块名和参数横线的拼写；预处理算法没有改动。训练前严格检查所选图像/标签/道路 PLY 是否存在，避免原版静默丢帧让结果无法比较。

### B：在工程内部生成分割标签

分割源码及模型配置已在 `thirdparty/Mask2Former/`。当前能核实的是 Mapillary 65 类空间；**RoGS/RoMe 实际用的 checkpoint 和完整推理配置未公开核实**，所以本工程不会宣称任意同名模型产生的标签与论文相同。

提供官方 Swin-L Mapillary semantic 模型作为独立配置。权重存到 `thirdparty/weights/`，下载器验证官方 MD5 前缀并记录完整 SHA256：

```bash
python tools/fetch_weights.py --model mapillary-swin-l-semantic
export NUSC_SEG_ROOT=/data/rogs_generated_seg
rogs segment --config configs/nuscenes.yaml \
  --model-profile configs/segmentation/mapillary_swin_l.yaml
```

它调用本地 Mask2Former 的 `DefaultPredictor`，导出原图大小的 65 类 argmax PNG；随后仍由未修改的 RoGS 做 mask、7 类 remap、resize 与 crop。默认单尺度、无 TTA。新标签用独立目录；目录内记录 checkpoint、SHA256、模型配置、处理状态，不混入不同模型或未追踪的原标签。已有同一来源的合法文件可跳过继续处理。

分割是离线预处理，同一套标签供后续多次训练复用。再次运行会先校验来源、标签尺寸及类别范围，只推理缺失帧；全部命中时不会加载模型或初始化 GPU。缓存检查仍需读取权重以核对 SHA256、读取原图头和解码标签 PNG，但不再完整解码已缓存帧的 JPEG。缓存损坏会报错，需移走损坏文件后再补算。

当前候选 Swin-L 配置的测试尺寸上下限均为 2048，1600×900 输入实际缩放到约 2048×1152，且逐张推理。首次生成标签仍可能很慢，尚无本项目远端实测速度。控制台及 `segmentation_manifest.json` 的 `last_run` 记录缓存/新处理帧数、预检查、模型初始化、读图、推理、写标签及总耗时；`prediction_seconds` 包含 argmax 和回传 CPU，`end_to_end_images_per_second` 是本次新处理帧数除以总耗时（包括预检查、初始化）。中断后再次运行可复用已完整写出的标签。没有启用 AMP、降分辨率或更换模型，以免引入额外的标签差异。

## 算法对齐与输出

```bash
python tools/verify_algorithm.py
rogs baseline --config configs/nuscenes.yaml
rogs train --config configs/nuscenes.yaml
python tools/compare_runs.py outputs/<baseline-run-id> outputs/<packaged-run-id>
```

`baseline` 在独立进程执行未修改的上游源码；两次使用相同配置和输入。对比器检查 checkpoint/曝光参数和最终 RGB/语义/mask 像素，失败时报告差异。CUDA 非确定性和环境变化需要调查，不能为了通过而随意放宽阈值。

每次运行单独生成 `outputs/<UTC时间-随机ID>/`，保存解析后的配置、源码与模型来源记录，内部保留原版输出目录及文件名。算法参数沿用 `local_nusc.yaml`；工程示例仅改变数据路径、所选场景和 `train.save=True`。

读取源码发现的损失归一化、语义梯度截断、GT 颜色值等问题均**保留**，记录在 [算法边界](docs/ALGORITHM_ALIGNMENT.md)。本阶段不做算法修正、类别变更或自采适配。

## 来源与许可

源码提交和树校验值见 `thirdparty/sources.lock.json`。RoGS 根目录许可不覆盖全部第三方代码和模型；每个依赖保留自己的 LICENSE，模型权重许可见 Mask2Former 模型表。相关入口见 [第三方说明](thirdparty/README.md)。
