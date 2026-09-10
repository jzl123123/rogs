# 环境与验证状态

目标运行环境：Ubuntu + NVIDIA GPU。当前构建工作在 Mac 完成，无法验证 NVIDIA CUDA 扩展，不将安装脚本当作已经通过的环境锁。

| 配方 | Python | Torch / torchvision | Toolkit 目标 | PyTorch3D |
|---|---|---|---|---|
| upstream-cu117 | 3.8.10 | 1.13.1+cu117 / 0.14.1+cu117 | 11.7 | V0.7.8 源码 |
| supported-cu118 | 3.8.10 | 2.1.2+cu118 / 0.16.2+cu118 | 11.8 | 同一 V0.7.8 源码 |

RoGS README 声称测试过第一组。PyTorch3D [V0.7.8 发布说明](https://github.com/facebookresearch/pytorch3d/releases/tag/V0.7.8) 声明支持 Torch 2.1–2.4。因此保留两组显式配方；没有验证前不指定某组为保证可用。第二组是依赖兼容候选，并不保证与论文旧环境数值一致。

这两套配方也不是对所有 NVIDIA 型号的承诺。尤其是比所列 Toolkit 更新的 GPU 架构，需要先核查驱动、编译器和 Torch 支持；不能通过只修改架构变量假装解决。

`requirements/runtime.txt` 固定直接 Python 依赖，避免旧 Detectron2 遇到 NumPy 1.24 删除别名等明显冲突。它不是完整传递依赖锁：首个远端通过的环境还需保存 `pip freeze`、`pip check`、GPU/驱动/Toolkit 信息。运行 manifest 会记录关键包版本。

源码依赖统一以精确提交保存于 thirdparty。所有 setup.py 都在 `_build` 副本执行，编译结果与原始源码分离。分割的自定义算子是 MSDeformAttn，不可只安装 Mask2Former Python 源码而漏编译它。

## 远端验收顺序

1. `nvidia-smi`、`nvcc --version`、`CUDA_HOME`，确认驱动、Toolkit、实际 GPU。
2. 选择并安装一套配方，记录实际输出，不覆盖另一套实验环境。
3. `rogs doctor --smoke --segmentation`：模块导入和 3/7 通道、透视/正交、前向/反向检查。
4. 用一张图确认分割输出为原尺寸、65 类 ID，并检查地面类别没有错位。
5. 用同一 nuScenes 场景分别运行 baseline 和工程版，对照参数、图片和指标。
6. 记录首个通过版本，之后才考虑依赖升级。

