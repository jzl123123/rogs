# 第三方源码

这里实际保存公开源码，来源与提交见 `sources.lock.json`。不要直接编辑；如果确需兼容补丁，应记录补丁、原因和验证结果，并与参考算法修改区分。

- `rogs_upstream`：未修改的 RoGS 基线。
- `diff-gs-depth-alpha`：官方定制 GS 栅格器，支持 RGB、depth、alpha、透视与正交投影。
- `glm`：该栅格器实际绑定的 GLM gitlink。构建时复制到 `third_party/glm`。
- `Mask2Former`：完整分割模型代码、Mapillary 配置与模型表。
- `detectron2`：v0.6，Mask2Former 的框架。
- `pytorch3d`：V0.7.8 的完整 `pytorch3d/` 包和根目录构建/许可文件；未打包上游示例媒体、文档网站和测试资源。RoGS 使用的近邻、点云渲染和旋转代码保持原样，稀疏目录规则记入来源锁。
- `_build`：自动生成且可重建的源码副本；构建产物不纳入源校验。
- `weights`：本地权重文件位置。

校验：`python tools/fetch_thirdparty.py --verify`。若目录缺失，可以运行同一命令去掉 `--verify`，按锁定提交恢复；已有且改动的目录不会被覆盖。

许可保留在各目录：RoGS Apache-2.0，Mask2Former MIT（部分代码另有许可），Detectron2 Apache-2.0，PyTorch3D BSD；GS 使用其 `LICENSE.md`。模型权重另遵循 Mask2Former `MODEL_ZOO.md` 所列条款，不能用工程根目录 LICENSE 替代。
