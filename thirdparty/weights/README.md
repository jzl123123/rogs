# 模型权重

大权重不提交到源码库。运行 `python tools/fetch_weights.py --model mapillary-swin-l-semantic` 将官方权重下载到此目录，并生成来源与 SHA256 记录。模型配置位于 `configs/segmentation/mapillary_swin_l.yaml`。

当前 RoGS/RoMe 原标签所用的精确 checkpoint 未核实。本目录的候选模型与论文原模型不可直接画等号；首个参考运行请使用 RoMe 预计算标签。权重许可与源码许可分别查看 Mask2Former 模型表和 LICENSE。
