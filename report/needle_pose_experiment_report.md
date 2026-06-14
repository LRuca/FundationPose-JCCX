# 细长针目标位姿与模型修复实验报告

生成日期：2026-05-04

## 关键结论

- FoundationPose 复现最终可运行，成功输出 25 个位姿矩阵和 24 张跟踪可视化图。
- 原始扫描模型经过扫描清理、去平面背景、轴对称重建、结构化尾部 v1/v2/v3 三轮修正，最终模型为 `C:/Users/lenovo/Desktop/JXCX/model/fixed_unnamed_object_3/needle_structured_tail_reconstruction_v3.stl`。
- YOLOv8n-seg 使用 95 张训练图和 24 张验证图训练 100 epoch；训练中关闭验证，因此 metrics/mAP 曲线为 0，主要依据训练损失下降判断收敛。
- YOLO 分割损失从 2.2786 降至 0.5192，最佳训练分割损失 0.5184 出现在 epoch 98。
- 主要困难集中在 WSL/Conda/CUDA 依赖、本地 Orbbec 帧文件竞争、深度图全零、mask 与有效深度不重合、FoundationPose mesh dtype 不一致。

## 证据文件

- YOLO 曲线：`C:/Users/lenovo/Desktop/JXCX/report/assets/yolo_training_curves.png`
- 模型迭代图：`C:/Users/lenovo/Desktop/JXCX/report/assets/model_iteration_contact_sheet.png`
- 最终 PDF：`C:/Users/lenovo/Desktop/JXCX/report/needle_pose_experiment_report.pdf`

## 错误计数摘要

- Float/Double 类型问题：3
- 等待新帧超时：1
- 深度图全零：3
- mask/depth 有效点不足：9
- 文件复制/替换竞争：11
