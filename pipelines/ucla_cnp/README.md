# UCLA CNP 预处理流水线

本包包含可复现的 UCLA CNP 神经影像处理工作流。

- `paths.py`：共享的仓库、数据集、fMRIPrep 及 BD-Core20 路径。
- `download.sh`：原始 BIDS 数据和表型信息下载工作流。
- `build_bd_core20.py`：基于 AAL3 构建 BD-Core20 图谱并验证。
- `extract_bd_core20.py`：详尽的 BD-Core20 时间序列提取与受试者质量控制。
- `compare_preprocessing.py`：去噪前后的时间序列与频谱质量控制对比。
- `render_bd_core20.py`：三张出版物风格的三维 ROI 视图，带有合并双侧图例。
- `preprocess.py`：fMRIPrep 调度与标准 ROI 提取入口。

推荐模块调用方式：

```bash
python -m pipelines.ucla_cnp.preprocess --all
python -m pipelines.ucla_cnp.extract_bd_core20 --group HC
python -m pipelines.ucla_cnp.compare_preprocessing --subject 10159 --group HC
python -m pipelines.ucla_cnp.render_bd_core20
```

所有默认路径均从仓库根目录解析，与当前工作目录无关。
