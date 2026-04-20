# WildDet3D 论文阅读笔记

这是整理后的 WildDet3D 中文阅读笔记。笔记已删去模板占位、重复问答和编译产物，只保留论文理解、方法拆解、训练损失、实验结论与个人思考。

## 项目结构

- `main.tex`：LaTeX 主文件。
- `sections/`：整理后的四个章节。
- `figures/`：论文关键图示。
- `WildDet3D.pdf`：原始论文 PDF。
- `scripts/`：辅助理解 RGB 与深度对齐关系的脚本。

## 编译方式

推荐使用 `latexmk`：

```powershell
latexmk -xelatex main.tex
```

或手动编译：

```powershell
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

## 辅助脚本

`scripts/visualize_rgb_depth_alignment.py` 用于说明 RGB 图像与深度图的像素级对应关系，并演示如何结合相机内参把选定像素反投影到三维空间。

```powershell
python scripts/visualize_rgb_depth_alignment.py --rgb path\to\rgb.png --depth path\to\depth.png --u 500 --v 400 --fx 1000 --fy 1000
```
