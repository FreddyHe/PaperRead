# WildDet3D 论文阅读笔记

这是一个本地 LaTeX 论文阅读笔记项目，适合中文论文精读与归档。

## 项目结构

- `main.tex`：主文档
- `sections/`：按主题拆分的笔记章节
- `references.bib`：参考文献
- `latexmkrc`：本地构建配置
- `figures/`：插图目录
- `WildDet3D.pdf`：原始论文

## 推荐编译方式

### 使用 latexmk

```powershell
latexmk -xelatex main.tex
```

### 手动编译

```powershell
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

## 写作建议

- 在 `01_overview.tex` 里记录论文问题、贡献和背景
- 在 `02_method.tex` 里整理方法结构、公式和模块设计
- 在 `03_experiments.tex` 里提炼实验设置、结果和消融
- 在 `04_reflection.tex` 里写个人思考与可迁移点
