# FORGE 论文阅读笔记

这是根据 `add.md` 整理出的 FORGE 中文 LaTeX 阅读笔记项目。原始问答材料保留在 `add.md`，正式笔记位于 `sections/`。

## 项目结构

- `main.tex`：LaTeX 主文件。
- `sections/`：整理后的阅读笔记章节。
- `figures/`：论文图表，包括 `figure1.png`、`table1.png`、`figure2.png`。
- `references.bib`：参考文献占位。
- `add.md`：原始问答式阅读材料。
- `FORGE Fine-grained Multimodal Evaluation for Manufacturing Scenarios*.pdf`：原始论文 PDF。

## 编译方式

推荐使用：

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
