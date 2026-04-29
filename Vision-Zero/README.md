# Vision-Zero 论文阅读笔记

这是根据 `Vision-Zero.pdf` 整理出的中文 LaTeX 阅读笔记项目。正式笔记位于 `sections/`，关键论文页截图位于 `figures/`。

## 项目结构

- `main.tex`：LaTeX 主文件。
- `sections/`：整理后的阅读笔记章节。
- `figures/`：论文图示页截图，包括 `figure1_page.png`、`figure3_page.png`、`figure8_page.png`。
- `references.bib`：参考文献。
- `Vision-Zero.pdf`：原始论文 PDF，本仓库全局忽略 PDF 文件。

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
