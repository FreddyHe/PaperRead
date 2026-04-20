# HY-Embodied-0.5 论文阅读笔记

这是根据 `add.md` 整理出的 HY-Embodied-0.5 中文 LaTeX 阅读笔记项目。原始问答材料仍保留在 `add.md`，整理后的正式笔记位于 `sections/`。

## 项目结构

- `main.tex`：LaTeX 主文件。
- `sections/`：整理后的章节内容。
- `figures/`：论文图示，目前包含 `figure2.png`。
- `references.bib`：参考文献占位。
- `HY-Embodied-0.5 Embodied Foundation Models for Real-World Agents.pdf`：原始论文 PDF。
- `add.md`：原始问答式阅读材料。

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
