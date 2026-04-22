# Demystifying Video Reasoning 论文阅读笔记

这是根据 `add.md` 整理出的中文 LaTeX 阅读笔记项目。由于这篇论文涉及扩散模型、视频 latent、Chain-of-Steps、噪声扰动、突现行为和逐层机制分析，正式笔记拆成 6 个章节并保留较多技术解释。

## 项目结构

- `main.tex`：LaTeX 主文件。
- `sections/`：整理后的详细阅读笔记。
- `figures/`：论文图示，包含 `figure1.png` 到 `figure6.png`。
- `references.bib`：参考文献占位。
- `add.md`：原始问答式阅读材料。
- `Demystifing Video Reasoning.pdf`：原始论文 PDF。

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
