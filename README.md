# 金融数据科学（Financial Data Science）

面向**中国本科生与研究生**的金融数据科学教材。

- **正文**用 Markdown 编写（`book/` 目录）
- **代码**放在 Jupyter Notebook 中（`notebooks/` 目录），可本地逐格运行
- 正文按需**引用 / 嵌入** notebook 的代码与输出（通过 `scripts/export_notebooks.py` 手动导出）
- 数据：**内置示例数据集**（离线可跑）+ **中国市场接口**（akshare / tushare，联网抓取）
- 环境：**uv** 管理，推荐 **Python 3.14**（兼容 3.11+）；成书：**MkDocs + Material 主题**
- 复现：`uv.lock` 锁定全部依赖精确版本；可读版本表见[附录A](book/appendix/a-setup.md)

---

## 在线访问

- 📖 **在线阅读**（GitHub Pages）：<https://albertandking.github.io/financial-data-science/>
- ▶️ **在线运行代码**：每章正文顶部都有 **Colab** 与 **Binder** 徽章，点开即可在云端运行该章 notebook（无需本地环境）。
  - Colab 会自动克隆本仓库并安装 `fds` 包（notebook 内置「自举单元」，本地运行时自动跳过）；
  - Binder 通过 `binder/` 下的配置自动装好环境与内置数据。

> 网站由 GitHub Actions 自动部署：push 到 `main` → 构建 MkDocs → 发布到 Pages（见 `.github/workflows/deploy.yml`）。
> 首次启用：仓库 **Settings → Pages → Source 选「GitHub Actions」**。

---

## 一、目录结构

```
ds/
├── pyproject.toml          # uv 项目与依赖定义
├── uv.lock                 # 锁定版本（提交入库，保证复现）
├── .python-version         # Python 版本（3.14）
├── mkdocs.yml              # 成书配置与章节导航
├── README.md
│
├── book/                   # 【正文】MkDocs 文档源（Markdown）
│   ├── index.md            #   首页 / 前言
│   ├── assets/figures/     #   正文图示（PNG，由 scripts/make_figures.py 生成）
│   ├── part1/              #   第一部分·基础
│   ├── part2/              #   第二部分·金融计量与统计
│   ├── part3/              #   第三部分·机器学习
│   ├── part4/              #   第四部分·应用与项目
│   └── appendix/           #   附录
│
├── notebooks/              # 【代码】各章可执行 Jupyter Notebook
│   └── ch01_introduction.ipynb
│
├── src/fds/                # 全书复用的工具包（import fds）
│   ├── data.py             #   数据读取/内置数据集加载
│   ├── metrics.py          #   收益、风险等金融指标
│   └── plotting.py         #   统一的中文绘图样式
│
├── data/
│   ├── raw/                #   原始/下载数据（不入库）
│   ├── processed/          #   清洗好的内置示例数据（入库，离线可跑）
│   └── README.md           #   数据说明
│
├── scripts/
│   ├── make_sample_data.py #   离线生成内置示例数据集
│   ├── fetch_data.py       #   从中国市场接口抓取真实数据
│   └── export_notebooks.py #   notebook -> markdown 手动导出
│
└── tests/                  # 冒烟测试，保证代码可跑
```

---

## 二、快速开始

### 1. 安装环境（一次）

```bash
# 安装核心依赖 + 成书工具链（uv 会自动下载 Python 3.14）
uv sync

# 如需联网抓取中国市场数据，额外安装 data 组
uv sync --extra data

# 进阶章节（深度学习/NLP）按需安装
uv sync --extra advanced
```

### 2. 生成内置示例数据（一次，离线）

```bash
uv run python scripts/make_sample_data.py
```

### 3. 运行书中代码

```bash
# 启动 Jupyter，逐章打开 notebooks/ 下的 .ipynb
uv run jupyter lab

# 或命令行执行某个 notebook 验证可跑
uv run jupyter nbconvert --to notebook --execute notebooks/ch01_introduction.ipynb --stdout > /dev/null
```

### 4. 本地预览整本书

```bash
# 把 notebook 的代码与输出导出为 markdown 片段
uv run python scripts/export_notebooks.py

# 启动本地文档服务器，浏览器打开 http://127.0.0.1:8000
uv run mkdocs serve

# 导出静态网站到 site/
uv run mkdocs build
```

---

## 三、写作约定

- **一章 = 一个 `book/partX/NN-题目.md` + 一个 `notebooks/chNN_主题.ipynb`**
- 正文里需要展示代码/图表时，引用 notebook 导出的片段，避免代码两处维护
- 所有 notebook 默认依赖**内置示例数据**，保证**断网也能跑**；用到联网接口的代码放在明确标注的「联网」小节
- 复用逻辑（指标计算、绘图样式、数据加载）抽进 `src/fds`，正文与 notebook 都 `from fds import ...`
