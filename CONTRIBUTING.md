# 贡献与勘误

欢迎为本书纠错、补充与改进！

## 报告勘误 / 提问

发现**错别字、公式错误、代码报错、表述不清**等问题，请到
[Issues](https://github.com/albertandking/financial-data-science/issues) 新建一个 issue，
尽量附上：

- 章节与小节号（如「第7章 7.4.3」）或 notebook 文件名与单元；
- 问题描述；如是代码报错，附上完整报错信息与你的环境（`uv run python --version`）。

## 提交修改（Pull Request）

1. Fork 本仓库并克隆到本地；
2. 安装环境：`uv sync`；生成内置数据：`uv run python scripts/make_sample_data.py`；
3. 修改后本地自检：
   - 代码风格：`uv run ruff check src scripts tests` 与 `uv run ruff format src scripts tests`
   - 测试：`uv run pytest`
   - 若改了 notebook：`uv run jupyter nbconvert --to notebook --execute notebooks/对应文件.ipynb`
   - 若改了正文：`uv run mkdocs build --strict`（确认无断链/缺图）
4. 提交 PR，说明改动内容与动机。

## 写作约定

- 正文用简体中文，引号用中文全角「" "」（代码块、admonition 标题 `!!! x "..."`、HTML 属性中的引号保留 ASCII）；
- 小节统一编号 `X.Y`、`X.Y.Z`；
- 代码遵循 PEP8 + 类型注解（见 `pyproject.toml` 的 `[tool.ruff]`）；
- 正文图示由 `scripts/make_figures.py` 生成（见[附录A](book/appendix/a-setup.md)的图示工作流）。

感谢你的贡献！
