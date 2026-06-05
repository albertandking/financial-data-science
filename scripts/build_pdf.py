"""把全书导出为单个 PDF。

原理：mkdocs-print-site-plugin 把全书合并为一个 HTML 打印页
（site/print_page/index.html），再用无头 Chrome/Edge 打印为 PDF——
浏览器会真实执行 MathJax 渲染公式、加载图片与 Material 样式，
得到与网页一致的高保真 PDF（含封面、目录、全部章节与附录）。

运行：
    uv run python scripts/build_pdf.py
输出：
    金融数据科学.pdf（仓库根目录）
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRINT_PAGE = REPO_ROOT / "site" / "print_page" / "index.html"
OUT_PDF = REPO_ROOT / "金融数据科学.pdf"

# 常见的 Chromium 系浏览器位置（Windows / 跨平台 which）
_WIN_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
_WHICH_NAMES = ["google-chrome", "chromium", "chromium-browser", "chrome", "msedge"]


def find_browser() -> str:
    """定位一个 Chromium 系浏览器（Chrome 或 Edge）。"""
    for path in _WIN_CANDIDATES:
        if Path(path).exists():
            return path
    for name in _WHICH_NAMES:
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("未找到 Chrome / Edge，请安装其一后重试。")


def main() -> None:
    """构建站点并打印全书 PDF。"""
    print("① 构建站点（生成合并打印页）...")
    subprocess.run([sys.executable, "-m", "mkdocs", "build"], cwd=REPO_ROOT, check=True)
    if not PRINT_PAGE.exists():
        raise RuntimeError(f"未生成打印页 {PRINT_PAGE}（请确认 mkdocs.yml 已启用 print-site 插件）")

    browser = find_browser()
    url = PRINT_PAGE.as_uri()
    print(f"② 用无头浏览器打印 PDF：{browser}")
    subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=90000",  # 给 MathJax 充足时间渲染所有公式
            "--run-all-compositor-stages-before-draw",
            f"--print-to-pdf={OUT_PDF}",
            url,
        ],
        check=True,
    )
    size_mb = OUT_PDF.stat().st_size / 1e6
    print(f"完成：{OUT_PDF}（{size_mb:.1f} MB）")


if __name__ == "__main__":
    main()
