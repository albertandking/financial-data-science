"""统一的绘图配置。

金融图表常需显示中文（标的名称、轴标签），这里集中处理中文字体与样式，
避免每章 notebook 重复设置导致中文乱码。
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# 按操作系统优先尝试的中文字体（Windows / macOS / Linux）
_CJK_FONTS = [
    "Microsoft YaHei",   # Windows 微软雅黑
    "SimHei",            # Windows 黑体
    "PingFang SC",       # macOS
    "Heiti SC",          # macOS
    "Noto Sans CJK SC",  # Linux
    "WenQuanYi Micro Hei",
]


def set_chinese_font() -> None:
    """配置 matplotlib 以正确显示中文与负号。

    在每章 notebook 开头调用一次即可：

        from fds import set_chinese_font
        set_chinese_font()
    """
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for font in _CJK_FONTS:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            break
    # 解决坐标轴负号显示为方块的问题
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (9, 5)
    plt.rcParams["figure.dpi"] = 110
