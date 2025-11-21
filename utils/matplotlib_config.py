"""
Matplotlib 中文字体配置模块

自动检测操作系统并配置合适的中文字体，
确保所有图形输出都能正常显示中文。

用法:
    from utils.matplotlib_config import setup_chinese_font
    setup_chinese_font()
"""

import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_chinese_font(verbose=True):
    """
    配置 matplotlib 的中文字体支持

    自动检测操作系统并选择合适的中文字体：
    - macOS: Arial Unicode MS
    - Windows: SimHei (黑体) 或 Microsoft YaHei (微软雅黑)
    - Linux: WenQuanYi Micro Hei (文泉驿微米黑)

    Args:
        verbose: bool, 是否打印配置信息，默认 True

    Returns:
        str: 配置的字体名称
    """
    system = platform.system()
    font_name = None

    if system == 'Darwin':  # macOS
        font_name = 'Arial Unicode MS'
        plt.rcParams['font.sans-serif'] = [font_name]

    elif system == 'Windows':
        # 尝试多个常见字体
        for font in ['SimHei', 'Microsoft YaHei', 'KaiTi']:
            if font in [f.name for f in fm.fontManager.ttflist]:
                font_name = font
                plt.rcParams['font.sans-serif'] = [font_name]
                break
        if not font_name:
            font_name = 'SimHei'
            plt.rcParams['font.sans-serif'] = [font_name]

    else:  # Linux
        # 尝试多个常见字体
        for font in ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']:
            if font in [f.name for f in fm.fontManager.ttflist]:
                font_name = font
                plt.rcParams['font.sans-serif'] = [font_name]
                break
        if not font_name:
            font_name = 'WenQuanYi Micro Hei'
            plt.rcParams['font.sans-serif'] = [font_name]

    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

    if verbose:
        print(f"✅ 已配置 {system} 系统的中文字体: {font_name}")

    return font_name


def list_available_chinese_fonts():
    """
    列出系统中所有可用的中文字体

    Returns:
        list: 包含中文的字体名称列表
    """
    all_fonts = [f.name for f in fm.fontManager.ttflist]

    # 常见的中文字体关键词
    chinese_keywords = [
        'Hei', 'Song', 'Kai', 'FangSong',  # 黑、宋、楷、仿宋
        'SimHei', 'SimSun', 'KaiTi',        # 简体中文
        'Microsoft', 'YaHei',                # 微软雅黑
        'WenQuanYi', 'Noto Sans CJK',       # Linux 字体
        'PingFang', 'Arial Unicode',         # macOS 字体
        'STHeiti', 'STSong', 'STKaiti'      # 华文字体
    ]

    chinese_fonts = []
    for font in all_fonts:
        if any(keyword in font for keyword in chinese_keywords):
            chinese_fonts.append(font)

    return sorted(set(chinese_fonts))


def test_chinese_display():
    """
    测试中文显示是否正常

    创建一个简单的测试图形，包含中文标题和标签
    """
    setup_chinese_font()

    import numpy as np

    # 创建测试数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # 创建图形
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='正弦曲线')
    plt.title('中文显示测试图')
    plt.xlabel('X 轴（横轴）')
    plt.ylabel('Y 轴（纵轴）')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("✅ 中文显示测试完成！")
    print("💡 如果图形中的中文显示正常，说明配置成功。")


if __name__ == '__main__':
    # 运行时直接执行，显示可用字体和测试图形
    print("=" * 50)
    print("Matplotlib 中文字体配置工具")
    print("=" * 50)

    # 显示当前系统
    system = platform.system()
    print(f"\n📍 当前操作系统: {system}")

    # 列出可用的中文字体
    print("\n📚 系统中可用的中文字体:")
    chinese_fonts = list_available_chinese_fonts()
    if chinese_fonts:
        for i, font in enumerate(chinese_fonts[:10], 1):
            print(f"  {i}. {font}")
        if len(chinese_fonts) > 10:
            print(f"  ... 还有 {len(chinese_fonts) - 10} 个字体")
    else:
        print("  ⚠️  未找到中文字体")

    # 配置字体
    print(f"\n🔧 配置中文字体...")
    font = setup_chinese_font()

    # 运行测试
    print(f"\n🧪 运行中文显示测试...")
    test_chinese_display()
