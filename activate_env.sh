#!/bin/bash

# 世界模型学习项目 - 虚拟环境激活脚本
# 用法: source activate_env.sh

PROJECT_ROOT="/Users/gelin/Desktop/store/dev/python/3.10/WorldModel"

# 检查是否在项目目录
if [ "$PWD" != "$PROJECT_ROOT" ]; then
    echo "🔄 切换到项目目录..."
    cd "$PROJECT_ROOT" || exit
fi

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在！请先运行: python3 -m venv venv"
    return 1
fi

# 激活虚拟环境
echo "🐍 激活虚拟环境..."
source venv/bin/activate

# 验证 Python 版本
PYTHON_VERSION=$(python --version)
echo "✅ $PYTHON_VERSION"

# 显示已安装的核心包
echo ""
echo "📦 核心依赖包："
pip list | grep -E "numpy|matplotlib|opencv-python|pillow|torch" | head -5

echo ""
echo "🎯 虚拟环境已激活！"
echo "💡 现在可以运行代码了，例如:"
echo "   python code/iteration_01/spatial_concept.py"
