#!/bin/bash

# 世界模型学习项目 - 迭代运行脚本
# 用法: ./run_iteration.sh <迭代编号>
# 例如: ./run_iteration.sh 1

PROJECT_ROOT="/Users/gelin/Desktop/store/dev/python/3.10/WorldModel"

# 检查参数
if [ $# -eq 0 ]; then
    echo "❌ 请指定迭代编号！"
    echo "用法: ./run_iteration.sh <迭代编号>"
    echo "例如: ./run_iteration.sh 1"
    exit 1
fi

ITERATION=$1
ITERATION_PADDED=$(printf "%02d" $ITERATION)
ITERATION_DIR="$PROJECT_ROOT/code/iteration_$ITERATION_PADDED"

# 检查迭代目录是否存在
if [ ! -d "$ITERATION_DIR" ]; then
    echo "❌ 迭代 $ITERATION 的目录不存在！"
    echo "路径: $ITERATION_DIR"
    exit 1
fi

# 切换到项目目录
cd "$PROJECT_ROOT" || exit

# 激活虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在！请先运行: python3 -m venv venv"
    exit 1
fi

echo "🐍 激活虚拟环境..."
source venv/bin/activate

# 显示迭代信息
echo ""
echo "=========================================="
echo "🚀 运行迭代 $ITERATION"
echo "=========================================="
echo ""

# 列出该迭代的所有 Python 文件
echo "📁 可用的脚本："
find "$ITERATION_DIR" -name "*.py" -type f | while read -r file; do
    filename=$(basename "$file")
    echo "  - $filename"
done

echo ""
echo "请选择要运行的脚本（输入文件名，不含 .py）:"
read -r script_name

# 运行选定的脚本
SCRIPT_PATH="$ITERATION_DIR/${script_name}.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ 文件不存在: $SCRIPT_PATH"
    exit 1
fi

echo ""
echo "🎯 运行: $script_name.py"
echo "=========================================="
python "$SCRIPT_PATH"

echo ""
echo "=========================================="
echo "✅ 完成！"
echo "=========================================="
