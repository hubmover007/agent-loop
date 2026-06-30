#!/bin/bash
set -e

# Agent-Loop 一键安装脚本
# 用法: curl -fsSL https://raw.githubusercontent.com/hubmover007/agent-loop/master/install.sh | bash
# 或: git clone https://github.com/hubmover007/agent-loop.git && cd agent-loop && bash install.sh

echo "🐕 Agent-Loop 安装脚本"
echo "========================="

# 1. 检查 Python 版本
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 python3，请先安装 Python 3.12+"
    exit 1
fi
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python $PY_VERSION"

# 2. 创建虚拟环境
VENV_DIR="${1:-.venv}"
echo "📦 创建虚拟环境: $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# 3. 升级 pip
echo "⬆️  升级 pip..."
pip install --upgrade pip setuptools wheel -q

# 4. 安装依赖
echo "📥 安装依赖..."
pip install -e ".[dev]" -q

# 5. 初始化配置
echo "⚙️  初始化配置..."
python3 -m src.cli init-config

# 6. 创建必要目录
mkdir -p logs state/agents

# 7. 运行测试验证
echo "🧪 运行测试..."
python3 -m pytest tests/ -q 2>&1 | tail -3

# 8. 完成
echo ""
echo "✅ 安装完成！"
echo ""
echo "🚀 启动系统:"
echo "   source $VENV_DIR/bin/activate"
echo "   agent-loop start"
echo ""
echo "📖 其他命令:"
echo "   agent-loop chat \"你的问题\"    # 单次对话"
echo "   agent-loop serve --port 8000   # 启动 API 服务"
echo "   agent-loop status              # 查看系统状态"
echo "   make test                      # 运行测试"
echo "   make clean                     # 清理缓存"
