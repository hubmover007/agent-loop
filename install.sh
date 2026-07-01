#!/bin/bash
set -e

# Agent-Loop 一键安装脚本
# 用法: curl -fsSL https://raw.githubusercontent.com/hubmover007/agent-loop/master/install.sh | bash
# 或: git clone https://github.com/hubmover007/agent-loop.git && cd agent-loop && bash install.sh

echo "🐕 Agent-Loop 安装脚本"
echo "========================="

# 1. 检查 Python
if ! command -v python3 &>/dev/null; then
    echo "❌ 未找到 python3，请先安装 Python 3.12+"
    exit 1
fi
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python $PY_VERSION"

# 2. 检查 SurrealDB（可选，提示用户）
if ! docker ps 2>/dev/null | grep -q surrealdb; then
    echo "⚠️  SurrealDB 未运行。将在 setup 步骤中引导启动。"
fi

# 3. 安装方式选择
echo ""
echo "选择安装方式:"
echo "  1) 全局安装 (pip install -e .)"
echo "  2) 虚拟环境 (.venv)"
read -p "选择 [1/2] (默认1): " install_mode
install_mode=${install_mode:-1}

if [ "$install_mode" = "2" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    echo "✅ 虚拟环境 .venv 已创建"
fi

# 4. 安装依赖
echo "📥 安装依赖..."
pip install --upgrade pip setuptools wheel -q
pip install -e ".[dev]" -q

# 5. 注册 agent-loop 命令
echo "✅ agent-loop 命令已注册: $(which agent-loop)"

# 6. 引导 setup
echo ""
echo "接下来运行配置向导:"
echo "  agent-loop setup"
echo ""

# 7. 运行测试验证
echo "🧪 运行快速测试..."
python3 -m pytest tests/ -q --tb=no 2>&1 | tail -3

echo ""
echo "✅ 安装完成！"
echo ""
echo "🚀 下一步:"
echo "  agent-loop setup       # 配置 LLM 和环境"
echo "  agent-loop start       # 启动系统"
echo "  agent-loop chat \"你好\"  # 开始对话"
