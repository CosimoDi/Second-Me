#!/bin/bash
# WSL 代理配置脚本
# 使用方法: 
#   bash setup_wsl_proxy.sh [IP] [端口]
#   例如: bash setup_wsl_proxy.sh 192.168.2.101 16780

# 从命令行参数获取IP和端口，如果没有提供则使用默认值
PROXY_HOST="${1:-192.168.2.101}"
PROXY_PORT="${2:-16780}"
PROXY_URL="http://${PROXY_HOST}:${PROXY_PORT}"

echo "正在配置 WSL 代理..."
echo "代理地址: ${PROXY_URL}"

# 创建代理配置文件
cat > ~/.bash_proxy << EOF
# WSL 代理配置
export http_proxy=${PROXY_URL}
export https_proxy=${PROXY_URL}
export HTTP_PROXY=${PROXY_URL}
export HTTPS_PROXY=${PROXY_URL}
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY=localhost,127.0.0.1,::1
EOF

# 备份 .bashrc
if [ ! -f ~/.bashrc.bak ]; then
    cp ~/.bashrc ~/.bashrc.bak
    echo "✓ 已备份 ~/.bashrc 到 ~/.bashrc.bak"
fi

# 添加到 .bashrc
if ! grep -q "source.*bash_proxy" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# 加载代理配置" >> ~/.bashrc
    echo "if [ -f ~/.bash_proxy ]; then" >> ~/.bashrc
    echo "    source ~/.bash_proxy" >> ~/.bashrc
    echo "fi" >> ~/.bashrc
    echo "✓ 已添加代理配置到 ~/.bashrc"
else
    echo "✓ 代理配置已存在于 ~/.bashrc"
fi

# 为 zsh 用户也添加配置
if [ -f ~/.zshrc ]; then
    if ! grep -q "source.*bash_proxy" ~/.zshrc; then
        echo "" >> ~/.zshrc
        echo "# 加载代理配置" >> ~/.zshrc
        echo "if [ -f ~/.bash_proxy ]; then" >> ~/.zshrc
        echo "    source ~/.bash_proxy" >> ~/.zshrc
        echo "fi" >> ~/.zshrc
        echo "✓ 已添加代理配置到 ~/.zshrc"
    fi
fi

# 立即应用配置
source ~/.bash_proxy

echo ""
echo "=== 代理配置完成 ==="
echo "当前代理设置:"
env | grep -i proxy | grep -v LESS
echo ""
echo "提示："
echo "1. 请重新打开 WSL 终端使配置生效"
echo "2. 或者执行: source ~/.bashrc"
echo ""
echo "测试代理连接："
echo "curl -I https://www.google.com"
