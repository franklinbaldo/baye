#!/bin/bash
# run.sh - Script rápido para rodar Baye

set -e

echo "🚀 Baye - Quick Run Script"
echo "=========================="
echo ""

# Verifica API key
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  GOOGLE_API_KEY não configurada"
    echo ""
    echo "Opções:"
    echo "  1. export GOOGLE_API_KEY='sua-chave'"
    echo "  2. source /home/frank/workspace/.envrc"
    echo ""
    read -p "Quer usar a chave do workspace? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        source /home/frank/workspace/.envrc
        export GOOGLE_API_KEY="AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"
    else
        exit 1
    fi
fi

echo "✅ API key configurada"
echo ""

# Verifica uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv não encontrado. Instale com:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv encontrado"
echo ""

# Sync se necessário
if [ ! -d ".venv" ]; then
    echo "📦 Instalando dependências..."
    uv sync
    echo ""
fi

echo "🧠 Rodando exemplo com LLM..."
echo ""

uv run python examples/example_llm_integration.py

echo ""
echo "✅ Exemplo completo!"
echo ""
echo "Próximos passos:"
echo "  - Leia QUICKSTART.md para mais exemplos"
echo "  - Rode: uv run python -i examples/example_llm_integration.py"
echo "  - Explore: examples/ e tests/"
