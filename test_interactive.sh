#!/bin/bash
# Interactive testing script for baye-chat

export GOOGLE_API_KEY="AIzaSyAa0KiCkrBLKwblc1SnNlpRf3ohQnK4uic"

# Test 1: Ask about president
echo "Test 1: Asking about US president..."
echo "quem é presidente dos eua?" | uv run baye-chat --mode claim-based 2>&1 | tail -20

echo ""
echo "Press Enter to continue..."
read
