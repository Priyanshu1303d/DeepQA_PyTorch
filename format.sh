#!/bin/bash

echo "🔧 Running isort..."
isort .

echo "🖋️ Running black..."
black .

echo "📋 Optionally running flake8 (if installed)..."
if command -v flake8 &> /dev/null; then
    flake8 .
else
    echo "flake8 not found. Skipping."
fi
