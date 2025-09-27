#!/bin/bash

# Script para iniciar el servidor de benchmarks en tiempo real
# Este servidor muestra los resultados en una interfaz web actualizada en tiempo real

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo "🚀 MCP SERVER AI - BENCHMARKS EN TIEMPO REAL"
echo "================================================"
echo ""

# Verificar si Go está instalado
if ! command -v go &> /dev/null; then
    echo "❌ Error: Go no está instalado"
    exit 1
fi

# Cambiar al directorio del proyecto
cd "$PROJECT_ROOT"

# Instalar dependencias si es necesario
echo "📦 Verificando dependencias..."
if ! go list github.com/gorilla/websocket &>/dev/null 2>&1; then
    echo "📦 Instalando gorilla/websocket..."
    go get -u github.com/gorilla/websocket
fi

# Compilar y ejecutar el servidor
echo ""
echo "🔨 Compilando servidor de benchmarks..."
go build -o scripts/benchmark-server scripts/realtime-benchmark-server.go

echo ""
echo "================================================"
echo "🌐 SERVIDOR DE BENCHMARKS EN TIEMPO REAL"
echo "================================================"
echo ""
echo "📊 Abriendo en el navegador: http://localhost:8095"
echo "⏱️  Los benchmarks comenzarán automáticamente en 2 segundos"
echo ""
echo "Características:"
echo "  ✅ Actualización en tiempo real vía WebSocket"
echo "  ✅ Visualización de métricas mientras se ejecutan"
echo "  ✅ Gráficos y estadísticas dinámicas"
echo "  ✅ Log de ejecución en vivo"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo "================================================"
echo ""

# Intentar abrir en el navegador
if command -v xdg-open &> /dev/null; then
    (sleep 2 && xdg-open http://localhost:8095) &
elif command -v open &> /dev/null; then
    (sleep 2 && open http://localhost:8095) &
fi

# Ejecutar el servidor
./scripts/benchmark-server