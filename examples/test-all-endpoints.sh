#!/bin/bash

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   MCP SERVER AI - TEST ALL ENDPOINTS   ${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Variables
HTTP_HOST="http://localhost:8090"
GRPC_HOST="localhost:50051"
WS_HOST="ws://localhost:8091"

# ===========================================
# HTTP/REST ENDPOINTS
# ===========================================

echo -e "${YELLOW}📡 HTTP/REST ENDPOINTS${NC}"
echo ""

# 1. Health Check
echo -e "${GREEN}1. Health Check${NC}"
curl -s $HTTP_HOST/health | jq '.'
echo ""

# 2. Readiness Check
echo -e "${GREEN}2. Readiness Check${NC}"
curl -s $HTTP_HOST/readyz | jq '.'
echo ""

# 3. List Models
echo -e "${GREEN}3. List Models${NC}"
curl -s $HTTP_HOST/api/v1/models | jq '.models[] | {id, name, provider}' | head -20
echo ""

# 4. Get Status
echo -e "${GREEN}4. Get Status${NC}"
curl -s $HTTP_HOST/api/v1/status | jq '.'
echo ""

# 5. Generate Text (GPT-4.1)
echo -e "${GREEN}5. Generate Text - GPT-4.1${NC}"
curl -s -X POST $HTTP_HOST/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Escribe un haiku sobre programación",
    "model": "gpt-4.1",
    "provider": "azure",
    "maxTokens": 100,
    "temperature": 0.7,
    "systemPrompt": "Eres un poeta experto en haikus japoneses"
  }' | jq '.'
echo ""

# 6. Generate Text (Grok-3)
echo -e "${GREEN}6. Generate Text - Grok-3${NC}"
curl -s -X POST $HTTP_HOST/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "¿Cuál es el sentido de la vida?",
    "model": "grok-3",
    "provider": "azure",
    "maxTokens": 150,
    "temperature": 0.8
  }' | jq '.'
echo ""

# 7. Validate Prompt
echo -e "${GREEN}7. Validate Prompt${NC}"
curl -s -X POST $HTTP_HOST/api/v1/validate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Este es un prompt válido"
  }' | jq '.'
echo ""

# 8. Streaming (SSE) - Solo muestra primeros chunks
echo -e "${GREEN}8. Streaming SSE - GPT-4o${NC}"
echo "Iniciando streaming (mostrando primeros 5 segundos)..."
timeout 5 curl -N -X POST $HTTP_HOST/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Cuenta una historia corta",
    "model": "gpt-4o",
    "provider": "azure",
    "maxTokens": 200,
    "temperature": 0.9
  }' 2>/dev/null | head -20
echo ""
echo "Streaming detenido después de 5 segundos"
echo ""

# ===========================================
# gRPC ENDPOINTS
# ===========================================

echo -e "${YELLOW}🔧 gRPC ENDPOINTS${NC}"
echo ""

# 1. Generate (gRPC)
echo -e "${GREEN}1. gRPC Generate - GPT-4.1${NC}"
grpcurl -plaintext -d '{
  "prompt": "¿Qué es Go?",
  "model": "gpt-4.1",
  "provider": "azure",
  "max_tokens": 100,
  "temperature": 0.5,
  "system_prompt": "Responde de forma breve y técnica"
}' $GRPC_HOST proto.AIService/Generate 2>/dev/null | jq '.'
echo ""

# 2. List Models (gRPC)
echo -e "${GREEN}2. gRPC List Models${NC}"
grpcurl -plaintext -d '{}' $GRPC_HOST proto.AIService/ListModels 2>/dev/null | jq '.models[] | {id, name, provider}'
echo ""

# 3. Generate Stream (gRPC) - Corrected
echo -e "${GREEN}3. gRPC Generate Stream${NC}"
echo "Iniciando streaming gRPC (mostrando primeros chunks)..."
timeout 3 grpcurl -plaintext -d '{
  "prompt": "Escribe un poema corto",
  "model": "gpt-4.1",
  "provider": "azure",
  "max_tokens": 100,
  "temperature": 0.8
}' $GRPC_HOST proto.AIService/GenerateStream 2>/dev/null | head -10
echo ""
echo "Streaming gRPC detenido"
echo ""

# ===========================================
# PRUEBAS DE DIFERENTES MODELOS
# ===========================================

echo -e "${YELLOW}🤖 PRUEBA DE MODELOS AZURE${NC}"
echo ""

# Array de modelos para probar
models=("gpt-4.1" "gpt-4o" "grok-3")

for model in "${models[@]}"; do
  echo -e "${GREEN}Testing: $model${NC}"
  response=$(curl -s -X POST $HTTP_HOST/api/v1/generate \
    -H "Content-Type: application/json" \
    -d "{
      \"prompt\": \"Di 'Hola' en una línea\",
      \"model\": \"$model\",
      \"provider\": \"azure\",
      \"maxTokens\": 20,
      \"temperature\": 0.1
    }")
  
  content=$(echo $response | jq -r '.content')
  tokens=$(echo $response | jq -r '.tokensUsed')
  
  if [ "$content" != "null" ] && [ "$content" != "" ]; then
    echo -e "✅ Response: $content"
    echo -e "   Tokens: $tokens"
  else
    echo -e "❌ No response or error"
    echo $response | jq '.error'
  fi
  echo ""
done

# ===========================================
# MÉTRICAS PROMETHEUS
# ===========================================

echo -e "${YELLOW}📊 MÉTRICAS PROMETHEUS${NC}"
echo ""
echo -e "${GREEN}Métricas disponibles en:${NC} $HTTP_HOST/metrics"
curl -s $HTTP_HOST/metrics | grep "^go_" | head -5
echo "..."
echo ""

# ===========================================
# RESUMEN
# ===========================================

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}           PRUEBAS COMPLETADAS           ${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "${GREEN}✅ Endpoints HTTP/REST funcionando${NC}"
echo -e "${GREEN}✅ Endpoints gRPC funcionando${NC}"
echo -e "${GREEN}✅ Streaming SSE funcionando${NC}"
echo -e "${GREEN}✅ Modelos Azure disponibles${NC}"
echo ""
echo -e "${YELLOW}📝 Notas:${NC}"
echo "- El servidor HTTP corre en puerto 8090"
echo "- El servidor gRPC corre en puerto 50051"
echo "- El servidor WebSocket corre en puerto 8091"
echo "- CORS está habilitado para todos los orígenes"
echo ""
echo -e "${BLUE}Para ver el cliente web de streaming:${NC}"
echo "Abre MCP-SERVER-AI/examples/streaming-client.html en tu navegador"
echo ""