# Comandos de Prueba para MCP Server AI

## 🔄 Reiniciar el servidor

```bash
# Detener el servidor actual (Ctrl+C) y reiniciar
cd MCP-SERVER-AI
go run cmd/server/main.go -debug
```

## 🧪 Pruebas con Azure OpenAI

### Test 1: Usar modelo por defecto (gpt-4o)

```bash
curl -X POST http://localhost:8090/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Di hola en una línea",
    "provider": "azure",
    "maxTokens": 50,
    "temperature": 0.5,
    "systemPrompt": "Responde de forma breve y directa"
  }' | jq '.'
```

### Test 2: Usar gpt-4o explícitamente

```bash
curl -X POST http://localhost:8090/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "¿Qué es Go?",
    "model": "gpt-4o",
    "provider": "azure",
    "maxTokens": 100,
    "temperature": 0.7
  }' | jq '.'
```

### Test 3: Usar DeepSeek-R1

```bash
curl -X POST http://localhost:8090/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explica Python en 2 líneas",
    "model": "DeepSeek-R1",
    "provider": "azure",
    "maxTokens": 100,
    "temperature": 0.5
  }' | jq '.'
```

### Test 4: Usar gpt-4.1

```bash
curl -X POST http://localhost:8090/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Escribe un hello world en Go",
    "model": "gpt-4.1",
    "provider": "azure",
    "maxTokens": 200,
    "temperature": 0.3
  }' | jq '.'
```

### Test 5: Usar Grok-3

```bash
curl -X POST http://localhost:8090/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Cuéntame algo interesante",
    "model": "grok-3",
    "provider": "azure",
    "maxTokens": 150,
    "temperature": 0.8
  }' | jq '.'
```

## 📋 Verificar modelos disponibles

```bash
curl -X GET http://localhost:8090/api/v1/models | jq '.models[] | select(.provider == "azure-openai") | {id: .id, name: .name, maxTokens: .maxTokens}'
```

## 🔍 Verificar estado del servicio

```bash
curl -X GET http://localhost:8090/api/v1/status | jq '.'
```

## 🏥 Health check

```bash
curl -X GET http://localhost:8090/health | jq '.'
```

## 🐛 Debugging

### Si obtienes error 404:

1. Verifica que el nombre del deployment sea exacto (case sensitive)
2. Verifica en Azure Portal los nombres exactos de tus deployments
3. Actualiza el mapeo en `azure_provider.go` si es necesario

### Si obtienes error de autenticación:

1. Verifica que `AZURE_API_KEY` en `.env` sea correcto
2. Verifica que `AZURE_RESOURCE_NAME` sea correcto

### Ver logs del servidor:

El servidor muestra logs detallados cuando se ejecuta con `-debug`

## 📝 Notas importantes

- Los nombres de deployment en Azure son **case sensitive**
- El modelo por defecto es `gpt-4o` (configurado en `.env`)
- Si no especificas modelo, se usa el por defecto
- Si especificas un modelo, se usa ese exactamente

## 🚀 Ejemplo completo con streaming

```bash
curl -X POST http://localhost:8090/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "prompt": "Escribe un poema corto sobre programación",
    "model": "gpt-4o",
    "provider": "azure",
    "maxTokens": 200,
    "temperature": 0.9
  }'
```

## 🔄 Test rápido de todos los modelos

```bash
#!/bin/bash
models=("gpt-4o" "gpt-4.1" "DeepSeek-R1" "grok-3")

for model in "${models[@]}"; do
  echo "=== Testing $model ==="
  curl -s -X POST http://localhost:8090/api/v1/generate \
    -H "Content-Type: application/json" \
    -d "{
      \"prompt\": \"Di hola\",
      \"model\": \"$model\",
      \"provider\": \"azure\",
      \"maxTokens\": 20
    }" | jq -r '.model + ": " + .content' || echo "Error con $model"
  echo ""
done
```
