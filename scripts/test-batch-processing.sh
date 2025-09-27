#!/bin/bash

# Script para probar el procesamiento en batch con WorkerPool

echo "==================================="
echo "Test de Procesamiento en Batch"
echo "==================================="

# URL base del servidor
BASE_URL="http://localhost:8090/api/v1"

# Colores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para verificar el estado del servidor
check_server() {
    echo -e "${YELLOW}Verificando estado del servidor...${NC}"
    response=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/status)
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ Servidor activo${NC}"
        
        # Obtener detalles del estado
        status=$(curl -s $BASE_URL/status | jq '.')
        echo "Estado del servidor:"
        echo "$status" | jq '{parallel: .parallel, poolStats: .poolStats}'
    else
        echo -e "${RED}✗ Servidor no responde${NC}"
        exit 1
    fi
}

# Test 1: Procesamiento en batch pequeño
test_small_batch() {
    echo -e "\n${YELLOW}Test 1: Batch pequeño (5 requests)${NC}"
    
    batch_request='{
        "requests": [
            {"prompt": "Genera un haiku sobre programación", "model": "gpt-4o", "maxTokens": 100},
            {"prompt": "¿Qué es un WorkerPool?", "model": "gpt-4o", "maxTokens": 200},
            {"prompt": "Explica concurrencia en Go", "model": "gpt-4o", "maxTokens": 300},
            {"prompt": "Ventajas de usar Redis", "model": "gpt-4o", "maxTokens": 150},
            {"prompt": "¿Qué es Prometheus?", "model": "gpt-4o", "maxTokens": 200}
        ]
    }'
    
    start_time=$(date +%s%N)
    response=$(curl -s -X POST $BASE_URL/generate/batch \
        -H "Content-Type: application/json" \
        -d "$batch_request")
    end_time=$(date +%s%N)
    
    duration=$((($end_time - $start_time) / 1000000))
    
    if echo "$response" | jq -e '.responses' > /dev/null 2>&1; then
        count=$(echo "$response" | jq '.count')
        rate=$(echo "$response" | jq '.rate')
        echo -e "${GREEN}✓ Batch procesado exitosamente${NC}"
        echo "  - Respuestas: $count"
        echo "  - Duración: ${duration}ms"
        echo "  - Rate: $rate req/s"
    else
        echo -e "${RED}✗ Error en procesamiento${NC}"
        echo "$response" | jq '.'
    fi
}

# Test 2: Procesamiento en batch mediano
test_medium_batch() {
    echo -e "\n${YELLOW}Test 2: Batch mediano (50 requests)${NC}"
    
    # Generar 50 requests
    requests="["
    for i in {1..50}; do
        requests+="{\"prompt\": \"Genera un número aleatorio y explica su significado: $i\", \"model\": \"gpt-4o\", \"maxTokens\": 100}"
        if [ $i -lt 50 ]; then
            requests+=","
        fi
    done
    requests+="]"
    
    batch_request="{\"requests\": $requests}"
    
    echo "Enviando 50 requests..."
    start_time=$(date +%s%N)
    response=$(curl -s -X POST $BASE_URL/generate/batch \
        -H "Content-Type: application/json" \
        -d "$batch_request")
    end_time=$(date +%s%N)
    
    duration=$((($end_time - $start_time) / 1000000))
    
    if echo "$response" | jq -e '.responses' > /dev/null 2>&1; then
        count=$(echo "$response" | jq '.count')
        rate=$(echo "$response" | jq '.rate')
        echo -e "${GREEN}✓ Batch procesado exitosamente${NC}"
        echo "  - Respuestas: $count"
        echo "  - Duración: ${duration}ms"
        echo "  - Rate: $rate req/s"
    else
        echo -e "${RED}✗ Error en procesamiento${NC}"
        echo "$response" | jq '.'
    fi
}

# Test 3: Batch con streaming SSE
test_batch_streaming() {
    echo -e "\n${YELLOW}Test 3: Batch con streaming (10 requests)${NC}"
    
    batch_request='{
        "requests": [
            {"prompt": "Cuenta del 1 al 5", "model": "gpt-4o", "maxTokens": 50},
            {"prompt": "Lista 3 colores", "model": "gpt-4o", "maxTokens": 50},
            {"prompt": "Nombra 3 lenguajes de programación", "model": "gpt-4o", "maxTokens": 50},
            {"prompt": "Lista días de la semana", "model": "gpt-4o", "maxTokens": 100},
            {"prompt": "Nombra 5 países", "model": "gpt-4o", "maxTokens": 50}
        ]
    }'
    
    echo "Iniciando streaming..."
    
    # Usar timeout para limitar el tiempo de streaming
    timeout 30 curl -s -N -X POST $BASE_URL/generate/batch/stream \
        -H "Content-Type: application/json" \
        -d "$batch_request" | while IFS= read -r line; do
        if [[ $line == data:* ]]; then
            data=${line#data: }
            if echo "$data" | jq -e '.finished' > /dev/null 2>&1; then
                echo -e "${GREEN}✓ Streaming completado${NC}"
                break
            elif echo "$data" | jq -e '.index' > /dev/null 2>&1; then
                index=$(echo "$data" | jq '.index')
                total=$(echo "$data" | jq '.total')
                echo "  Procesado: $((index + 1))/$total"
            fi
        fi
    done
}

# Test 4: Estadísticas del pool
test_pool_stats() {
    echo -e "\n${YELLOW}Test 4: Estadísticas del WorkerPool${NC}"
    
    response=$(curl -s $BASE_URL/pool/stats)
    
    if echo "$response" | jq -e '.pool_stats' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Estadísticas obtenidas${NC}"
        echo "$response" | jq '.'
    else
        echo -e "${RED}✗ No se pudieron obtener estadísticas${NC}"
        echo "$response" | jq '.'
    fi
}

# Test 5: Prueba de carga
test_load() {
    echo -e "\n${YELLOW}Test 5: Prueba de carga (100 requests concurrentes)${NC}"
    
    echo "Enviando 100 requests en paralelo..."
    
    # Crear archivo temporal para requests
    temp_file=$(mktemp)
    
    for i in {1..100}; do
        echo "{\"prompt\": \"Test request $i\", \"model\": \"gpt-4o\", \"maxTokens\": 50}" >> $temp_file
    done
    
    # Ejecutar requests en paralelo usando xargs
    start_time=$(date +%s%N)
    cat $temp_file | xargs -P 10 -I {} curl -s -X POST $BASE_URL/generate \
        -H "Content-Type: application/json" \
        -d {} -o /dev/null
    end_time=$(date +%s%N)
    
    duration=$((($end_time - $start_time) / 1000000))
    
    echo -e "${GREEN}✓ Prueba de carga completada${NC}"
    echo "  - Requests: 100"
    echo "  - Duración total: ${duration}ms"
    echo "  - Rate promedio: $(echo "scale=2; 100000 / $duration" | bc) req/s"
    
    # Limpiar
    rm $temp_file
    
    # Verificar estadísticas del pool después de la carga
    echo -e "\nEstadísticas del pool después de la carga:"
    curl -s $BASE_URL/pool/stats | jq '.pool_stats'
}

# Función principal
main() {
    echo "Iniciando pruebas del WorkerPool..."
    echo "Asegúrate de que el servidor esté ejecutándose en el puerto 8090"
    echo ""
    
    # Verificar dependencias
    if ! command -v curl &> /dev/null; then
        echo -e "${RED}Error: curl no está instalado${NC}"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}Error: jq no está instalado${NC}"
        exit 1
    fi
    
    # Ejecutar tests
    check_server
    test_small_batch
    test_medium_batch
    test_batch_streaming
    test_pool_stats
    test_load
    
    echo -e "\n${GREEN}==================================="
    echo "Todas las pruebas completadas"
    echo "===================================${NC}"
}

# Ejecutar
main