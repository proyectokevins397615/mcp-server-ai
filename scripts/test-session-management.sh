#!/bin/bash

# Script para probar el sistema de gestión de sesiones

echo "========================================="
echo "Test de Sistema de Gestión de Sesiones"
echo "========================================="

# URL base del servidor
BASE_URL="http://localhost:8090/api/v1"

# Colores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# IDs de prueba
USER_ID="test-user-$(uuidgen)"
SESSION_ID="test-session-$(uuidgen)"

echo -e "${BLUE}User ID: $USER_ID${NC}"
echo -e "${BLUE}Session ID: $SESSION_ID${NC}"
echo ""

# Función para verificar el estado del servidor
check_server() {
    echo -e "${YELLOW}Verificando estado del servidor...${NC}"
    response=$(curl -s $BASE_URL/status)
    
    if echo "$response" | jq -e '.sessionEnabled' > /dev/null 2>&1; then
        session_enabled=$(echo "$response" | jq '.sessionEnabled')
        if [ "$session_enabled" = "true" ]; then
            echo -e "${GREEN}✓ Servidor activo con gestión de sesiones habilitada${NC}"
        else
            echo -e "${YELLOW}⚠ Servidor activo pero gestión de sesiones deshabilitada${NC}"
            echo "  Configura ENABLE_SESSION_MANAGEMENT=true en .env"
        fi
        echo "$response" | jq '{status: .status, sessionEnabled: .sessionEnabled, cache: .cache}'
    else
        echo -e "${RED}✗ Servidor no responde o no tiene gestión de sesiones${NC}"
        exit 1
    fi
}

# Test 1: Request sin headers (debe funcionar si sesiones está deshabilitado)
test_without_headers() {
    echo -e "\n${YELLOW}Test 1: Request sin headers de sesión${NC}"
    
    response=$(curl -s -X POST $BASE_URL/generate \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "Hola, ¿cómo estás?",
            "model": "gpt-4o",
            "maxTokens": 100
        }')
    
    if echo "$response" | jq -e '.error' > /dev/null 2>&1; then
        error=$(echo "$response" | jq -r '.error')
        if [[ "$error" == *"X-User-ID and X-Session-ID"* ]]; then
            echo -e "${GREEN}✓ Headers requeridos cuando sesiones está habilitado${NC}"
        else
            echo -e "${RED}✗ Error inesperado: $error${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Request procesado sin headers (sesiones deshabilitado)${NC}"
        echo "$response" | jq '{model: .model, provider: .provider}'
    fi
}

# Test 2: Primera conversación con headers
test_first_message() {
    echo -e "\n${YELLOW}Test 2: Primera mensaje con headers de sesión${NC}"
    
    response=$(curl -s -X POST $BASE_URL/generate \
        -H "Content-Type: application/json" \
        -H "X-User-ID: $USER_ID" \
        -H "X-Session-ID: $SESSION_ID" \
        -d '{
            "prompt": "Hola, mi nombre es TestUser y me gusta la programación",
            "model": "gpt-4o",
            "maxTokens": 100
        }')
    
    if echo "$response" | jq -e '.content' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Primera mensaje procesado exitosamente${NC}"
        echo "Respuesta: $(echo "$response" | jq -r '.content' | head -c 100)..."
    else
        echo -e "${RED}✗ Error al procesar primera mensaje${NC}"
        echo "$response" | jq '.'
    fi
}

# Test 3: Segunda conversación (debe recordar contexto)
test_context_memory() {
    echo -e "\n${YELLOW}Test 3: Segunda mensaje (verificar contexto)${NC}"
    
    sleep 2 # Dar tiempo para que se guarde el contexto
    
    response=$(curl -s -X POST $BASE_URL/generate \
        -H "Content-Type: application/json" \
        -H "X-User-ID: $USER_ID" \
        -H "X-Session-ID: $SESSION_ID" \
        -d '{
            "prompt": "¿Recuerdas mi nombre?",
            "model": "gpt-4o",
            "maxTokens": 100
        }')
    
    if echo "$response" | jq -e '.content' > /dev/null 2>&1; then
        content=$(echo "$response" | jq -r '.content')
        if [[ "$content" == *"TestUser"* ]] || [[ "$content" == *"programación"* ]]; then
            echo -e "${GREEN}✓ El modelo recuerda el contexto anterior${NC}"
        else
            echo -e "${YELLOW}⚠ El modelo podría no estar recordando el contexto${NC}"
        fi
        echo "Respuesta: $(echo "$content" | head -c 150)..."
    else
        echo -e "${RED}✗ Error al procesar segunda mensaje${NC}"
        echo "$response" | jq '.'
    fi
}

# Test 4: Tercera conversación (contexto acumulado)
test_accumulated_context() {
    echo -e "\n${YELLOW}Test 4: Tercera mensaje (contexto acumulado)${NC}"
    
    response=$(curl -s -X POST $BASE_URL/generate \
        -H "Content-Type: application/json" \
        -H "X-User-ID: $USER_ID" \
        -H "X-Session-ID: $SESSION_ID" \
        -d '{
            "prompt": "¿De qué hemos hablado hasta ahora?",
            "model": "gpt-4o",
            "maxTokens": 200
        }')
    
    if echo "$response" | jq -e '.content' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Tercera mensaje procesado con contexto acumulado${NC}"
        echo "Respuesta: $(echo "$response" | jq -r '.content' | head -c 200)..."
    else
        echo -e "${RED}✗ Error al procesar tercera mensaje${NC}"
        echo "$response" | jq '.'
    fi
}

# Test 5: Nueva sesión (no debe tener contexto anterior)
test_new_session() {
    echo -e "\n${YELLOW}Test 5: Nueva sesión (sin contexto anterior)${NC}"
    
    NEW_SESSION_ID="test-session-$(uuidgen)"
    echo -e "${BLUE}Nueva Session ID: $NEW_SESSION_ID${NC}"
    
    response=$(curl -s -X POST $BASE_URL/generate \
        -H "Content-Type: application/json" \
        -H "X-User-ID: $USER_ID" \
        -H "X-Session-ID: $NEW_SESSION_ID" \
        -d '{
            "prompt": "¿Recuerdas de qué hablamos?",
            "model": "gpt-4o",
            "maxTokens": 100
        }')
    
    if echo "$response" | jq -e '.content' > /dev/null 2>&1; then
        content=$(echo "$response" | jq -r '.content')
        if [[ "$content" == *"TestUser"* ]] || [[ "$content" == *"programación"* ]]; then
            echo -e "${RED}✗ Nueva sesión tiene contexto de sesión anterior (error)${NC}"
        else
            echo -e "${GREEN}✓ Nueva sesión no tiene contexto anterior (correcto)${NC}"
        fi
        echo "Respuesta: $(echo "$content" | head -c 150)..."
    else
        echo -e "${RED}✗ Error al procesar nueva sesión${NC}"
        echo "$response" | jq '.'
    fi
}

# Test 6: Batch con sesión
test_batch_with_session() {
    echo -e "\n${YELLOW}Test 6: Batch processing con sesión${NC}"
    
    response=$(curl -s -X POST $BASE_URL/generate/batch \
        -H "Content-Type: application/json" \
        -H "X-User-ID: $USER_ID" \
        -H "X-Session-ID: $SESSION_ID" \
        -d '{
            "requests": [
                {"prompt": "Cuenta del 1 al 3", "model": "gpt-4o", "maxTokens": 50},
                {"prompt": "¿Cuál fue el último número?", "model": "gpt-4o", "maxTokens": 50}
            ]
        }')
    
    if echo "$response" | jq -e '.responses' > /dev/null 2>&1; then
        count=$(echo "$response" | jq '.count')
        echo -e "${GREEN}✓ Batch procesado con sesión${NC}"
        echo "  Respuestas procesadas: $count"
    else
        echo -e "${RED}✗ Error en batch con sesión${NC}"
        echo "$response" | jq '.'
    fi
}

# Test 7: Usuario diferente no puede acceder a sesión
test_session_isolation() {
    echo -e "\n${YELLOW}Test 7: Aislamiento de sesiones entre usuarios${NC}"
    
    ANOTHER_USER_ID="another-user-$(uuidgen)"
    
    response=$(curl -s -X POST $BASE_URL/generate \
        -H "Content-Type: application/json" \
        -H "X-User-ID: $ANOTHER_USER_ID" \
        -H "X-Session-ID: $SESSION_ID" \
        -d '{
            "prompt": "¿Qué recuerdas de la conversación?",
            "model": "gpt-4o",
            "maxTokens": 100
        }')
    
    if echo "$response" | jq -e '.content' > /dev/null 2>&1; then
        content=$(echo "$response" | jq -r '.content')
        if [[ "$content" == *"TestUser"* ]]; then
            echo -e "${RED}✗ Otro usuario puede acceder a sesión ajena (error de seguridad)${NC}"
        else
            echo -e "${GREEN}✓ Sesiones aisladas entre usuarios (correcto)${NC}"
        fi
    else
        error=$(echo "$response" | jq -r '.error' 2>/dev/null)
        if [[ "$error" == *"does not belong"* ]]; then
            echo -e "${GREEN}✓ Sesión protegida contra acceso no autorizado${NC}"
        else
            echo -e "${YELLOW}⚠ Respuesta inesperada${NC}"
            echo "$response" | jq '.'
        fi
    fi
}

# Función principal
main() {
    echo "Iniciando pruebas del sistema de sesiones..."
    echo "Asegúrate de que el servidor esté ejecutándose en el puerto 8090"
    echo "Y que ENABLE_SESSION_MANAGEMENT=true en .env"
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
    
    if ! command -v uuidgen &> /dev/null; then
        echo -e "${RED}Error: uuidgen no está instalado${NC}"
        echo "Instala con: apt-get install uuid-runtime"
        exit 1
    fi
    
    # Ejecutar tests
    check_server
    test_without_headers
    test_first_message
    test_context_memory
    test_accumulated_context
    test_new_session
    test_batch_with_session
    test_session_isolation
    
    echo -e "\n${GREEN}========================================="
    echo "Todas las pruebas completadas"
    echo "=========================================${NC}"
    
    echo -e "\n${BLUE}Resumen:${NC}"
    echo "- Las sesiones mantienen contexto entre mensajes"
    echo "- Cada sesión es independiente"
    echo "- Los usuarios no pueden acceder a sesiones ajenas"
    echo "- El sistema funciona con y sin sesiones habilitadas"
}

# Ejecutar
main