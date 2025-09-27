#!/bin/bash
# ===========================================
# Script de despliegue completo para MCP Server AI en Kubernetes
# Despliega todos los componentes en el orden correcto
# ===========================================

set -e  # Salir si cualquier comando falla

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes con colores
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Función para verificar si kubectl está disponible
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl no está instalado o no está en el PATH"
        exit 1
    fi
    print_success "kubectl está disponible"
}

# Función para verificar conexión al cluster
check_cluster() {
    if ! kubectl cluster-info &> /dev/null; then
        print_error "No se puede conectar al cluster de Kubernetes"
        exit 1
    fi
    print_success "Conexión al cluster establecida"
}

# Función para esperar a que un deployment esté listo
wait_for_deployment() {
    local deployment=$1
    local namespace=$2
    local timeout=${3:-300}  # 5 minutos por defecto
    
    print_status "Esperando a que el deployment $deployment esté listo..."
    
    if kubectl wait --for=condition=available --timeout=${timeout}s deployment/$deployment -n $namespace; then
        print_success "Deployment $deployment está listo"
    else
        print_error "Timeout esperando el deployment $deployment"
        return 1
    fi
}

# Función para esperar a que un pod esté listo
wait_for_pod() {
    local label_selector=$1
    local namespace=$2
    local timeout=${3:-300}  # 5 minutos por defecto
    
    print_status "Esperando a que los pods con selector $label_selector estén listos..."
    
    if kubectl wait --for=condition=ready --timeout=${timeout}s pod -l $label_selector -n $namespace; then
        print_success "Pods están listos"
    else
        print_error "Timeout esperando los pods"
        return 1
    fi
}

# Función para aplicar un manifest y verificar
apply_manifest() {
    local file=$1
    local description=$2
    
    print_status "Aplicando $description..."
    
    if kubectl apply -f $file; then
        print_success "$description aplicado correctamente"
    else
        print_error "Error aplicando $description"
        return 1
    fi
}

# Función principal de despliegue
deploy_mcp_server() {
    print_status "==================================="
    print_status "Iniciando despliegue de MCP Server AI"
    print_status "==================================="
    
    # Verificar prerrequisitos
    check_kubectl
    check_cluster
    
    # Paso 1: Crear namespace
    print_status "Paso 1: Creando namespace..."
    apply_manifest "namespace.yaml" "Namespace"
    
    # Paso 2: Aplicar RBAC
    print_status "Paso 2: Configurando RBAC..."
    apply_manifest "rbac.yaml" "ServiceAccounts y RBAC"
    
    # Paso 3: Aplicar ConfigMaps y Secrets
    print_status "Paso 3: Aplicando configuración..."
    apply_manifest "configmap.yaml" "ConfigMaps"
    
    print_warning "IMPORTANTE: Asegúrate de haber actualizado los secrets con valores reales"
    read -p "¿Has configurado los secrets correctamente? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Por favor configura los secrets antes de continuar"
        exit 1
    fi
    
    apply_manifest "secret.yaml" "Secrets"
    
    # Paso 4: Desplegar PostgreSQL
    print_status "Paso 4: Desplegando PostgreSQL..."
    apply_manifest "postgres.yaml" "PostgreSQL"
    wait_for_deployment "postgres-deployment" "mcp-server" 600
    wait_for_pod "app=postgres" "mcp-server" 300
    
    # Paso 5: Desplegar Redis
    print_status "Paso 5: Desplegando Redis..."
    apply_manifest "redis.yaml" "Redis"
    wait_for_deployment "redis-deployment" "mcp-server" 300
    wait_for_pod "app=redis" "mcp-server" 180
    
    # Paso 6: Desplegar aplicación principal
    print_status "Paso 6: Desplegando MCP Server AI..."
    apply_manifest "deployment.yaml" "MCP Server Deployment"
    wait_for_deployment "mcp-server-deployment" "mcp-server" 600
    wait_for_pod "app=mcp-server-ai" "mcp-server" 300
    
    # Paso 7: Aplicar servicios
    print_status "Paso 7: Configurando servicios..."
    apply_manifest "service.yaml" "Services"
    
    # Paso 8: Configurar HPA y PDB
    print_status "Paso 8: Configurando escalado automático..."
    apply_manifest "hpa-pdb.yaml" "HPA y PodDisruptionBudget"
    
    # Paso 9: Configurar Ingress
    print_status "Paso 9: Configurando Ingress..."
    apply_manifest "ingress.yaml" "Ingress"
    
    # Verificar estado final
    print_status "==================================="
    print_status "Verificando estado del despliegue..."
    print_status "==================================="
    
    # Mostrar estado de los pods
    print_status "Estado de los pods:"
    kubectl get pods -n mcp-server -o wide
    
    # Mostrar estado de los servicios
    print_status "Estado de los servicios:"
    kubectl get services -n mcp-server
    
    # Mostrar estado del ingress
    print_status "Estado del ingress:"
    kubectl get ingress -n mcp-server
    
    # Mostrar estado del HPA
    print_status "Estado del HPA:"
    kubectl get hpa -n mcp-server
    
    # Verificar health checks
    print_status "Verificando health checks..."
    
    # Obtener IP del servicio
    SERVICE_IP=$(kubectl get service mcp-server-service -n mcp-server -o jsonpath='{.spec.clusterIP}')
    
    if [ ! -z "$SERVICE_IP" ]; then
        print_status "Probando health check en $SERVICE_IP:8090/health"
        
        # Crear un pod temporal para hacer la prueba
        kubectl run test-pod --rm -i --tty --image=curlimages/curl --restart=Never -- \
            curl -f http://$SERVICE_IP:8090/health || print_warning "Health check falló"
    fi
    
    print_success "==================================="
    print_success "Despliegue completado exitosamente!"
    print_success "==================================="
    
    # Mostrar información útil
    print_status "Información útil:"
    echo "- Namespace: mcp-server"
    echo "- Pods: kubectl get pods -n mcp-server"
    echo "- Logs: kubectl logs -f deployment/mcp-server-deployment -n mcp-server"
    echo "- Port-forward: kubectl port-forward service/mcp-server-service 8090:8090 -n mcp-server"
    echo "- Scaling: kubectl scale deployment mcp-server-deployment --replicas=5 -n mcp-server"
    
    # Mostrar URLs de acceso si hay ingress
    INGRESS_IP=$(kubectl get ingress mcp-server-ingress -n mcp-server -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    if [ ! -z "$INGRESS_IP" ]; then
        echo "- API URL: http://$INGRESS_IP/api/v1"
        echo "- WebSocket URL: ws://$INGRESS_IP/ws"
    fi
}

# Función para limpiar el despliegue
cleanup_deployment() {
    print_status "==================================="
    print_status "Limpiando despliegue de MCP Server AI"
    print_status "==================================="
    
    print_warning "Esto eliminará todos los recursos de MCP Server AI"
    read -p "¿Estás seguro? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Operación cancelada"
        exit 0
    fi
    
    # Eliminar en orden inverso
    kubectl delete -f ingress.yaml --ignore-not-found=true
    kubectl delete -f hpa-pdb.yaml --ignore-not-found=true
    kubectl delete -f service.yaml --ignore-not-found=true
    kubectl delete -f deployment.yaml --ignore-not-found=true
    kubectl delete -f redis.yaml --ignore-not-found=true
    kubectl delete -f postgres.yaml --ignore-not-found=true
    kubectl delete -f secret.yaml --ignore-not-found=true
    kubectl delete -f configmap.yaml --ignore-not-found=true
    kubectl delete -f rbac.yaml --ignore-not-found=true
    kubectl delete -f namespace.yaml --ignore-not-found=true
    
    print_success "Limpieza completada"
}

# Función para mostrar el estado actual
show_status() {
    print_status "==================================="
    print_status "Estado actual de MCP Server AI"
    print_status "==================================="
    
    if ! kubectl get namespace mcp-server &> /dev/null; then
        print_warning "El namespace mcp-server no existe"
        return 1
    fi
    
    echo
    print_status "Pods:"
    kubectl get pods -n mcp-server -o wide
    
    echo
    print_status "Services:"
    kubectl get services -n mcp-server
    
    echo
    print_status "Deployments:"
    kubectl get deployments -n mcp-server
    
    echo
    print_status "HPA:"
    kubectl get hpa -n mcp-server
    
    echo
    print_status "PVC:"
    kubectl get pvc -n mcp-server
    
    echo
    print_status "Ingress:"
    kubectl get ingress -n mcp-server
}

# Función para mostrar logs
show_logs() {
    local component=${1:-mcp-server}
    
    case $component in
        "mcp-server"|"app")
            kubectl logs -f deployment/mcp-server-deployment -n mcp-server
            ;;
        "redis")
            kubectl logs -f deployment/redis-deployment -n mcp-server
            ;;
        "postgres")
            kubectl logs -f deployment/postgres-deployment -n mcp-server
            ;;
        *)
            print_error "Componente no válido. Opciones: mcp-server, redis, postgres"
            exit 1
            ;;
    esac
}

# Función para hacer port-forward
port_forward() {
    local service=${1:-mcp-server-service}
    local local_port=${2:-8090}
    local remote_port=${3:-8090}
    
    print_status "Iniciando port-forward para $service..."
    print_status "Acceso local: http://localhost:$local_port"
    print_status "Presiona Ctrl+C para detener"
    
    kubectl port-forward service/$service $local_port:$remote_port -n mcp-server
}

# Función para ejecutar tests
run_tests() {
    print_status "==================================="
    print_status "Ejecutando tests de MCP Server AI"
    print_status "==================================="
    
    # Verificar que el servicio esté disponible
    SERVICE_IP=$(kubectl get service mcp-server-service -n mcp-server -o jsonpath='{.spec.clusterIP}')
    
    if [ -z "$SERVICE_IP" ]; then
        print_error "No se pudo obtener la IP del servicio"
        exit 1
    fi
    
    # Crear pod de test temporal
    print_status "Creando pod de test..."
    
    kubectl run mcp-test-pod --rm -i --tty --image=curlimages/curl --restart=Never -- /bin/sh -c "
        echo 'Probando health check...'
        curl -f http://$SERVICE_IP:8090/health || exit 1
        
        echo 'Probando listado de modelos...'
        curl -f http://$SERVICE_IP:8090/api/v1/models || exit 1
        
        echo 'Probando generación de contenido...'
        curl -f -X POST http://$SERVICE_IP:8090/api/v1/generate \
            -H 'Content-Type: application/json' \
            -d '{\"prompt\": \"Test\", \"model\": \"gpt-4o\", \"provider\": \"azure\", \"maxTokens\": 10}' || exit 1
        
        echo 'Todos los tests pasaron exitosamente!'
    "
    
    if [ $? -eq 0 ]; then
        print_success "Todos los tests pasaron"
    else
        print_error "Algunos tests fallaron"
        exit 1
    fi
}

# Función de ayuda
show_help() {
    echo "Uso: $0 [COMANDO]"
    echo
    echo "Comandos disponibles:"
    echo "  deploy     - Despliega MCP Server AI completo"
    echo "  cleanup    - Elimina todos los recursos"
    echo "  status     - Muestra el estado actual"
    echo "  logs       - Muestra logs [componente]"
    echo "  forward    - Port-forward [servicio] [puerto-local] [puerto-remoto]"
    echo "  test       - Ejecuta tests básicos"
    echo "  help       - Muestra esta ayuda"
    echo
    echo "Ejemplos:"
    echo "  $0 deploy"
    echo "  $0 logs mcp-server"
    echo "  $0 forward mcp-server-service 8090 8090"
    echo "  $0 test"
}

# Script principal
main() {
    case "${1:-deploy}" in
        "deploy")
            deploy_mcp_server
            ;;
        "cleanup")
            cleanup_deployment
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs $2
            ;;
        "forward")
            port_forward $2 $3 $4
            ;;
        "test")
            run_tests
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Comando no válido: $1"
            show_help
            exit 1
            ;;
    esac
}

# Ejecutar función principal con todos los argumentos
main "$@"