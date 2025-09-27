# 🚀 Guía de Despliegue - MCP Server AI

Esta guía completa te ayudará a desplegar MCP Server AI en diferentes plataformas y entornos de manera profesional y segura.

## 📋 Tabla de Contenidos

- [Prerrequisitos](#-prerrequisitos)
- [Configuración Inicial](#-configuración-inicial)
- [Despliegue Local](#-despliegue-local)
- [Despliegue en Docker](#-despliegue-en-docker)
- [Despliegue en Kubernetes](#-despliegue-en-kubernetes)
- [Despliegue en AWS](#-despliegue-en-aws)
- [Despliegue en Azure](#-despliegue-en-azure)
- [Despliegue en Google Cloud](#-despliegue-en-google-cloud)
- [CI/CD](#-cicd)
- [Monitoreo y Logging](#-monitoreo-y-logging)
- [Seguridad](#-seguridad)
- [Troubleshooting](#-troubleshooting)

## 🔧 Prerrequisitos

### Software Requerido

```bash
# Go 1.23 o superior
go version

# Docker y Docker Compose
docker --version
docker-compose --version

# Kubernetes CLI (kubectl)
kubectl version --client

# Helm (para despliegues K8s)
helm version

# Git
git --version
```

### Credenciales Necesarias

- **AWS**: Access Key ID y Secret Access Key con permisos para Bedrock
- **Azure**: API Key y Resource Name para OpenAI
- **Redis**: URL de conexión (opcional para cache)
- **PostgreSQL**: URL de conexión (opcional para persistencia)

## ⚙️ Configuración Inicial

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/mcp-server-ai.git
cd mcp-server-ai
```

### 2. Configurar Variables de Entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar configuración
nano .env
```

### 3. Configuración Mínima Requerida

```env
# Proveedores de IA
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key
AWS_REGION=us-west-2

AZURE_API_KEY=tu_azure_api_key
AZURE_RESOURCE_NAME=tu_resource_name
AZURE_ENDPOINT=https://tu-recurso.openai.azure.com/

# Configuración del servidor
HTTP_PORT=8090
GRPC_PORT=50051
WS_PORT=8091
```

## 🏠 Despliegue Local

### Desarrollo Rápido

```bash
# Instalar dependencias
go mod download

# Generar código protobuf
protoc --go_out=. --go-grpc_out=. internal/proto/ai_service.proto

# Ejecutar en modo desarrollo
go run cmd/server/main.go -debug

# O compilar y ejecutar
go build -o mcp-server cmd/server/main.go
./mcp-server
```

### Con Servicios Locales

```bash
# Iniciar Redis y PostgreSQL localmente
docker run -d --name redis -p 6379:6379 redis:7-alpine
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=mcp_server \
  postgres:16-alpine

# Configurar .env para servicios locales
echo "REDIS_URL=redis://localhost:6379/0" >> .env
echo "POSTGRES_URL=postgres://postgres:password@localhost:5432/mcp_server?sslmode=disable" >> .env

# Ejecutar servidor
./mcp-server
```

## 🐳 Despliegue en Docker

### Docker Compose - Desarrollo

```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f mcp-server

# Escalar servicio
docker-compose up -d --scale mcp-server=3

# Detener servicios
docker-compose down
```

### Docker Compose - Producción

```bash
# Usar configuración de producción
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verificar estado
docker-compose ps

# Backup de datos
docker-compose exec postgres pg_dump -U mcp_user mcp_server > backup.sql
```

### Docker Standalone

```bash
# Build de imagen
docker build -t mcp-server-ai:latest .

# Ejecutar contenedor
docker run -d \
  --name mcp-server \
  -p 8090:8090 \
  -p 8091:8091 \
  -p 50051:50051 \
  --env-file .env \
  mcp-server-ai:latest

# Ver logs
docker logs -f mcp-server
```

## ☸️ Despliegue en Kubernetes

### Despliegue Rápido

```bash
# Navegar al directorio de K8s
cd deploy/k8s

# Hacer ejecutable el script
chmod +x deploy-all.sh

# Desplegar todo
./deploy-all.sh deploy

# Verificar estado
./deploy-all.sh status

# Ver logs
./deploy-all.sh logs mcp-server
```

### Despliegue Manual Paso a Paso

```bash
# 1. Crear namespace
kubectl apply -f namespace.yaml

# 2. Configurar RBAC
kubectl apply -f rbac.yaml

# 3. Aplicar configuración
kubectl apply -f configmap.yaml

# 4. Configurar secrets (IMPORTANTE: actualizar valores reales)
kubectl apply -f secret.yaml

# 5. Desplegar PostgreSQL
kubectl apply -f postgres.yaml

# 6. Desplegar Redis
kubectl apply -f redis.yaml

# 7. Desplegar aplicación principal
kubectl apply -f deployment.yaml

# 8. Configurar servicios
kubectl apply -f service.yaml

# 9. Configurar escalado automático
kubectl apply -f hpa-pdb.yaml

# 10. Configurar ingress
kubectl apply -f ingress.yaml
```

### Verificación del Despliegue

```bash
# Ver todos los recursos
kubectl get all -n mcp-server

# Verificar pods
kubectl get pods -n mcp-server -o wide

# Ver logs
kubectl logs -f deployment/mcp-server-deployment -n mcp-server

# Port-forward para pruebas
kubectl port-forward service/mcp-server-service 8090:8090 -n mcp-server

# Probar API
curl http://localhost:8090/health
```

### Escalado y Mantenimiento

```bash
# Escalar manualmente
kubectl scale deployment mcp-server-deployment --replicas=5 -n mcp-server

# Actualizar imagen
kubectl set image deployment/mcp-server-deployment \
  mcp-server=ghcr.io/tu-usuario/mcp-server-ai:v1.2.0 \
  -n mcp-server

# Rolling restart
kubectl rollout restart deployment/mcp-server-deployment -n mcp-server

# Ver historial de despliegues
kubectl rollout history deployment/mcp-server-deployment -n mcp-server

# Rollback a versión anterior
kubectl rollout undo deployment/mcp-server-deployment -n mcp-server
```

## ☁️ Despliegue en AWS

### AWS EKS (Kubernetes)

```bash
# Crear cluster EKS
eksctl create cluster \
  --name mcp-server-cluster \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Configurar kubectl
aws eks update-kubeconfig --region us-west-2 --name mcp-server-cluster

# Desplegar usando los manifests de K8s
cd deploy/k8s
./deploy-all.sh deploy

# Configurar Load Balancer
kubectl patch service mcp-server-lb -n mcp-server -p '{"spec":{"type":"LoadBalancer"}}'
```

### AWS ECS (Fargate)

```bash
# Crear task definition
aws ecs register-task-definition --cli-input-json file://deploy/aws/task-definition.json

# Crear servicio
aws ecs create-service \
  --cluster mcp-server-cluster \
  --service-name mcp-server-service \
  --task-definition mcp-server:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

### AWS Lambda (Serverless)

```bash
# Compilar para Lambda
GOOS=linux GOARCH=amd64 go build -o bootstrap cmd/lambda/main.go
zip lambda-deployment.zip bootstrap

# Desplegar con AWS CLI
aws lambda create-function \
  --function-name mcp-server-lambda \
  --runtime provided.al2 \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --handler bootstrap \
  --zip-file fileb://lambda-deployment.zip
```

## 🔵 Despliegue en Azure

### Azure Kubernetes Service (AKS)

```bash
# Crear grupo de recursos
az group create --name mcp-server-rg --location eastus

# Crear cluster AKS
az aks create \
  --resource-group mcp-server-rg \
  --name mcp-server-aks \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Obtener credenciales
az aks get-credentials --resource-group mcp-server-rg --name mcp-server-aks

# Desplegar aplicación
cd deploy/k8s
./deploy-all.sh deploy
```

### Azure Container Instances

```bash
# Crear container group
az container create \
  --resource-group mcp-server-rg \
  --name mcp-server-aci \
  --image ghcr.io/tu-usuario/mcp-server-ai:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8090 8091 50051 \
  --environment-variables \
    HTTP_PORT=8090 \
    GRPC_PORT=50051 \
    WS_PORT=8091
```

### Azure App Service

```bash
# Crear App Service Plan
az appservice plan create \
  --name mcp-server-plan \
  --resource-group mcp-server-rg \
  --sku B1 \
  --is-linux

# Crear Web App
az webapp create \
  --resource-group mcp-server-rg \
  --plan mcp-server-plan \
  --name mcp-server-app \
  --deployment-container-image-name ghcr.io/tu-usuario/mcp-server-ai:latest
```

## 🟡 Despliegue en Google Cloud

### Google Kubernetes Engine (GKE)

```bash
# Crear cluster GKE
gcloud container clusters create mcp-server-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Obtener credenciales
gcloud container clusters get-credentials mcp-server-cluster --zone us-central1-a

# Desplegar aplicación
cd deploy/k8s
./deploy-all.sh deploy
```

### Cloud Run

```bash
# Desplegar en Cloud Run
gcloud run deploy mcp-server \
  --image ghcr.io/tu-usuario/mcp-server-ai:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8090 \
  --memory 2Gi \
  --cpu 2
```

## 🔄 CI/CD

### GitHub Actions

Los pipelines están configurados en `.github/workflows/`:

- **ci-cd.yml**: Pipeline principal de CI/CD
- **security.yml**: Análisis de seguridad continuo

#### Configurar Secrets

```bash
# En GitHub, ir a Settings > Secrets and variables > Actions
# Agregar los siguientes secrets:

# AWS
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY

# Azure
AZURE_API_KEY
AZURE_RESOURCE_NAME
AZURE_ENDPOINT

# Kubernetes
KUBE_CONFIG_STAGING    # Base64 del kubeconfig para staging
KUBE_CONFIG_PRODUCTION # Base64 del kubeconfig para producción

# Notificaciones
SLACK_WEBHOOK

# Seguridad
SNYK_TOKEN
```

#### Flujo de Trabajo

1. **Push a `develop`** → Despliegue automático a staging
2. **Tag `v*`** → Despliegue automático a producción
3. **Pull Request** → Ejecuta tests y análisis de seguridad
4. **Manual** → Permite elegir ambiente de despliegue

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: golang:1.23
  services:
    - redis:7-alpine
    - postgres:16-alpine
  script:
    - go test -v ./...

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/mcp-server-deployment mcp-server=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

## 📊 Monitoreo y Logging

### Prometheus y Grafana

```bash
# Instalar Prometheus Operator
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# Aplicar ServiceMonitor
kubectl apply -f deploy/k8s/hpa-pdb.yaml

# Acceder a Grafana
kubectl port-forward service/grafana 3000:3000 -n monitoring
```

### Métricas Disponibles

- `mcp_requests_total`: Total de requests
- `mcp_request_duration_seconds`: Latencia de requests
- `mcp_tokens_used_total`: Tokens consumidos
- `mcp_cache_hits_total`: Hits de cache
- `mcp_errors_total`: Total de errores

### Logs Centralizados

```bash
# ELK Stack
kubectl apply -f deploy/monitoring/elasticsearch.yaml
kubectl apply -f deploy/monitoring/logstash.yaml
kubectl apply -f deploy/monitoring/kibana.yaml

# Fluentd para recolección
kubectl apply -f deploy/monitoring/fluentd.yaml
```

## 🔒 Seguridad

### Configuración de Secrets

```bash
# Usar herramientas como Sealed Secrets
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml

# Crear sealed secret
echo -n mypassword | kubectl create secret generic mysecret --dry-run=client --from-file=password=/dev/stdin -o yaml | kubeseal -o yaml > mysealedsecret.yaml
```

### Network Policies

```bash
# Aplicar políticas de red
kubectl apply -f deploy/k8s/rbac.yaml
```

### Análisis de Seguridad

```bash
# Ejecutar análisis local
docker run --rm -v $(pwd):/app aquasec/trivy fs /app

# Escanear imagen
docker run --rm aquasec/trivy image ghcr.io/tu-usuario/mcp-server-ai:latest
```

## 🔧 Troubleshooting

### Problemas Comunes

#### 1. Pod no inicia

```bash
# Ver eventos
kubectl describe pod <pod-name> -n mcp-server

# Ver logs
kubectl logs <pod-name> -n mcp-server

# Verificar recursos
kubectl top pods -n mcp-server
```

#### 2. Conexión a base de datos falla

```bash
# Verificar conectividad
kubectl exec -it <pod-name> -n mcp-server -- nc -zv postgres-service 5432

# Verificar secrets
kubectl get secret mcp-server-secrets -n mcp-server -o yaml
```

#### 3. Alta latencia

```bash
# Verificar HPA
kubectl get hpa -n mcp-server

# Ver métricas de CPU/memoria
kubectl top pods -n mcp-server

# Escalar manualmente
kubectl scale deployment mcp-server-deployment --replicas=10 -n mcp-server
```

#### 4. Problemas de red

```bash
# Verificar servicios
kubectl get svc -n mcp-server

# Probar conectividad interna
kubectl run test-pod --rm -i --tty --image=busybox -- /bin/sh
# Dentro del pod: nc -zv mcp-server-service 8090
```

### Comandos Útiles

```bash
# Ver todos los recursos
kubectl get all -n mcp-server

# Describir deployment
kubectl describe deployment mcp-server-deployment -n mcp-server

# Ver configuración
kubectl get configmap mcp-server-config -n mcp-server -o yaml

# Ejecutar shell en pod
kubectl exec -it <pod-name> -n mcp-server -- /bin/sh

# Port-forward para debug
kubectl port-forward <pod-name> 8090:8090 -n mcp-server

# Ver eventos del namespace
kubectl get events -n mcp-server --sort-by='.lastTimestamp'
```

### Logs y Debug

```bash
# Logs en tiempo real
kubectl logs -f deployment/mcp-server-deployment -n mcp-server

# Logs de todos los pods
kubectl logs -l app=mcp-server-ai -n mcp-server

# Logs anteriores (si el pod se reinició)
kubectl logs <pod-name> -n mcp-server --previous

# Aumentar verbosidad de logs
kubectl set env deployment/mcp-server-deployment LOG_LEVEL=debug -n mcp-server
```

## 📞 Soporte

- 📧 **Email**: devops@empresa.com
- 💬 **Slack**: #mcp-server-support
- 📖 **Documentación**: [docs.mcp-server.com](https://docs.mcp-server.com)
- 🐛 **Issues**: [GitHub Issues](https://github.com/tu-usuario/mcp-server-ai/issues)

---

**¡Despliegue exitoso! 🎉**

Para más información, consulta la [documentación completa](README.md) del proyecto.
