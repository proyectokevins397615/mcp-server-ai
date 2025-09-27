# ===========================================
# ETAPA 1: Construcción del binario
# ===========================================
# Usamos la imagen oficial de Go con Alpine para mantener el tamaño mínimo
FROM golang:1.23-alpine AS builder

# Instalamos las dependencias necesarias para la compilación
# - git: para obtener información de versión
# - gcc y musl-dev: para compilación con CGO si es necesario
# - protobuf y protobuf-dev: para generar código desde archivos .proto
RUN apk add --no-cache git gcc musl-dev protobuf protobuf-dev

# Establecemos el directorio de trabajo para la construcción
WORKDIR /build

# Copiamos los archivos de dependencias de Go
# Esto se hace primero para aprovechar la caché de Docker
COPY go.mod go.sum ./

# Descargamos todas las dependencias del proyecto
# Esto se cachea a menos que cambien go.mod o go.sum
RUN go mod download

# Copiamos todo el código fuente del proyecto
COPY . .

# Generamos los archivos de Go desde los archivos protobuf
# Esto es necesario para el servicio gRPC
RUN protoc --go_out=. --go-grpc_out=. internal/proto/ai_service.proto

# Compilamos el binario con optimizaciones de producción
# - CGO_ENABLED=0: Deshabilitamos CGO para un binario completamente estático
# - GOOS=linux: Compilamos para Linux
# - GOARCH=amd64: Arquitectura x86_64
# - ldflags -w -s: Eliminamos información de debug para reducir tamaño
# - ldflags -X: Inyectamos información de versión y tiempo de compilación
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s -X main.Version=$(git describe --tags --always --dirty) \
    -X main.BuildTime=$(date -u +%Y%m%d-%H%M%S)" \
    -a -installsuffix cgo \
    -o mcp-server cmd/server/main.go

# ===========================================
# ETAPA 2: Escaneo de seguridad
# ===========================================
# Usamos Trivy para escanear vulnerabilidades en el binario
FROM aquasec/trivy:latest AS scanner

# Copiamos el binario compilado
COPY --from=builder /build/mcp-server /mcp-server

# Ejecutamos el escaneo de seguridad
# exit-code 0: No falla el build si encuentra vulnerabilidades (cambiar a 1 para fallar)
RUN trivy fs --exit-code 0 --no-progress /mcp-server

# ===========================================
# ETAPA 3: Imagen final de producción
# ===========================================
# Usamos Alpine Linux para la imagen más pequeña posible
FROM alpine:3.19

# Instalamos solo las dependencias mínimas necesarias en runtime
# - ca-certificates: Para conexiones HTTPS/TLS
# - tzdata: Para manejo correcto de zonas horarias
RUN apk --no-cache add ca-certificates tzdata && \
    addgroup -g 1000 -S mcp && \
    adduser -u 1000 -S mcp -G mcp

# Configuramos la zona horaria (se puede sobrescribir con variable de entorno)
ENV TZ=UTC

# Copiamos el binario compilado desde la etapa de construcción
COPY --from=builder /build/mcp-server /usr/local/bin/mcp-server

# Copiamos los archivos de migración de base de datos
COPY --from=builder /build/migrations /app/migrations

# Creamos los directorios necesarios para la aplicación
# - /app/logs: Para archivos de log
# - /app/data: Para datos persistentes
RUN mkdir -p /app/logs /app/data && \
    chown -R mcp:mcp /app

# Cambiamos al usuario no-root por seguridad
USER mcp

# Establecemos el directorio de trabajo
WORKDIR /app

# Exponemos los puertos que usa la aplicación
# - 8090: API HTTP/REST
# - 8091: WebSocket
# - 50051: gRPC
EXPOSE 8090 8091 50051

# Configuramos el health check para Kubernetes/Docker
# - interval: Cada 30 segundos
# - timeout: Timeout de 3 segundos
# - start-period: Esperar 5 segundos antes del primer check
# - retries: 3 intentos antes de marcar como unhealthy
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8090/health || exit 1

# Definimos el punto de entrada (el ejecutable)
ENTRYPOINT ["/usr/local/bin/mcp-server"]

# Argumentos por defecto (se pueden sobrescribir al ejecutar el contenedor)
CMD ["-http=8090", "-ws=8091", "-grpc=50051"]