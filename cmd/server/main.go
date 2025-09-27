package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"kevins-tools/mcp-server-ai/internal/ai"
	"kevins-tools/mcp-server-ai/internal/cache"
	"kevins-tools/mcp-server-ai/internal/handlers"
	"kevins-tools/mcp-server-ai/internal/proto"
	"kevins-tools/mcp-server-ai/internal/session"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/joho/godotenv"
	_ "github.com/lib/pq" // PostgreSQL driver
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/reflection"
)

var (
	version = "1.0.0"
)

func main() {
	var (
		grpcPort = flag.String("grpc", "50051", "gRPC server port")
		httpPort = flag.String("http", "8090", "HTTP server port")
		wsPort   = flag.String("ws", "8091", "WebSocket server port")
		debug    = flag.Bool("debug", false, "Enable debug mode")
	)
	flag.Parse()

	// Cargar variables de entorno
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, using environment variables")
	}

	// Configurar logger basado en IN_MODE
	var logger *zap.Logger
	var err error
	inMode := os.Getenv("IN_MODE")

	// Configurar GIN_MODE basado en IN_MODE
	if inMode == "development" || *debug {
		os.Setenv("GIN_MODE", "debug")
		gin.SetMode(gin.DebugMode)
		logger, err = zap.NewDevelopment()
	} else {
		os.Setenv("GIN_MODE", "release")
		gin.SetMode(gin.ReleaseMode)
		logger, err = zap.NewProduction()
	}
	if err != nil {
		log.Fatalf("Failed to initialize logger: %v", err)
	}
	defer logger.Sync()

	logger.Info("Starting MCP AI Server",
		zap.String("version", version),
		zap.String("grpc_port", *grpcPort),
		zap.String("http_port", *httpPort),
		zap.String("ws_port", *wsPort))

	// Inicializar Redis Cache
	redisClient, err := cache.NewRedisClient(os.Getenv("REDIS_URL"))
	if err != nil {
		logger.Warn("Failed to connect to Redis, continuing without cache", zap.Error(err))
	} else {
		logger.Info("Connected to Redis cache")
	}

	// Inicializar proveedores de IA
	aiManager := ai.NewManager(logger, redisClient)

	// Inicializar SessionManager si está habilitado
	if os.Getenv("ENABLE_SESSION_MANAGEMENT") == "true" {
		sessionManager, err := initSessionManager(logger, redisClient)
		if err != nil {
			logger.Error("Failed to initialize SessionManager", zap.Error(err))
		} else {
			aiManager.SetSessionManager(sessionManager)
			logger.Info("SessionManager initialized and integrated")
		}
	} else {
		logger.Info("SessionManager disabled - running in stateless mode")
	}

	// Configurar AWS Bedrock
	if os.Getenv("AWS_ACCESS_KEY_ID") != "" {
		if err := aiManager.InitAWSBedrock(); err != nil {
			logger.Error("Failed to initialize AWS Bedrock", zap.Error(err))
		} else {
			logger.Info("AWS Bedrock initialized")
		}
	}

	// Configurar Azure OpenAI
	if os.Getenv("AZURE_API_KEY") != "" {
		if err := aiManager.InitAzureOpenAI(); err != nil {
			logger.Error("Failed to initialize Azure OpenAI", zap.Error(err))
		} else {
			logger.Info("Azure OpenAI initialized")
		}
	}

	// Configurar Vertex AI (Google)
	if os.Getenv("GOOGLE_APPLICATION_CREDENTIALS") != "" {
		if err := aiManager.InitVertexAI(); err != nil {
			logger.Error("Failed to initialize Vertex AI", zap.Error(err))
		} else {
			logger.Info("Vertex AI initialized")
		}
	}

	// Crear handlers
	handler := handlers.NewHandler(aiManager, logger)

	// Iniciar servidor gRPC
	go startGRPCServer(*grpcPort, handler, logger)

	// Iniciar servidor HTTP/REST
	go startHTTPServer(*httpPort, handler, logger)

	// Iniciar servidor WebSocket
	go startWebSocketServer(*wsPort, handler, logger)

	// Esperar señal de interrupción
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down servers...")

	// Graceful shutdown
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Shutdown del AI Manager (incluye el WorkerPool)
	if err := aiManager.Shutdown(30 * time.Second); err != nil {
		logger.Error("Failed to shutdown AI Manager gracefully", zap.Error(err))
	}

	// Cerrar conexiones
	if redisClient != nil {
		redisClient.Close()
	}

	// Usar el contexto para el shutdown
	_ = shutdownCtx

	logger.Info("MCP AI Server stopped")
}

// getCORSConfig obtiene la configuración de CORS desde las variables de entorno
func getCORSConfig() cors.Config {
	config := cors.DefaultConfig()

	// Configurar orígenes permitidos
	if os.Getenv("CORS_ALLOW_ALL_ORIGINS") == "true" {
		config.AllowAllOrigins = true
	} else {
		// Usar orígenes específicos
		allowedOrigins := os.Getenv("CORS_ALLOWED_ORIGINS")
		if allowedOrigins != "" {
			config.AllowOrigins = strings.Split(allowedOrigins, ",")
			for i := range config.AllowOrigins {
				config.AllowOrigins[i] = strings.TrimSpace(config.AllowOrigins[i])
			}
		}
		config.AllowAllOrigins = false
	}

	// Configurar métodos permitidos
	allowMethods := os.Getenv("CORS_ALLOW_METHODS")
	if allowMethods != "" {
		config.AllowMethods = strings.Split(allowMethods, ",")
		for i := range config.AllowMethods {
			config.AllowMethods[i] = strings.TrimSpace(config.AllowMethods[i])
		}
	} else {
		config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"}
	}

	// Configurar headers permitidos
	allowHeaders := os.Getenv("CORS_ALLOW_HEADERS")
	if allowHeaders != "" {
		config.AllowHeaders = strings.Split(allowHeaders, ",")
		for i := range config.AllowHeaders {
			config.AllowHeaders[i] = strings.TrimSpace(config.AllowHeaders[i])
		}
	} else {
		config.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization", "X-Request-ID", "X-User-ID", "X-Session-ID"}
	}

	// Configurar headers expuestos
	exposeHeaders := os.Getenv("CORS_EXPOSE_HEADERS")
	if exposeHeaders != "" {
		config.ExposeHeaders = strings.Split(exposeHeaders, ",")
		for i := range config.ExposeHeaders {
			config.ExposeHeaders[i] = strings.TrimSpace(config.ExposeHeaders[i])
		}
	} else {
		config.ExposeHeaders = []string{"Content-Length", "Content-Type"}
	}

	// Configurar credenciales
	if os.Getenv("CORS_ALLOW_CREDENTIALS") == "true" {
		config.AllowCredentials = true
	}

	// Configurar MaxAge
	if maxAge := os.Getenv("CORS_MAX_AGE"); maxAge != "" {
		if seconds, err := strconv.Atoi(maxAge); err == nil {
			config.MaxAge = time.Duration(seconds) * time.Second
		}
	} else {
		config.MaxAge = 12 * time.Hour
	}

	return config
}

// startGRPCServer inicia el servidor gRPC con soporte CORS
func startGRPCServer(port string, handler *handlers.Handler, logger *zap.Logger) {
	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		logger.Fatal("Failed to listen for gRPC", zap.Error(err))
	}

	// Configurar opciones de gRPC con soporte para CORS
	opts := []grpc.ServerOption{
		grpc.MaxRecvMsgSize(100 * 1024 * 1024), // 100MB
		grpc.MaxSendMsgSize(100 * 1024 * 1024), // 100MB
	}

	// En modo development, permitir conexiones inseguras
	if os.Getenv("IN_MODE") == "development" {
		opts = append(opts, grpc.Creds(insecure.NewCredentials()))
	}

	grpcServer := grpc.NewServer(opts...)
	proto.RegisterAIServiceServer(grpcServer, handler)

	// Habilitar reflection para debugging
	if os.Getenv("IN_MODE") == "development" {
		reflection.Register(grpcServer)
	}

	logger.Info("gRPC server listening",
		zap.String("port", port),
		zap.String("mode", os.Getenv("IN_MODE")))

	if err := grpcServer.Serve(lis); err != nil {
		logger.Fatal("Failed to serve gRPC", zap.Error(err))
	}
}

// startHTTPServer inicia el servidor HTTP/REST con configuración de CORS desde env
func startHTTPServer(port string, handler *handlers.Handler, logger *zap.Logger) {
	router := gin.New()
	router.Use(gin.Recovery())

	// Configurar proxies confiables
	trustedProxies := os.Getenv("TRUSTED_PROXIES")
	if trustedProxies != "" {
		proxies := strings.Split(trustedProxies, ",")
		for i := range proxies {
			proxies[i] = strings.TrimSpace(proxies[i])
		}
		router.SetTrustedProxies(proxies)
	} else {
		router.SetTrustedProxies(nil) // No confiar en ningún proxy
	}

	// Usar logger solo en modo development
	if os.Getenv("IN_MODE") == "development" {
		router.Use(gin.Logger())
	}

	// Aplicar configuración de CORS desde variables de entorno
	corsConfig := getCORSConfig()
	router.Use(cors.New(corsConfig))

	logger.Info("HTTP server CORS configured",
		zap.Bool("allow_all_origins", corsConfig.AllowAllOrigins),
		zap.Strings("allowed_origins", corsConfig.AllowOrigins),
		zap.Strings("allowed_methods", corsConfig.AllowMethods))

	// Health checks
	router.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "healthy", "version": version})
	})

	router.GET("/healthz", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	router.GET("/readyz", func(c *gin.Context) {
		c.JSON(200, gin.H{"ready": true})
	})

	// Metrics
	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// API endpoints
	api := router.Group("/api/v1")
	{
		// Generar código
		api.POST("/generate", handler.GenerateHTTP)

		// Streaming con SSE
		api.POST("/generate/stream", handler.GenerateStreamHTTP)

		// Procesamiento en batch con WorkerPool
		api.POST("/generate/batch", handler.GenerateBatchHTTP)

		// Batch con streaming SSE
		api.POST("/generate/batch/stream", handler.GenerateBatchStreamHTTP)

		// Listar modelos disponibles
		api.GET("/models", handler.ListModelsHTTP)

		// Validar prompt
		api.POST("/validate", handler.ValidatePrompt)

		// Estado del servicio
		api.GET("/status", handler.GetStatus)

		// Estadísticas del WorkerPool
		api.GET("/pool/stats", handler.GetPoolStats)
	}

	logger.Info("HTTP server listening", zap.String("port", port))

	if err := router.Run(":" + port); err != nil {
		logger.Fatal("Failed to start HTTP server", zap.Error(err))
	}
}

// initSessionManager inicializa el SessionManager con la configuración del entorno
func initSessionManager(logger *zap.Logger, redisClient *cache.RedisClient) (*session.Manager, error) {
	config := session.Config{
		Redis:    redisClient,
		Logger:   logger,
		Enabled:  os.Getenv("ENABLE_SESSION_MANAGEMENT") == "true",
		UseRedis: os.Getenv("USE_REDIS_CACHE") == "true",
		UseDB:    os.Getenv("USE_DATABASE_PERSISTENCE") == "true",
	}

	// Configurar límites
	if maxMessages := os.Getenv("SESSION_MAX_MESSAGES"); maxMessages != "" {
		if val, err := strconv.Atoi(maxMessages); err == nil {
			config.MaxMessages = val
		}
	}

	if contextWindow := os.Getenv("SESSION_CONTEXT_WINDOW"); contextWindow != "" {
		if val, err := strconv.Atoi(contextWindow); err == nil {
			config.ContextWindow = val
		}
	}

	if cacheTTL := os.Getenv("SESSION_CACHE_TTL"); cacheTTL != "" {
		if val, err := strconv.Atoi(cacheTTL); err == nil {
			config.CacheTTL = time.Duration(val) * time.Second
		}
	}

	// Conectar a PostgreSQL si está habilitado
	if config.UseDB {
		db, err := connectPostgreSQL()
		if err != nil {
			logger.Error("Failed to connect to PostgreSQL", zap.Error(err))
			// Continuar sin DB, solo con Redis si está disponible
			config.UseDB = false
		} else {
			config.DB = db
			logger.Info("Connected to PostgreSQL for session persistence")
		}
	}

	return session.NewManager(config), nil
}

// connectPostgreSQL conecta a la base de datos PostgreSQL
func connectPostgreSQL() (*sql.DB, error) {
	// Primero intentar con POSTGRES_URL si está disponible
	postgresURL := os.Getenv("POSTGRES_URL")
	if postgresURL != "" {
		db, err := sql.Open("postgres", postgresURL)
		if err != nil {
			return nil, fmt.Errorf("failed to open database with POSTGRES_URL: %w", err)
		}

		// Verificar conexión
		if err := db.Ping(); err != nil {
			return nil, fmt.Errorf("failed to ping database: %w", err)
		}

		// Configurar pool de conexiones
		db.SetMaxOpenConns(25)
		db.SetMaxIdleConns(5)
		db.SetConnMaxLifetime(5 * time.Minute)

		return db, nil
	}

	// Si no hay POSTGRES_URL, usar variables individuales
	host := os.Getenv("DB_HOST")
	port := os.Getenv("DB_PORT")
	user := os.Getenv("DB_USER")
	password := os.Getenv("DB_PASSWORD")
	dbname := os.Getenv("DB_NAME")
	sslmode := os.Getenv("DB_SSL_MODE")

	if host == "" {
		host = "localhost"
	}
	if port == "" {
		port = "5432"
	}
	if sslmode == "" {
		sslmode = "disable"
	}

	psqlInfo := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=%s",
		host, port, user, password, dbname, sslmode)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		return nil, err
	}

	// Verificar conexión
	if err := db.Ping(); err != nil {
		return nil, err
	}

	// Configurar pool de conexiones
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	return db, nil
}

// startWebSocketServer inicia el servidor WebSocket con configuración de CORS desde env
func startWebSocketServer(port string, handler *handlers.Handler, logger *zap.Logger) {
	router := gin.New()
	router.Use(gin.Recovery())

	// Configurar proxies confiables
	trustedProxies := os.Getenv("TRUSTED_PROXIES")
	if trustedProxies != "" {
		proxies := strings.Split(trustedProxies, ",")
		for i := range proxies {
			proxies[i] = strings.TrimSpace(proxies[i])
		}
		router.SetTrustedProxies(proxies)
	} else {
		router.SetTrustedProxies(nil) // No confiar en ningún proxy
	}

	// Aplicar configuración de CORS desde variables de entorno
	corsConfig := getCORSConfig()
	router.Use(cors.New(corsConfig))

	// Configurar el upgrader de WebSocket basado en CORS
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			// Si se permite todos los orígenes
			if os.Getenv("CORS_ALLOW_ALL_ORIGINS") == "true" {
				return true
			}

			// Verificar origen específico
			origin := r.Header.Get("Origin")
			allowedOrigins := strings.Split(os.Getenv("CORS_ALLOWED_ORIGINS"), ",")
			for _, allowed := range allowedOrigins {
				if strings.TrimSpace(allowed) == origin {
					return true
				}
			}

			// En modo development, permitir localhost
			if os.Getenv("IN_MODE") == "development" {
				if strings.Contains(origin, "localhost") || strings.Contains(origin, "127.0.0.1") {
					return true
				}
			}

			return false
		},
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
	}

	router.GET("/ws", func(c *gin.Context) {
		conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
		if err != nil {
			logger.Error("Failed to upgrade WebSocket", zap.Error(err))
			return
		}
		defer conn.Close()

		handler.HandleWebSocket(conn)
	})

	logger.Info("WebSocket server listening", zap.String("port", port))

	if err := router.Run(":" + port); err != nil {
		logger.Fatal("Failed to start WebSocket server", zap.Error(err))
	}
}
