package ai

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"kevins-tools/mcp-server-ai/internal/cache"
	"kevins-tools/mcp-server-ai/internal/session"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"go.uber.org/zap"
)

// Manager gestiona múltiples proveedores de IA
type Manager struct {
	logger             *zap.Logger
	cache              *cache.RedisClient
	awsClient          *bedrockruntime.Client
	vertexClient       interface{} // Cliente de Vertex AI
	providers          map[string]Provider
	defaultModel       string
	parallelProcessor  *ParallelProcessor
	enableParallelMode bool
	sessionManager     *session.Manager
}

// Provider interfaz para proveedores de IA
type Provider interface {
	Generate(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error)
	StreamGenerate(ctx context.Context, req *GenerateRequest, stream chan<- *StreamChunk) error
	ListModels() []Model
	GetName() string
}

// GenerateRequest solicitud de generación
type GenerateRequest struct {
	Prompt       string                 `json:"prompt"`
	Model        string                 `json:"model"`
	Provider     string                 `json:"provider"`
	MaxTokens    int                    `json:"maxTokens"`
	Temperature  float64                `json:"temperature"`
	SystemPrompt string                 `json:"systemPrompt"`
	Options      map[string]interface{} `json:"options"`
	UserID       string                 `json:"-"` // Viene del header X-User-ID
	SessionID    string                 `json:"-"` // Viene del header X-Session-ID
}

// GenerateResponse respuesta de generación
type GenerateResponse struct {
	Content    string                 `json:"content"`
	Model      string                 `json:"model"`
	Provider   string                 `json:"provider"`
	TokensUsed int                    `json:"tokensUsed"`
	Duration   time.Duration          `json:"duration"`
	Cached     bool                   `json:"cached"`
	Metadata   map[string]interface{} `json:"metadata"`
}

// StreamChunk chunk de streaming
type StreamChunk struct {
	Content   string    `json:"content"`
	Index     int       `json:"index"`
	Finished  bool      `json:"finished"`
	Error     error     `json:"error,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// Model información del modelo
type Model struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	Provider     string   `json:"provider"`
	MaxTokens    int      `json:"maxTokens"`
	Description  string   `json:"description"`
	Capabilities []string `json:"capabilities"`
}

// NewManager crea un nuevo manager de IA
func NewManager(logger *zap.Logger, cache *cache.RedisClient) *Manager {
	m := &Manager{
		logger:             logger,
		cache:              cache,
		providers:          make(map[string]Provider),
		defaultModel:       "gpt-4.1",
		enableParallelMode: getEnv("ENABLE_PARALLEL_MODE", "true") == "true",
	}

	// Inicializar ParallelProcessor si está habilitado
	if m.enableParallelMode {
		m.parallelProcessor = NewParallelProcessor(m, logger)
		logger.Info("Parallel processing mode enabled with WorkerPool")
	}

	return m
}

// SetSessionManager configura el SessionManager
func (m *Manager) SetSessionManager(sm *session.Manager) {
	m.sessionManager = sm
	if sm != nil && sm.IsEnabled() {
		m.logger.Info("SessionManager integrated with AI Manager")
	}
}

// InitAWSBedrock inicializa AWS Bedrock
func (m *Manager) InitAWSBedrock() error {
	// Configurar AWS usando las variables de entorno
	cfg, err := config.LoadDefaultConfig(context.Background(),
		config.WithRegion(getEnv("AWS_REGION", "us-east-1")),
	)
	if err != nil {
		return fmt.Errorf("failed to load AWS config: %w", err)
	}

	m.awsClient = bedrockruntime.NewFromConfig(cfg)

	// Crear proveedor de AWS
	awsProvider := NewAWSProvider(m.awsClient, m.logger)
	m.providers["aws"] = awsProvider

	m.logger.Info("AWS Bedrock provider initialized")
	return nil
}

// InitAzureOpenAI inicializa Azure OpenAI
func (m *Manager) InitAzureOpenAI() error {
	// Crear proveedor de Azure usando el SDK oficial
	azureProvider, err := NewAzureProvider(m.logger)
	if err != nil {
		return fmt.Errorf("failed to create Azure provider: %w", err)
	}

	if m.providers == nil {
		m.providers = make(map[string]Provider)
	}
	m.providers["azure"] = azureProvider

	m.logger.Info("Azure OpenAI provider initialized")
	return nil
}

// InitVertexAI inicializa Vertex AI
func (m *Manager) InitVertexAI() error {
	// TODO: Implementar cliente de Vertex AI
	m.logger.Info("Vertex AI provider initialized (mock)")
	return nil
}

// Generate genera contenido usando el proveedor apropiado
func (m *Manager) Generate(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
	startTime := time.Now()

	// Si el SessionManager está habilitado, construir prompt con contexto
	if m.sessionManager != nil && m.sessionManager.IsEnabled() && req.UserID != "" && req.SessionID != "" {
		// Agregar mensaje del usuario al historial
		userMessage := session.Message{
			Role:    "user",
			Content: req.Prompt,
			Model:   req.Model,
		}

		if err := m.sessionManager.AddMessage(ctx, req.UserID, req.SessionID, userMessage); err != nil {
			m.logger.Warn("Failed to add user message to history", zap.Error(err))
		}

		// Construir prompt con contexto
		promptWithContext, err := m.sessionManager.BuildPromptWithContext(ctx, req.UserID, req.SessionID, req.Prompt)
		if err != nil {
			m.logger.Warn("Failed to build prompt with context", zap.Error(err))
		} else {
			// Usar el prompt con contexto
			req.Prompt = promptWithContext
		}
	}

	// Intentar obtener de caché
	if m.cache != nil {
		cacheKey := m.getCacheKey(req)
		if cached, err := m.cache.Get(ctx, cacheKey); err == nil && cached != "" {
			m.logger.Debug("Cache hit", zap.String("key", cacheKey))
			return &GenerateResponse{
				Content:  cached,
				Model:    req.Model,
				Provider: "cache",
				Cached:   true,
				Duration: time.Since(startTime),
			}, nil
		}
	}

	// Determinar proveedor basado en el modelo o proveedor especificado
	var provider Provider
	if req.Provider != "" {
		provider = m.providers[req.Provider]
	} else {
		provider = m.selectProvider(req.Model)
	}

	if provider == nil {
		return nil, fmt.Errorf("no provider available for model: %s", req.Model)
	}

	// Generar contenido
	response, err := provider.Generate(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("generation failed: %w", err)
	}

	response.Duration = time.Since(startTime)

	// Guardar en caché
	if m.cache != nil && response.Content != "" {
		cacheKey := m.getCacheKey(req)
		ttl := time.Hour // TTL configurable
		if err := m.cache.Set(ctx, cacheKey, response.Content, ttl); err != nil {
			m.logger.Warn("Failed to cache response", zap.Error(err))
		}
	}

	// Si el SessionManager está habilitado, guardar respuesta en el historial
	if m.sessionManager != nil && m.sessionManager.IsEnabled() && req.UserID != "" && req.SessionID != "" {
		assistantMessage := session.Message{
			Role:    "assistant",
			Content: response.Content,
			Model:   response.Model,
			Tokens:  response.TokensUsed,
		}

		if err := m.sessionManager.AddMessage(ctx, req.UserID, req.SessionID, assistantMessage); err != nil {
			m.logger.Warn("Failed to add assistant message to history", zap.Error(err))
		}
	}

	return response, nil
}

// GenerateBatch procesa múltiples requests en paralelo usando el WorkerPool
func (m *Manager) GenerateBatch(ctx context.Context, requests []*GenerateRequest) ([]*GenerateResponse, error) {
	if !m.enableParallelMode || m.parallelProcessor == nil {
		// Fallback a procesamiento secuencial si el modo paralelo no está habilitado
		responses := make([]*GenerateResponse, len(requests))
		for i, req := range requests {
			resp, err := m.Generate(ctx, req)
			if err != nil {
				m.logger.Error("Failed to generate response", zap.Error(err))
				responses[i] = &GenerateResponse{
					Model:    req.Model,
					Provider: req.Provider,
					Content:  "",
					Metadata: map[string]interface{}{"error": err.Error()},
				}
			} else {
				responses[i] = resp
			}
		}
		return responses, nil
	}

	// Usar ParallelProcessor para procesamiento en paralelo
	return m.parallelProcessor.ProcessBatch(ctx, requests)
}

// StreamGenerate genera contenido con streaming
func (m *Manager) StreamGenerate(ctx context.Context, req *GenerateRequest, stream chan<- *StreamChunk) error {
	// Determinar proveedor basado en el modelo o proveedor especificado
	var provider Provider
	if req.Provider != "" {
		provider = m.providers[req.Provider]
	} else {
		provider = m.selectProvider(req.Model)
	}

	if provider == nil {
		return fmt.Errorf("no provider available for model: %s", req.Model)
	}

	// Generar con streaming
	return provider.StreamGenerate(ctx, req, stream)
}

// ListModels lista todos los modelos disponibles
func (m *Manager) ListModels() []Model {
	var models []Model

	for _, provider := range m.providers {
		models = append(models, provider.ListModels()...)
	}

	return models
}

// selectProvider selecciona el proveedor basado en el modelo
func (m *Manager) selectProvider(model string) Provider {
	// Mapeo de modelos a proveedores
	modelLower := strings.ToLower(model)

	// AWS Bedrock models
	if strings.Contains(modelLower, "claude") ||
		strings.Contains(modelLower, "titan") ||
		strings.Contains(modelLower, "llama") {
		if provider, ok := m.providers["aws"]; ok {
			return provider
		}
	}

	// Azure OpenAI models - actualizado con todos los modelos disponibles
	if strings.Contains(modelLower, "gpt") ||
		strings.Contains(modelLower, "deepseek") ||
		strings.Contains(modelLower, "llama") ||
		strings.Contains(modelLower, "o4") ||
		strings.Contains(modelLower, "grok") ||
		strings.Contains(modelLower, "dall-e") {
		if provider, ok := m.providers["azure"]; ok {
			return provider
		}
	}

	// Vertex AI models
	if strings.Contains(modelLower, "gemini") ||
		strings.Contains(modelLower, "palm") {
		if provider, ok := m.providers["vertex"]; ok {
			return provider
		}
	}

	// Default to first available provider
	for _, provider := range m.providers {
		return provider
	}

	return nil
}

// getCacheKey genera una clave de caché para la solicitud
func (m *Manager) getCacheKey(req *GenerateRequest) string {
	// Crear clave única basada en los parámetros
	key := fmt.Sprintf("ai:%s:%s:%d:%.2f",
		req.Model,
		hashString(req.Prompt),
		req.MaxTokens,
		req.Temperature,
	)
	return key
}

// GetStatus obtiene el estado del servicio
func (m *Manager) GetStatus() map[string]interface{} {
	status := map[string]interface{}{
		"healthy":      true,
		"providers":    []string{},
		"models":       len(m.ListModels()),
		"cache":        m.cache != nil,
		"parallelMode": m.enableParallelMode,
	}

	for name := range m.providers {
		status["providers"] = append(status["providers"].([]string), name)
	}

	// Agregar estadísticas del pool si está habilitado
	if m.parallelProcessor != nil {
		status["poolStats"] = m.parallelProcessor.GetStats()
	}

	return status
}

// Shutdown cierra el manager y sus recursos
func (m *Manager) Shutdown(timeout time.Duration) error {
	if m.parallelProcessor != nil {
		return m.parallelProcessor.Shutdown(timeout)
	}
	return nil
}

// GetSessionManager retorna el SessionManager
func (m *Manager) GetSessionManager() *session.Manager {
	return m.sessionManager
}

// GetProviders obtiene la lista de proveedores disponibles
func (m *Manager) GetProviders() []string {
	providers := make([]string, 0, len(m.providers))
	for name := range m.providers {
		providers = append(providers, name)
	}
	return providers
}

// hashString genera un hash corto de un string
func hashString(s string) string {
	// Simple hash para la clave de caché
	h := 0
	for _, c := range s {
		h = h*31 + int(c)
	}
	return fmt.Sprintf("%x", h)
}

// getEnv obtiene variable de entorno con valor por defecto
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
