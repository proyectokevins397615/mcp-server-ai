package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"kevins-tools/mcp-server-ai/internal/ai"
	"kevins-tools/mcp-server-ai/internal/proto"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Handler maneja las peticiones HTTP, gRPC y WebSocket
type Handler struct {
	proto.UnimplementedAIServiceServer
	aiManager *ai.Manager
	logger    *zap.Logger
}

// NewHandler crea un nuevo handler
func NewHandler(aiManager *ai.Manager, logger *zap.Logger) *Handler {
	return &Handler{
		aiManager: aiManager,
		logger:    logger,
	}
}

// Generate implementa el método gRPC para generar contenido
func (h *Handler) Generate(ctx context.Context, req *proto.GenerateRequest) (*proto.GenerateResponse, error) {
	// Validar request
	if req.Prompt == "" {
		return nil, status.Error(codes.InvalidArgument, "prompt is required")
	}

	// Crear request interno
	aiReq := &ai.GenerateRequest{
		Prompt:       req.Prompt,
		SystemPrompt: req.SystemPrompt,
		Model:        req.Model,
		Provider:     req.Provider,
		MaxTokens:    int(req.MaxTokens),
		Temperature:  float64(req.Temperature),
	}

	// Aplicar valores por defecto si no se proporcionan
	if aiReq.MaxTokens == 0 {
		// Intentar obtener el valor por defecto del modelo específico
		models := h.aiManager.ListModels()
		modelFound := false
		for _, model := range models {
			if model.ID == aiReq.Model || model.Name == aiReq.Model {
				if model.MaxTokens > 0 {
					aiReq.MaxTokens = model.MaxTokens
					modelFound = true
					break
				}
			}
		}

		// Si no se encuentra el modelo o no tiene MaxTokens, usar el valor del .env
		if !modelFound || aiReq.MaxTokens == 0 {
			if maxTokensStr := os.Getenv("MAX_TOKENS"); maxTokensStr != "" {
				if maxTokens, err := strconv.Atoi(maxTokensStr); err == nil {
					aiReq.MaxTokens = maxTokens
				}
			} else {
				aiReq.MaxTokens = 4096 // Valor por defecto si no hay configuración
			}
		}
	}

	if aiReq.Temperature == 0 {
		// Usar el valor del .env o un valor por defecto
		if tempStr := os.Getenv("TEMPERATURE"); tempStr != "" {
			if temp, err := strconv.ParseFloat(tempStr, 64); err == nil {
				aiReq.Temperature = temp
			}
		} else {
			aiReq.Temperature = 0.7 // Valor por defecto si no hay configuración
		}
	}

	// Generar contenido
	resp, err := h.aiManager.Generate(ctx, aiReq)
	if err != nil {
		h.logger.Error("Failed to generate", zap.Error(err))
		return nil, status.Error(codes.Internal, err.Error())
	}

	// Aplicar lógica de separación de tokens para gRPC
	tokensUsed := int32(resp.TokensUsed)
	metadata := convertMetadata(resp.Metadata)

	// Si es una respuesta cacheada, aplicar la misma lógica que HTTP
	if resp.Cached {
		// Para respuestas cacheadas, mostrar tokens actuales como 0
		tokensUsed = 0

		// Agregar campos de tokens históricos al metadata
		if originalTokens, exists := resp.Metadata["original_tokens"]; exists {
			if tokens, ok := originalTokens.(int); ok {
				metadata["completion_tokens_cache"] = fmt.Sprintf("%d", tokens)
				metadata["prompt_tokens_cache"] = fmt.Sprintf("%d", tokens/2) // Estimación
			} else if tokensStr, ok := originalTokens.(string); ok {
				if tokens, err := strconv.Atoi(tokensStr); err == nil {
					metadata["completion_tokens_cache"] = fmt.Sprintf("%d", tokens)
					metadata["prompt_tokens_cache"] = fmt.Sprintf("%d", tokens/2)
				}
			} else if tokensFloat, ok := originalTokens.(float64); ok {
				tokens := int(tokensFloat)
				metadata["completion_tokens_cache"] = fmt.Sprintf("%d", tokens)
				metadata["prompt_tokens_cache"] = fmt.Sprintf("%d", tokens/2)
			}
		} else {
			// Estimación basada en contenido si no hay metadata de tokens originales
			estimatedTokens := len(resp.Content) / 4
			if estimatedTokens < 1 {
				estimatedTokens = 1
			}
			metadata["completion_tokens_cache"] = fmt.Sprintf("%d", estimatedTokens)
			metadata["prompt_tokens_cache"] = fmt.Sprintf("%d", estimatedTokens/2)
		}

		// Agregar campos de tokens actuales como 0
		metadata["completion_tokens"] = "0"
		metadata["prompt_tokens"] = "0"
	} else {
		// Para respuestas no cacheadas, agregar campos de tokens actuales
		metadata["completion_tokens"] = fmt.Sprintf("%d", resp.TokensUsed)
		metadata["prompt_tokens"] = fmt.Sprintf("%d", resp.TokensUsed/2) // Estimación
		metadata["completion_tokens_cache"] = "0"
		metadata["prompt_tokens_cache"] = "0"
	}

	// Convertir respuesta
	return &proto.GenerateResponse{
		Content:    resp.Content,
		Model:      resp.Model,
		Provider:   resp.Provider,
		TokensUsed: tokensUsed,
		Duration:   resp.Duration.Milliseconds(),
		Metadata:   metadata,
	}, nil
}

// GenerateStream implementa el método gRPC para streaming
func (h *Handler) GenerateStream(req *proto.GenerateRequest, stream proto.AIService_GenerateStreamServer) error {
	// Validar request
	if req.Prompt == "" {
		return status.Error(codes.InvalidArgument, "prompt is required")
	}

	// Crear request interno
	aiReq := &ai.GenerateRequest{
		Prompt:       req.Prompt,
		SystemPrompt: req.SystemPrompt,
		Model:        req.Model,
		Provider:     req.Provider,
		MaxTokens:    int(req.MaxTokens),
		Temperature:  float64(req.Temperature),
	}

	// Aplicar valores por defecto si no se proporcionan
	if aiReq.MaxTokens == 0 {
		// Intentar obtener el valor por defecto del modelo específico
		models := h.aiManager.ListModels()
		modelFound := false
		for _, model := range models {
			if model.ID == aiReq.Model || model.Name == aiReq.Model {
				if model.MaxTokens > 0 {
					aiReq.MaxTokens = model.MaxTokens
					modelFound = true
					break
				}
			}
		}

		// Si no se encuentra el modelo o no tiene MaxTokens, usar el valor del .env
		if !modelFound || aiReq.MaxTokens == 0 {
			if maxTokensStr := os.Getenv("MAX_TOKENS"); maxTokensStr != "" {
				if maxTokens, err := strconv.Atoi(maxTokensStr); err == nil {
					aiReq.MaxTokens = maxTokens
				}
			} else {
				aiReq.MaxTokens = 4096 // Valor por defecto si no hay configuración
			}
		}
	}

	if aiReq.Temperature == 0 {
		// Usar el valor del .env o un valor por defecto
		if tempStr := os.Getenv("TEMPERATURE"); tempStr != "" {
			if temp, err := strconv.ParseFloat(tempStr, 64); err == nil {
				aiReq.Temperature = temp
			}
		} else {
			aiReq.Temperature = 0.7 // Valor por defecto si no hay configuración
		}
	}

	// Canal para chunks
	chunks := make(chan *ai.StreamChunk, 100)

	// Variables para rastrear contenido completo y detectar cache
	var fullContent strings.Builder
	var startTime = time.Now()

	// Generar con streaming
	go func() {
		err := h.aiManager.StreamGenerate(stream.Context(), aiReq, chunks)
		if err != nil {
			h.logger.Error("Stream generation failed", zap.Error(err))
		}
	}()

	// Enviar chunks al cliente
	for chunk := range chunks {
		// Construir contenido completo
		if !chunk.Finished {
			fullContent.WriteString(chunk.Content)
		}

		resp := &proto.StreamChunk{
			Content:   chunk.Content,
			Index:     int32(chunk.Index),
			Finished:  chunk.Finished,
			Timestamp: chunk.Timestamp.Unix(),
		}

		// Si es el último chunk, agregar información de tokens en la respuesta
		if chunk.Finished {
			duration := time.Since(startTime)
			// Detectar cache basándose en duración muy rápida (< 1 segundo para streaming simulado)
			isCached := duration < 1000*time.Millisecond

			// Crear información de tokens
			var tokenInfo map[string]interface{}

			if isCached {
				estimatedTokens := len(fullContent.String()) / 4
				if estimatedTokens < 1 {
					estimatedTokens = 1
				}
				tokenInfo = map[string]interface{}{
					"completion_tokens":       0,
					"prompt_tokens":           0,
					"completion_tokens_cache": estimatedTokens * 3 / 4,
					"prompt_tokens_cache":     len(aiReq.Prompt) / 4,
					"cached":                  true,
					"provider":                "cache",
					"model":                   aiReq.Model,
					"duration_ms":             duration.Milliseconds(),
				}
			} else {
				estimatedTokens := len(fullContent.String()) / 4
				if estimatedTokens < 1 {
					estimatedTokens = 1
				}
				tokenInfo = map[string]interface{}{
					"completion_tokens":       estimatedTokens * 3 / 4,
					"prompt_tokens":           len(aiReq.Prompt) / 4,
					"completion_tokens_cache": 0,
					"prompt_tokens_cache":     0,
					"cached":                  false,
					"provider":                aiReq.Provider,
					"model":                   aiReq.Model,
					"duration_ms":             duration.Milliseconds(),
				}
			}

			// Convertir a JSON y enviar como contenido del chunk final
			if tokenInfoJSON, err := json.Marshal(tokenInfo); err == nil {
				resp.Content = string(tokenInfoJSON)
			}
		}

		if err := stream.Send(resp); err != nil {
			return err
		}
	}

	return nil
}

// ListModels implementa el método gRPC para listar modelos
func (h *Handler) ListModels(ctx context.Context, req *proto.ListModelsRequest) (*proto.ListModelsResponse, error) {
	models := h.aiManager.ListModels()

	protoModels := make([]*proto.Model, len(models))
	for i, model := range models {
		protoModels[i] = &proto.Model{
			Id:           model.ID,
			Name:         model.Name,
			Provider:     model.Provider,
			MaxTokens:    int32(model.MaxTokens),
			Description:  model.Description,
			Capabilities: model.Capabilities,
		}
	}

	return &proto.ListModelsResponse{
		Models: protoModels,
	}, nil
}

// ListModelsHTTP maneja las peticiones HTTP para listar modelos
func (h *Handler) ListModelsHTTP(c *gin.Context) {
	models := h.aiManager.ListModels()
	c.JSON(http.StatusOK, gin.H{
		"models": models,
		"count":  len(models),
	})
}

// GenerateHTTP maneja las peticiones HTTP para generar contenido
func (h *Handler) GenerateHTTP(c *gin.Context) {
	var req ai.GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Aplicar valores por defecto si no se proporcionan
	if req.MaxTokens == 0 {
		// Intentar obtener el valor por defecto del modelo específico
		models := h.aiManager.ListModels()
		modelFound := false
		for _, model := range models {
			if model.ID == req.Model || model.Name == req.Model {
				if model.MaxTokens > 0 {
					req.MaxTokens = model.MaxTokens
					modelFound = true
					break
				}
			}
		}

		// Si no se encuentra el modelo o no tiene MaxTokens, usar el valor del .env
		if !modelFound || req.MaxTokens == 0 {
			if maxTokensStr := os.Getenv("MAX_TOKENS"); maxTokensStr != "" {
				if maxTokens, err := strconv.Atoi(maxTokensStr); err == nil {
					req.MaxTokens = maxTokens
				}
			} else {
				req.MaxTokens = 4096 // Valor por defecto si no hay configuración
			}
		}
	}

	if req.Temperature == 0 {
		// Usar el valor del .env o un valor por defecto
		if tempStr := os.Getenv("TEMPERATURE"); tempStr != "" {
			if temp, err := strconv.ParseFloat(tempStr, 64); err == nil {
				req.Temperature = temp
			}
		} else {
			req.Temperature = 0.7 // Valor por defecto si no hay configuración
		}
	}

	// Extraer headers de sesión si están presentes
	userID := c.GetHeader("X-User-ID")
	sessionID := c.GetHeader("X-Session-ID")

	// Si el SessionManager está habilitado, validar headers
	if h.aiManager.GetSessionManager() != nil && h.aiManager.GetSessionManager().IsEnabled() {
		if userID == "" || sessionID == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "X-User-ID and X-Session-ID headers are required when session management is enabled",
			})
			return
		}
	}

	// Asignar IDs a la request
	req.UserID = userID
	req.SessionID = sessionID

	// Generar contenido
	resp, err := h.aiManager.Generate(c.Request.Context(), &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, resp)
}

// GenerateStreamHTTP maneja las peticiones HTTP con Server-Sent Events
func (h *Handler) GenerateStreamHTTP(c *gin.Context) {
	var req ai.GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Aplicar valores por defecto si no se proporcionan
	if req.MaxTokens == 0 {
		// Intentar obtener el valor por defecto del modelo específico
		models := h.aiManager.ListModels()
		modelFound := false
		for _, model := range models {
			if model.ID == req.Model || model.Name == req.Model {
				if model.MaxTokens > 0 {
					req.MaxTokens = model.MaxTokens
					modelFound = true
					break
				}
			}
		}

		// Si no se encuentra el modelo o no tiene MaxTokens, usar el valor del .env
		if !modelFound || req.MaxTokens == 0 {
			if maxTokensStr := os.Getenv("MAX_TOKENS"); maxTokensStr != "" {
				if maxTokens, err := strconv.Atoi(maxTokensStr); err == nil {
					req.MaxTokens = maxTokens
				}
			} else {
				req.MaxTokens = 4096 // Valor por defecto si no hay configuración
			}
		}
	}

	if req.Temperature == 0 {
		// Usar el valor del .env o un valor por defecto
		if tempStr := os.Getenv("TEMPERATURE"); tempStr != "" {
			if temp, err := strconv.ParseFloat(tempStr, 64); err == nil {
				req.Temperature = temp
			}
		} else {
			req.Temperature = 0.7 // Valor por defecto si no hay configuración
		}
	}

	// Extraer headers de sesión si están presentes
	userID := c.GetHeader("X-User-ID")
	sessionID := c.GetHeader("X-Session-ID")

	// Si el SessionManager está habilitado, validar headers
	if h.aiManager.GetSessionManager() != nil && h.aiManager.GetSessionManager().IsEnabled() {
		if userID == "" || sessionID == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "X-User-ID and X-Session-ID headers are required when session management is enabled",
			})
			return
		}
	}

	// Asignar IDs a la request
	req.UserID = userID
	req.SessionID = sessionID

	// Configurar SSE
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	// Canal para chunks
	chunks := make(chan *ai.StreamChunk, 100)

	// Variables para rastrear contenido completo y detectar cache
	var fullContent strings.Builder
	var startTime = time.Now()

	// Generar con streaming
	go func() {
		err := h.aiManager.StreamGenerate(c.Request.Context(), &req, chunks)
		if err != nil {
			h.logger.Error("Stream generation failed", zap.Error(err))
		}
	}()

	// Enviar chunks como SSE
	c.Stream(func(w io.Writer) bool {
		select {
		case chunk, ok := <-chunks:
			if !ok {
				return false
			}

			// Construir contenido completo
			if !chunk.Finished {
				fullContent.WriteString(chunk.Content)
			}

			// Si es el último chunk, agregar información de tokens
			if chunk.Finished {
				duration := time.Since(startTime)
				// Detectar cache basándose en duración muy rápida (< 1 segundo para streaming simulado)
				isCached := duration < 1000*time.Millisecond

				// Crear información de tokens
				var tokenInfo map[string]interface{}

				if isCached {
					estimatedTokens := len(fullContent.String()) / 4
					if estimatedTokens < 1 {
						estimatedTokens = 1
					}
					tokenInfo = map[string]interface{}{
						"completion_tokens":       0,
						"prompt_tokens":           0,
						"completion_tokens_cache": estimatedTokens * 3 / 4,
						"prompt_tokens_cache":     len(req.Prompt) / 4,
						"cached":                  true,
						"provider":                "cache",
						"model":                   req.Model,
						"duration_ms":             duration.Milliseconds(),
					}
				} else {
					estimatedTokens := len(fullContent.String()) / 4
					if estimatedTokens < 1 {
						estimatedTokens = 1
					}
					tokenInfo = map[string]interface{}{
						"completion_tokens":       estimatedTokens * 3 / 4,
						"prompt_tokens":           len(req.Prompt) / 4,
						"completion_tokens_cache": 0,
						"prompt_tokens_cache":     0,
						"cached":                  false,
						"provider":                req.Provider,
						"model":                   req.Model,
						"duration_ms":             duration.Milliseconds(),
					}
				}

				// Crear chunk modificado con información de tokens
				modifiedChunk := *chunk
				if tokenInfoJSON, err := json.Marshal(tokenInfo); err == nil {
					modifiedChunk.Content = string(tokenInfoJSON)
				}

				data, _ := json.Marshal(modifiedChunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
			} else {
				data, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
			}

			return true

		case <-c.Request.Context().Done():
			return false
		}
	})
}

// HandleWebSocket maneja conexiones WebSocket
func (h *Handler) HandleWebSocket(conn *websocket.Conn) {
	defer conn.Close()

	for {
		// Leer mensaje
		var req ai.GenerateRequest
		if err := conn.ReadJSON(&req); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				h.logger.Error("WebSocket error", zap.Error(err))
			}
			break
		}

		// Aplicar valores por defecto si no se proporcionan
		if req.MaxTokens == 0 {
			// Intentar obtener el valor por defecto del modelo específico
			models := h.aiManager.ListModels()
			modelFound := false
			for _, model := range models {
				if model.ID == req.Model || model.Name == req.Model {
					if model.MaxTokens > 0 {
						req.MaxTokens = model.MaxTokens
						modelFound = true
						break
					}
				}
			}

			// Si no se encuentra el modelo o no tiene MaxTokens, usar el valor del .env
			if !modelFound || req.MaxTokens == 0 {
				if maxTokensStr := os.Getenv("MAX_TOKENS"); maxTokensStr != "" {
					if maxTokens, err := strconv.Atoi(maxTokensStr); err == nil {
						req.MaxTokens = maxTokens
					}
				} else {
					req.MaxTokens = 4096 // Valor por defecto si no hay configuración
				}
			}
		}

		if req.Temperature == 0 {
			// Usar el valor del .env o un valor por defecto
			if tempStr := os.Getenv("TEMPERATURE"); tempStr != "" {
				if temp, err := strconv.ParseFloat(tempStr, 64); err == nil {
					req.Temperature = temp
				}
			} else {
				req.Temperature = 0.7 // Valor por defecto si no hay configuración
			}
		}

		// Canal para chunks
		chunks := make(chan *ai.StreamChunk, 100)

		// Generar con streaming
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()

			err := h.aiManager.StreamGenerate(ctx, &req, chunks)
			if err != nil {
				h.logger.Error("Stream generation failed", zap.Error(err))
				conn.WriteJSON(map[string]string{"error": err.Error()})
			}
		}()

		// Enviar chunks al cliente
		for chunk := range chunks {
			if err := conn.WriteJSON(chunk); err != nil {
				h.logger.Error("Failed to write to WebSocket", zap.Error(err))
				break
			}
		}
	}
}

// ValidatePrompt valida un prompt
func (h *Handler) ValidatePrompt(c *gin.Context) {
	var req struct {
		Prompt string `json:"prompt"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Validación básica
	if len(req.Prompt) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"valid": false, "error": "prompt is empty"})
		return
	}

	if len(req.Prompt) > 100000 {
		c.JSON(http.StatusBadRequest, gin.H{"valid": false, "error": "prompt is too long"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"valid": true})
}

// GetStatus obtiene el estado del servicio
func (h *Handler) GetStatus(c *gin.Context) {
	status := h.aiManager.GetStatus()

	// Agregar estado del SessionManager
	sessionEnabled := false
	if h.aiManager.GetSessionManager() != nil {
		sessionEnabled = h.aiManager.GetSessionManager().IsEnabled()
	}

	c.JSON(http.StatusOK, gin.H{
		"status":         "healthy",
		"providers":      status["providers"],
		"models":         status["models"],
		"cache":          status["cache"],
		"parallel":       status["parallelMode"],
		"poolStats":      status["poolStats"],
		"sessionEnabled": sessionEnabled,
		"timestamp":      time.Now().Unix(),
	})
}

// GenerateBatchHTTP maneja las peticiones HTTP para procesamiento en batch
func (h *Handler) GenerateBatchHTTP(c *gin.Context) {
	var req struct {
		Requests []*ai.GenerateRequest `json:"requests"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if len(req.Requests) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "no requests provided"})
		return
	}

	// Aplicar valores por defecto a cada request si no se proporcionan
	models := h.aiManager.ListModels()
	for _, r := range req.Requests {
		if r.MaxTokens == 0 {
			// Intentar obtener el valor por defecto del modelo específico
			modelFound := false
			for _, model := range models {
				if model.ID == r.Model || model.Name == r.Model {
					if model.MaxTokens > 0 {
						r.MaxTokens = model.MaxTokens
						modelFound = true
						break
					}
				}
			}

			// Si no se encuentra el modelo o no tiene MaxTokens, usar el valor del .env
			if !modelFound || r.MaxTokens == 0 {
				if maxTokensStr := os.Getenv("MAX_TOKENS"); maxTokensStr != "" {
					if maxTokens, err := strconv.Atoi(maxTokensStr); err == nil {
						r.MaxTokens = maxTokens
					}
				} else {
					r.MaxTokens = 4096 // Valor por defecto si no hay configuración
				}
			}
		}

		if r.Temperature == 0 {
			// Usar el valor del .env o un valor por defecto
			if tempStr := os.Getenv("TEMPERATURE"); tempStr != "" {
				if temp, err := strconv.ParseFloat(tempStr, 64); err == nil {
					r.Temperature = temp
				}
			} else {
				r.Temperature = 0.7 // Valor por defecto si no hay configuración
			}
		}
	}

	// Extraer headers de sesión si están presentes
	userID := c.GetHeader("X-User-ID")
	sessionID := c.GetHeader("X-Session-ID")

	// Si el SessionManager está habilitado, validar headers
	if h.aiManager.GetSessionManager() != nil && h.aiManager.GetSessionManager().IsEnabled() {
		if userID == "" || sessionID == "" {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "X-User-ID and X-Session-ID headers are required when session management is enabled",
			})
			return
		}

		// Asignar IDs a todas las requests
		for _, r := range req.Requests {
			r.UserID = userID
			r.SessionID = sessionID
		}
	}

	// Limitar el tamaño del batch
	maxBatchSize := 1000
	if len(req.Requests) > maxBatchSize {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("batch size exceeds maximum of %d", maxBatchSize),
		})
		return
	}

	h.logger.Info("Processing batch request",
		zap.Int("batch_size", len(req.Requests)))

	// Procesar batch usando el WorkerPool
	startTime := time.Now()
	responses, err := h.aiManager.GenerateBatch(c.Request.Context(), req.Requests)

	if err != nil {
		h.logger.Error("Batch processing failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":           err.Error(),
			"partial_results": responses,
		})
		return
	}

	duration := time.Since(startTime)

	// Calcular tokens de manera inteligente separando cache y provider
	var totalTokens int
	var totalTokensCache int
	var totalTokensProvider int

	for _, resp := range responses {
		if resp != nil {
			if resp.Cached {
				// Para respuestas cacheadas, obtener tokens originales del metadata si están disponibles
				if originalTokens, exists := resp.Metadata["original_tokens"]; exists {
					if tokens, ok := originalTokens.(int); ok {
						totalTokensCache += tokens
						totalTokens += tokens
					} else if tokensStr, ok := originalTokens.(string); ok {
						if tokens, err := strconv.Atoi(tokensStr); err == nil {
							totalTokensCache += tokens
							totalTokens += tokens
						}
					} else if tokensFloat, ok := originalTokens.(float64); ok {
						tokens := int(tokensFloat)
						totalTokensCache += tokens
						totalTokens += tokens
					}
				} else {
					// Si no hay metadata de tokens originales, usar TokensUsed (que debería ser 0 para cache)
					// pero intentar estimar basado en el contenido
					estimatedTokens := len(resp.Content) / 4 // Estimación aproximada: ~4 chars por token
					if estimatedTokens < 1 {
						estimatedTokens = 1
					}
					totalTokensCache += estimatedTokens
					totalTokens += estimatedTokens
				}
			} else {
				// Para respuestas no cacheadas, usar TokensUsed directamente
				totalTokensProvider += resp.TokensUsed
				totalTokens += resp.TokensUsed
			}
		}
	}

	// Formatear tiempo de procesamiento
	processingTime := fmt.Sprintf("%.1fs", duration.Seconds())

	// Formatear rate con unidades
	rate := fmt.Sprintf("%.2f req/s", float64(len(responses))/duration.Seconds())

	// Crear respuesta base con TODOS los campos requeridos
	responseData := gin.H{
		"count":              len(responses),
		"parallelProcessing": true,
		"processingTime":     processingTime,
		"rate":               rate,
		"responses":          responses,
		"totalTokens":        totalTokens,
	}

	// Agregar desglose de tokens si hay respuestas cacheadas
	if totalTokensCache > 0 {
		responseData["totalTokensCache"] = totalTokensCache
		responseData["totalTokensProvider"] = totalTokensProvider
	}

	c.JSON(http.StatusOK, responseData)
}

// GenerateBatchStreamHTTP maneja procesamiento en batch con streaming
func (h *Handler) GenerateBatchStreamHTTP(c *gin.Context) {
	var req struct {
		Requests []*ai.GenerateRequest `json:"requests"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if len(req.Requests) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "no requests provided"})
		return
	}

	// Aplicar valores por defecto a cada request si no se proporcionan
	models := h.aiManager.ListModels()
	for _, r := range req.Requests {
		if r.MaxTokens == 0 {
			// Intentar obtener el valor por defecto del modelo específico
			modelFound := false
			for _, model := range models {
				if model.ID == r.Model || model.Name == r.Model {
					if model.MaxTokens > 0 {
						r.MaxTokens = model.MaxTokens
						modelFound = true
						break
					}
				}
			}

			// Si no se encuentra el modelo o no tiene MaxTokens, usar el valor del .env
			if !modelFound || r.MaxTokens == 0 {
				if maxTokensStr := os.Getenv("MAX_TOKENS"); maxTokensStr != "" {
					if maxTokens, err := strconv.Atoi(maxTokensStr); err == nil {
						r.MaxTokens = maxTokens
					}
				} else {
					r.MaxTokens = 4096 // Valor por defecto si no hay configuración
				}
			}
		}

		if r.Temperature == 0 {
			// Usar el valor del .env o un valor por defecto
			if tempStr := os.Getenv("TEMPERATURE"); tempStr != "" {
				if temp, err := strconv.ParseFloat(tempStr, 64); err == nil {
					r.Temperature = temp
				}
			} else {
				r.Temperature = 0.7 // Valor por defecto si no hay configuración
			}
		}
	}

	// Configurar SSE
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	// Procesar cada request y enviar resultado como SSE
	startTime := time.Now()
	responses := make([]*ai.GenerateResponse, len(req.Requests))

	c.Stream(func(w io.Writer) bool {
		for i, request := range req.Requests {
			// Generar respuesta
			resp, err := h.aiManager.Generate(c.Request.Context(), request)

			result := map[string]interface{}{
				"index": i,
				"total": len(req.Requests),
			}

			if err != nil {
				result["error"] = err.Error()
			} else {
				result["response"] = resp
				responses[i] = resp // Guardar respuesta para el resumen
			}

			data, _ := json.Marshal(result)
			fmt.Fprintf(w, "data: %s\n\n", data)

			// Verificar si el contexto fue cancelado
			select {
			case <-c.Request.Context().Done():
				return false
			default:
				// Continuar
			}
		}

		// Calcular resumen final
		duration := time.Since(startTime)

		// Calcular tokens de manera inteligente separando cache y provider
		var totalTokens int
		var totalTokensCache int
		var totalTokensProvider int

		for _, resp := range responses {
			if resp != nil {
				if resp.Cached {
					// Para respuestas cacheadas, obtener tokens originales del metadata si están disponibles
					if originalTokens, exists := resp.Metadata["original_tokens"]; exists {
						if tokens, ok := originalTokens.(int); ok {
							totalTokensCache += tokens
							totalTokens += tokens
						} else if tokensStr, ok := originalTokens.(string); ok {
							if tokens, err := strconv.Atoi(tokensStr); err == nil {
								totalTokensCache += tokens
								totalTokens += tokens
							}
						} else if tokensFloat, ok := originalTokens.(float64); ok {
							tokens := int(tokensFloat)
							totalTokensCache += tokens
							totalTokens += tokens
						}
					} else {
						// Si no hay metadata de tokens originales, estimar basado en el contenido
						estimatedTokens := len(resp.Content) / 4
						if estimatedTokens < 1 {
							estimatedTokens = 1
						}
						totalTokensCache += estimatedTokens
						totalTokens += estimatedTokens
					}
				} else {
					// Para respuestas no cacheadas, usar TokensUsed directamente
					totalTokensProvider += resp.TokensUsed
					totalTokens += resp.TokensUsed
				}
			}
		}

		// Crear resumen final
		summary := map[string]interface{}{
			"count":              len(req.Requests),
			"parallelProcessing": true,
			"processingTime":     fmt.Sprintf("%.1fs", duration.Seconds()),
			"rate":               fmt.Sprintf("%.2f req/s", float64(len(req.Requests))/duration.Seconds()),
			"totalTokens":        totalTokens,
			"finished":           true,
		}

		// Agregar desglose de tokens si hay respuestas cacheadas
		if totalTokensCache > 0 {
			summary["totalTokensCache"] = totalTokensCache
			summary["totalTokensProvider"] = totalTokensProvider
		}

		// Enviar resumen final
		summaryData, _ := json.Marshal(summary)
		fmt.Fprintf(w, "data: %s\n\n", summaryData)
		return false
	})
}

// GetPoolStats obtiene estadísticas del WorkerPool
func (h *Handler) GetPoolStats(c *gin.Context) {
	status := h.aiManager.GetStatus()

	poolStats, ok := status["poolStats"]
	if !ok {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "WorkerPool not enabled",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"pool_stats":    poolStats,
		"parallel_mode": status["parallelMode"],
		"timestamp":     time.Now().Unix(),
	})
}

// convertMetadata convierte metadata a map[string]string para proto
func convertMetadata(metadata map[string]interface{}) map[string]string {
	result := make(map[string]string)
	for k, v := range metadata {
		result[k] = fmt.Sprintf("%v", v)
	}
	return result
}
