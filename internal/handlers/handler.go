package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
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

	// Convertir respuesta
	return &proto.GenerateResponse{
		Content:    resp.Content,
		Model:      resp.Model,
		Provider:   resp.Provider,
		TokensUsed: int32(resp.TokensUsed),
		Duration:   resp.Duration.Milliseconds(),
		Metadata:   convertMetadata(resp.Metadata),
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

	// Generar con streaming
	go func() {
		err := h.aiManager.StreamGenerate(stream.Context(), aiReq, chunks)
		if err != nil {
			h.logger.Error("Stream generation failed", zap.Error(err))
		}
	}()

	// Enviar chunks al cliente
	for chunk := range chunks {
		resp := &proto.StreamChunk{
			Content:   chunk.Content,
			Index:     int32(chunk.Index),
			Finished:  chunk.Finished,
			Timestamp: chunk.Timestamp.Unix(),
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

			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
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

	c.JSON(http.StatusOK, gin.H{
		"responses": responses,
		"count":     len(responses),
		"duration":  duration.Milliseconds(),
		"rate":      float64(len(responses)) / duration.Seconds(),
	})
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

		// Enviar evento de finalización
		fmt.Fprintf(w, "data: {\"finished\": true}\n\n")
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
