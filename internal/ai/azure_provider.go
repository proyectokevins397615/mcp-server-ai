package ai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"go.uber.org/zap"
)

// AzureProvider proveedor de Azure OpenAI usando HTTP directo
type AzureProvider struct {
	endpoint   string
	apiKey     string
	apiVersion string
	logger     *zap.Logger
	httpClient *http.Client
}

// AzureMessage representa un mensaje para Azure OpenAI
type AzureMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// AzureRequest representa una request a Azure OpenAI
type AzureRequest struct {
	Messages    []AzureMessage `json:"messages"`
	Temperature float64        `json:"temperature"`
	MaxTokens   int            `json:"max_tokens"`
	Stream      bool           `json:"stream,omitempty"`
}

// AzureChoice representa una choice en la respuesta
type AzureChoice struct {
	Index   int `json:"index"`
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	FinishReason string `json:"finish_reason"`
}

// AzureUsage representa el uso de tokens
type AzureUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// AzureResponse representa la respuesta de Azure OpenAI
type AzureResponse struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []AzureChoice `json:"choices"`
	Usage   AzureUsage    `json:"usage"`
}

// AzureStreamChoice para respuestas de streaming
type AzureStreamChoice struct {
	Index int `json:"index"`
	Delta struct {
		Content string `json:"content,omitempty"`
		Role    string `json:"role,omitempty"`
	} `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// AzureStreamResponse para respuestas de streaming
type AzureStreamResponse struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []AzureStreamChoice `json:"choices"`
}

// NewAzureProvider crea un nuevo proveedor de Azure usando HTTP directo
func NewAzureProvider(logger *zap.Logger) (*AzureProvider, error) {
	logger.Info("Initializing Azure OpenAI HTTP provider")

	endpoint := os.Getenv("AZURE_ENDPOINT")
	if endpoint == "" {
		logger.Error("AZURE_ENDPOINT environment variable not configured")
		return nil, fmt.Errorf("AZURE_ENDPOINT not configured")
	}

	endpoint = strings.TrimSuffix(endpoint, "/")
	logger.Debug("Azure endpoint configured", zap.String("endpoint", endpoint))

	apiKey := os.Getenv("AZURE_API_KEY")
	if apiKey == "" {
		logger.Error("AZURE_API_KEY environment variable not configured")
		return nil, fmt.Errorf("AZURE_API_KEY not configured")
	}

	apiVersion := os.Getenv("AZURE_API_VERSION")
	if apiVersion == "" {
		apiVersion = "2024-02-01" // versión por defecto
	}

	logger.Debug("Azure API key configured", zap.Int("key_length", len(apiKey)))

	httpClient := &http.Client{
		Timeout: 30 * time.Second,
	}

	logger.Info("Azure OpenAI HTTP provider initialized successfully")
	return &AzureProvider{
		endpoint:   endpoint,
		apiKey:     apiKey,
		apiVersion: apiVersion,
		logger:     logger,
		httpClient: httpClient,
	}, nil
}

// Generate genera contenido usando Azure OpenAI API directa
func (p *AzureProvider) Generate(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
	startTime := time.Now()
	requestID := fmt.Sprintf("azure-http-%d", startTime.UnixNano())

	p.logger.Info("Starting Azure OpenAI HTTP generation request",
		zap.String("request_id", requestID),
		zap.String("requested_model", req.Model))

	model := req.Model
	if model == "" {
		model = os.Getenv("AZURE_DEFAULT_MODEL")
		if model == "" {
			model = "gpt-4.1"
		}
	}

	deploymentName := p.mapDeploymentName(model)

	// Construir URL
	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
		p.endpoint, deploymentName, p.apiVersion)

	// Construir request
	azureReq := AzureRequest{
		Messages: []AzureMessage{
			{Role: "system", Content: req.SystemPrompt},
			{Role: "user", Content: req.Prompt},
		},
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Stream:      false,
	}

	requestBody, err := json.Marshal(azureReq)
	if err != nil {
		p.logger.Error("Error marshaling request", zap.String("request_id", requestID), zap.Error(err))
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	p.logger.Debug("Making HTTP request to Azure",
		zap.String("request_id", requestID),
		zap.String("url", url),
		zap.String("deployment", deploymentName))

	// Crear HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, fmt.Errorf("error creating HTTP request: %w", err)
	}

	// Configurar headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", p.apiKey)

	// Hacer la request
	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		p.logger.Error("HTTP request failed", zap.String("request_id", requestID), zap.Error(err))
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	// Leer respuesta
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		p.logger.Error("Azure API error",
			zap.String("request_id", requestID),
			zap.Int("status_code", resp.StatusCode),
			zap.String("response_body", string(body)))
		return nil, fmt.Errorf("Azure API error: %d - %s", resp.StatusCode, string(body))
	}

	// Parse respuesta
	var azureResp AzureResponse
	if err := json.Unmarshal(body, &azureResp); err != nil {
		return nil, fmt.Errorf("error unmarshaling response: %w", err)
	}

	// Extraer contenido - manejar posibles respuestas vacías de DeepSeek
	var content string
	if len(azureResp.Choices) > 0 {
		content = azureResp.Choices[0].Message.Content

		// Log adicional para debugging de DeepSeek
		if content == "" && strings.Contains(strings.ToLower(model), "deepseek") {
			p.logger.Warn("DeepSeek returned empty content",
				zap.String("request_id", requestID),
				zap.String("model", model),
				zap.String("finish_reason", azureResp.Choices[0].FinishReason),
				zap.Int("total_tokens", azureResp.Usage.TotalTokens),
				zap.String("raw_response", string(body)))
		}
	}

	totalDuration := time.Since(startTime)

	response := &GenerateResponse{
		Content:    content,
		Model:      model,
		Provider:   "azure-openai",
		TokensUsed: azureResp.Usage.TotalTokens,
		Duration:   totalDuration,
		Metadata: map[string]interface{}{
			"deployment":        deploymentName,
			"model":             model,
			"request_id":        requestID,
			"prompt_tokens":     azureResp.Usage.PromptTokens,
			"completion_tokens": azureResp.Usage.CompletionTokens,
		},
	}

	p.logger.Info("Azure OpenAI HTTP generation completed",
		zap.String("request_id", requestID),
		zap.Int("tokens_used", azureResp.Usage.TotalTokens),
		zap.Duration("duration", totalDuration))

	return response, nil
}

// StreamGenerate genera contenido con streaming usando HTTP directo
func (p *AzureProvider) StreamGenerate(ctx context.Context, req *GenerateRequest, stream chan<- *StreamChunk) error {
	defer close(stream)

	startTime := time.Now()
	requestID := fmt.Sprintf("azure-stream-%d", startTime.UnixNano())

	p.logger.Info("Starting Azure OpenAI HTTP streaming request", zap.String("request_id", requestID))

	model := req.Model
	if model == "" {
		model = "gpt-4.1"
	}

	deploymentName := p.mapDeploymentName(model)

	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
		p.endpoint, deploymentName, p.apiVersion)

	azureReq := AzureRequest{
		Messages: []AzureMessage{
			{Role: "system", Content: req.SystemPrompt},
			{Role: "user", Content: req.Prompt},
		},
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Stream:      true,
	}

	requestBody, err := json.Marshal(azureReq)
	if err != nil {
		return fmt.Errorf("error marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(requestBody))
	if err != nil {
		return fmt.Errorf("error creating HTTP request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", p.apiKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("Azure API error: %d - %s", resp.StatusCode, string(body))
	}

	// Procesar stream de Server-Sent Events
	index := 0
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")

			if data == "[DONE]" {
				break
			}

			var streamResp AzureStreamResponse
			if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
				continue // skip malformed chunks
			}

			if len(streamResp.Choices) > 0 && streamResp.Choices[0].Delta.Content != "" {
				stream <- &StreamChunk{
					Content:   streamResp.Choices[0].Delta.Content,
					Index:     index,
					Finished:  false,
					Timestamp: time.Now(),
				}
				index++
			}
		}
	}

	// Enviar chunk final
	stream <- &StreamChunk{
		Content:   "",
		Index:     index,
		Finished:  true,
		Timestamp: time.Now(),
	}

	p.logger.Info("Azure OpenAI HTTP streaming completed", zap.String("request_id", requestID))
	return nil
}

// ListModels lista los modelos disponibles
func (p *AzureProvider) ListModels() []Model {
	return []Model{
		{
			ID:           "DeepSeek-R1",
			Name:         "DeepSeek R1",
			Provider:     "azure-openai",
			MaxTokens:    20000,
			Description:  "DeepSeek R1 - Advanced reasoning model",
			Capabilities: []string{"text-generation", "code-generation", "reasoning", "analysis"},
		},
		{
			ID:           "DeepSeek-R1-0528",
			Name:         "DeepSeek R1 0528",
			Provider:     "azure-openai",
			MaxTokens:    20000,
			Description:  "DeepSeek R1 0528 version",
			Capabilities: []string{"text-generation", "code-generation", "reasoning", "analysis"},
		},
		{
			ID:           "DeepSeek-V3-0324",
			Name:         "DeepSeek V3 0324",
			Provider:     "azure-openai",
			MaxTokens:    20000,
			Description:  "DeepSeek V3 0324 version",
			Capabilities: []string{"text-generation", "code-generation", "reasoning"},
		},
		{
			ID:           "gpt-4.1",
			Name:         "GPT-4.1",
			Provider:     "azure-openai",
			MaxTokens:    32768,
			Description:  "GPT-4.1 - Latest GPT-4.1 version (2025-04-14)",
			Capabilities: []string{"text-generation", "code-generation", "analysis", "reasoning"},
		},
		{
			ID:           "gpt-4o",
			Name:         "GPT-4o",
			Provider:     "azure-openai",
			MaxTokens:    50000,
			Description:  "GPT-4o - Optimized GPT-4.1 (2024-11-20)",
			Capabilities: []string{"text-generation", "code-generation", "vision", "analysis"},
		},
		{
			ID:           "gpt-5-chat",
			Name:         "GPT-5 Chat",
			Provider:     "azure-openai",
			MaxTokens:    50000,
			Description:  "GPT-5 Chat - Next generation chat model",
			Capabilities: []string{"text-generation", "code-generation", "reasoning", "analysis"},
		},
		{
			ID:           "o4-mini",
			Name:         "O4 Mini",
			Provider:     "azure-openai",
			MaxTokens:    20000,
			Description:  "O4 Mini - Efficient reasoning model",
			Capabilities: []string{"text-generation", "code-generation", "reasoning"},
		},
		{
			ID:           "Llama-3.3-70B-Instruct",
			Name:         "Llama 3.3 70B Instruct",
			Provider:     "azure-openai",
			MaxTokens:    8192,
			Description:  "Meta's Llama 3.3 70B Instruct model",
			Capabilities: []string{"text-generation", "code-generation", "instruction-following"},
		},
		{
			ID:           "grok-3",
			Name:         "Grok 3",
			Provider:     "azure-openai",
			MaxTokens:    50000,
			Description:  "Grok 3 - xAI's advanced model",
			Capabilities: []string{"text-generation", "code-generation", "reasoning", "humor"},
		},
	}
}

// GetName obtiene el nombre del proveedor
func (p *AzureProvider) GetName() string {
	return "azure-openai-http"
}

// mapDeploymentName mapea el modelo al deployment name
func (p *AzureProvider) mapDeploymentName(model string) string {
	// Mapeo de modelos a deployment names exactos en Azure
	deploymentMap := map[string]string{
		"deepseek-r1":            "DeepSeek-R1",
		"deepseek-r1-0528":       "DeepSeek-R1-0528",
		"deepseek-v3":            "DeepSeek-V3-0324",
		"deepseek-v3-0324":       "DeepSeek-V3-0324",
		"gpt-4.1":                "gpt-4.1",
		"gpt-4o":                 "gpt-4o",
		"gpt4o":                  "gpt-4o",
		"gpt-5-chat":             "gpt-5-chat",
		"gpt5-chat":              "gpt-5-chat",
		"o4-mini":                "o4-mini",
		"o4mini":                 "o4-mini",
		"llama-3.3-70b":          "Llama-3.3-70B-Instruct",
		"llama-3.3-70b-instruct": "Llama-3.3-70B-Instruct",
		"grok-3":                 "grok-3",
		"grok3":                  "grok-3",
	}

	// Buscar en el mapa (case insensitive)
	modelLower := strings.ToLower(model)
	if deployment, ok := deploymentMap[modelLower]; ok {
		return deployment
	}

	// Si no está en el mapa, devolver tal cual
	return model
}
