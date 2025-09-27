package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"go.uber.org/zap"
)

// AWSProvider proveedor de AWS Bedrock
type AWSProvider struct {
	client *bedrockruntime.Client
	logger *zap.Logger
}

// NewAWSProvider crea un nuevo proveedor de AWS
func NewAWSProvider(client *bedrockruntime.Client, logger *zap.Logger) *AWSProvider {
	return &AWSProvider{
		client: client,
		logger: logger,
	}
}

// Generate genera contenido usando AWS Bedrock
func (p *AWSProvider) Generate(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
	startTime := time.Now()

	// Mapear modelo a ID de AWS
	modelID := p.mapModelID(req.Model)

	// Usar Converse API para modelos de Claude 3
	if strings.Contains(modelID, "claude-3") || strings.Contains(modelID, "claude-opus-4") ||
		strings.Contains(modelID, "claude-sonnet-4") || strings.Contains(modelID, "claude-3-7") {
		return p.generateWithConverse(ctx, req, modelID, startTime)
	}

	// Para otros modelos, usar InvokeModel tradicional
	return p.generateWithInvokeModel(ctx, req, modelID, startTime)
}

// generateWithConverse usa la Converse API para modelos de Claude
func (p *AWSProvider) generateWithConverse(ctx context.Context, req *GenerateRequest, modelID string, startTime time.Time) (*GenerateResponse, error) {
	// Construir mensajes para Converse API
	messages := []types.Message{
		{
			Role: types.ConversationRoleUser,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberText{
					Value: req.Prompt,
				},
			},
		},
	}

	// Configurar sistema si está presente
	var systemPrompts []types.SystemContentBlock
	if req.SystemPrompt != "" {
		systemPrompts = append(systemPrompts, &types.SystemContentBlockMemberText{
			Value: req.SystemPrompt,
		})
	}

	// Configurar parámetros de inferencia
	inferenceConfig := &types.InferenceConfiguration{
		MaxTokens:   aws.Int32(int32(req.MaxTokens)),
		Temperature: aws.Float32(float32(req.Temperature)),
		TopP:        aws.Float32(0.9),
	}

	// Crear input para Converse
	input := &bedrockruntime.ConverseInput{
		ModelId:         aws.String(modelID),
		Messages:        messages,
		InferenceConfig: inferenceConfig,
	}

	// Agregar sistema si existe
	if len(systemPrompts) > 0 {
		input.System = systemPrompts
	}

	// Invocar el modelo usando Converse
	result, err := p.client.Converse(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to converse with model: %w", err)
	}

	// Extraer contenido de la respuesta
	var content string
	if result.Output != nil {
		switch v := result.Output.(type) {
		case *types.ConverseOutputMemberMessage:
			if len(v.Value.Content) > 0 {
				if textBlock, ok := v.Value.Content[0].(*types.ContentBlockMemberText); ok {
					content = textBlock.Value
				}
			}
		}
	}

	// Calcular tokens usados
	tokensUsed := 0
	if result.Usage != nil {
		if result.Usage.InputTokens != nil {
			tokensUsed += int(*result.Usage.InputTokens)
		}
		if result.Usage.OutputTokens != nil {
			tokensUsed += int(*result.Usage.OutputTokens)
		}
	}

	return &GenerateResponse{
		Content:    content,
		Model:      req.Model,
		Provider:   "aws-bedrock",
		TokensUsed: tokensUsed,
		Duration:   time.Since(startTime),
		Metadata: map[string]interface{}{
			"model_id": modelID,
			"api":      "converse",
		},
	}, nil
}

// generateWithInvokeModel usa la API tradicional para otros modelos
func (p *AWSProvider) generateWithInvokeModel(ctx context.Context, req *GenerateRequest, modelID string, startTime time.Time) (*GenerateResponse, error) {
	// Construir el prompt según el modelo
	var requestBody []byte
	var err error

	if strings.Contains(modelID, "llama") {
		// Formato para Llama
		requestBody, err = json.Marshal(map[string]interface{}{
			"prompt":      req.Prompt,
			"max_gen_len": req.MaxTokens,
			"temperature": req.Temperature,
		})
	} else if strings.Contains(modelID, "titan") {
		// Formato para Titan
		requestBody, err = json.Marshal(map[string]interface{}{
			"inputText": req.Prompt,
			"textGenerationConfig": map[string]interface{}{
				"maxTokenCount": req.MaxTokens,
				"temperature":   req.Temperature,
			},
		})
	} else if strings.Contains(modelID, "nova") {
		// Formato para Nova
		requestBody, err = json.Marshal(map[string]interface{}{
			"messages": []map[string]interface{}{
				{
					"role":    "user",
					"content": []map[string]string{{"text": req.Prompt}},
				},
			},
			"inferenceConfig": map[string]interface{}{
				"max_new_tokens": req.MaxTokens,
				"temperature":    req.Temperature,
			},
		})
	} else {
		// Formato genérico
		requestBody, err = json.Marshal(map[string]interface{}{
			"inputText": req.Prompt,
			"textGenerationConfig": map[string]interface{}{
				"maxTokenCount": req.MaxTokens,
				"temperature":   req.Temperature,
			},
		})
	}

	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Invocar el modelo
	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelID),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		Body:        requestBody,
	}

	result, err := p.client.InvokeModel(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to invoke model: %w", err)
	}

	// Parsear respuesta
	var response map[string]interface{}
	if err := json.Unmarshal(result.Body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Extraer contenido según el formato de respuesta
	var content string
	if completion, ok := response["completion"].(string); ok {
		content = completion
	} else if generation, ok := response["generation"].(string); ok {
		content = generation
	} else if results, ok := response["results"].([]interface{}); ok && len(results) > 0 {
		if text, ok := results[0].(map[string]interface{})["outputText"].(string); ok {
			content = text
		}
	} else if output, ok := response["output"].(map[string]interface{}); ok {
		// Para Nova
		if message, ok := output["message"].(map[string]interface{}); ok {
			if contentArray, ok := message["content"].([]interface{}); ok && len(contentArray) > 0 {
				if textContent, ok := contentArray[0].(map[string]interface{}); ok {
					if text, ok := textContent["text"].(string); ok {
						content = text
					}
				}
			}
		}
	}

	return &GenerateResponse{
		Content:    content,
		Model:      req.Model,
		Provider:   "aws-bedrock",
		TokensUsed: len(content) / 4, // Aproximación
		Duration:   time.Since(startTime),
		Metadata: map[string]interface{}{
			"model_id": modelID,
			"api":      "invoke_model",
		},
	}, nil
}

// StreamGenerate genera contenido con streaming
func (p *AWSProvider) StreamGenerate(ctx context.Context, req *GenerateRequest, stream chan<- *StreamChunk) error {
	defer close(stream)

	modelID := p.mapModelID(req.Model)

	// Usar ConverseStream para modelos de Claude 3
	if strings.Contains(modelID, "claude-3") || strings.Contains(modelID, "claude-opus-4") ||
		strings.Contains(modelID, "claude-sonnet-4") || strings.Contains(modelID, "claude-3-7") {
		return p.streamWithConverseStream(ctx, req, modelID, stream)
	}

	// Para otros modelos, usar InvokeModelWithResponseStream
	return p.streamWithInvokeModel(ctx, req, modelID, stream)
}

// streamWithConverseStream usa la ConverseStream API para modelos de Claude
func (p *AWSProvider) streamWithConverseStream(ctx context.Context, req *GenerateRequest, modelID string, stream chan<- *StreamChunk) error {
	// Construir mensajes para Converse API
	messages := []types.Message{
		{
			Role: types.ConversationRoleUser,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberText{
					Value: req.Prompt,
				},
			},
		},
	}

	// Configurar sistema si está presente
	var systemPrompts []types.SystemContentBlock
	if req.SystemPrompt != "" {
		systemPrompts = append(systemPrompts, &types.SystemContentBlockMemberText{
			Value: req.SystemPrompt,
		})
	}

	// Configurar parámetros de inferencia
	inferenceConfig := &types.InferenceConfiguration{
		MaxTokens:   aws.Int32(int32(req.MaxTokens)),
		Temperature: aws.Float32(float32(req.Temperature)),
		TopP:        aws.Float32(0.9),
	}

	// Crear input para ConverseStream
	input := &bedrockruntime.ConverseStreamInput{
		ModelId:         aws.String(modelID),
		Messages:        messages,
		InferenceConfig: inferenceConfig,
	}

	// Agregar sistema si existe
	if len(systemPrompts) > 0 {
		input.System = systemPrompts
	}

	// Invocar con streaming
	output, err := p.client.ConverseStream(ctx, input)
	if err != nil {
		return fmt.Errorf("failed to converse stream with model: %w", err)
	}

	// Procesar stream
	eventStream := output.GetStream()
	defer eventStream.Close()

	index := 0
	for event := range eventStream.Events() {
		switch v := event.(type) {
		case *types.ConverseStreamOutputMemberContentBlockDelta:
			// Extraer texto del delta
			if v.Value.Delta != nil {
				if textDelta, ok := v.Value.Delta.(*types.ContentBlockDeltaMemberText); ok {
					// Enviar chunk
					stream <- &StreamChunk{
						Content:   textDelta.Value,
						Index:     index,
						Finished:  false,
						Timestamp: time.Now(),
					}
					index++
				}
			}

		case *types.ConverseStreamOutputMemberMessageStop:
			// Mensaje completado
			p.logger.Debug("Message completed")

		case *types.ConverseStreamOutputMemberMetadata:
			// Metadata del stream
			p.logger.Debug("Stream metadata received")

		default:
			// Otros tipos de eventos
			p.logger.Debug("Received event", zap.String("type", fmt.Sprintf("%T", v)))
		}
	}

	// Enviar chunk final
	stream <- &StreamChunk{
		Content:   "",
		Index:     index,
		Finished:  true,
		Timestamp: time.Now(),
	}

	return eventStream.Err()
}

// streamWithInvokeModel usa la API tradicional de streaming para otros modelos
func (p *AWSProvider) streamWithInvokeModel(ctx context.Context, req *GenerateRequest, modelID string, stream chan<- *StreamChunk) error {
	// Construir request para streaming según el modelo
	var requestBody []byte
	var err error

	if strings.Contains(modelID, "llama") {
		// Formato para Llama (no soporta streaming nativo)
		// Usar generación normal y simular streaming
		resp, err := p.generateWithInvokeModel(ctx, req, modelID, time.Now())
		if err != nil {
			return err
		}

		// Simular streaming dividiendo el contenido
		words := strings.Fields(resp.Content)
		for i, word := range words {
			stream <- &StreamChunk{
				Content:   word + " ",
				Index:     i,
				Finished:  false,
				Timestamp: time.Now(),
			}
			time.Sleep(10 * time.Millisecond) // Simular delay
		}

		stream <- &StreamChunk{
			Content:   "",
			Index:     len(words),
			Finished:  true,
			Timestamp: time.Now(),
		}
		return nil
	}

	// Para modelos que soportan streaming nativo
	requestBody, err = json.Marshal(map[string]interface{}{
		"inputText": req.Prompt,
		"textGenerationConfig": map[string]interface{}{
			"maxTokenCount": req.MaxTokens,
			"temperature":   req.Temperature,
		},
	})

	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	input := &bedrockruntime.InvokeModelWithResponseStreamInput{
		ModelId:     aws.String(modelID),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		Body:        requestBody,
	}

	// Invocar con streaming
	output, err := p.client.InvokeModelWithResponseStream(ctx, input)
	if err != nil {
		return fmt.Errorf("failed to invoke model with stream: %w", err)
	}

	// Procesar stream
	eventStream := output.GetStream()
	defer eventStream.Close()

	index := 0
	for event := range eventStream.Events() {
		switch v := event.(type) {
		case *types.ResponseStreamMemberChunk:
			// Parsear chunk
			var chunk map[string]interface{}
			if err := json.Unmarshal(v.Value.Bytes, &chunk); err != nil {
				p.logger.Error("Failed to unmarshal chunk", zap.Error(err))
				continue
			}

			// Extraer texto del chunk
			var text string
			if completion, ok := chunk["completion"].(string); ok {
				text = completion
			} else if delta, ok := chunk["delta"].(map[string]interface{}); ok {
				if deltaText, ok := delta["text"].(string); ok {
					text = deltaText
				}
			} else if outputText, ok := chunk["outputText"].(string); ok {
				text = outputText
			}

			// Enviar chunk
			stream <- &StreamChunk{
				Content:   text,
				Index:     index,
				Finished:  false,
				Timestamp: time.Now(),
			}
			index++

		default:
			// Otros tipos de eventos
			p.logger.Debug("Received event", zap.String("type", fmt.Sprintf("%T", v)))
		}
	}

	// Enviar chunk final
	stream <- &StreamChunk{
		Content:   "",
		Index:     index,
		Finished:  true,
		Timestamp: time.Now(),
	}

	return eventStream.Err()
}

// ListModels lista los modelos disponibles en AWS Bedrock
func (p *AWSProvider) ListModels() []Model {
	return []Model{
		// Claude Models
		{
			ID:           "us.anthropic.claude-3-haiku-20240307-v1:0",
			Name:         "Claude 3 Haiku",
			Provider:     "aws-bedrock",
			MaxTokens:    200000,
			Description:  "Claude 3 Haiku - Fast and efficient",
			Capabilities: []string{"text-generation", "code-generation"},
		},
		{
			ID:           "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
			Name:         "Claude 3.5 Sonnet",
			Provider:     "aws-bedrock",
			MaxTokens:    200000,
			Description:  "Claude 3.5 Sonnet - Advanced capabilities",
			Capabilities: []string{"text-generation", "code-generation", "analysis"},
		},
		{
			ID:           "us.anthropic.claude-3-sonnet-20240229-v1:0",
			Name:         "Claude 3 Sonnet",
			Provider:     "aws-bedrock",
			MaxTokens:    200000,
			Description:  "Claude 3 Sonnet - Balanced performance",
			Capabilities: []string{"text-generation", "code-generation", "analysis"},
		},
		{
			ID:           "us.anthropic.claude-3-opus-20240229-v1:0",
			Name:         "Claude 3 Opus",
			Provider:     "aws-bedrock",
			MaxTokens:    200000,
			Description:  "Claude 3 Opus - Most capable Claude 3 model",
			Capabilities: []string{"text-generation", "code-generation", "analysis", "reasoning"},
		},
		{
			ID:           "us.anthropic.claude-3-5-haiku-20241022-v1:0",
			Name:         "Claude 3.5 Haiku",
			Provider:     "aws-bedrock",
			MaxTokens:    200000,
			Description:  "Claude 3.5 Haiku - Fast with improved capabilities",
			Capabilities: []string{"text-generation", "code-generation"},
		},
		{
			ID:           "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
			Name:         "Claude 3.5 Sonnet v2",
			Provider:     "aws-bedrock",
			MaxTokens:    200000,
			Description:  "Claude 3.5 Sonnet v2 - Latest Sonnet version",
			Capabilities: []string{"text-generation", "code-generation", "analysis", "vision"},
		},
		{
			ID:           "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
			Name:         "Claude 3.7 Sonnet",
			Provider:     "aws-bedrock",
			MaxTokens:    200000,
			Description:  "Claude 3.7 Sonnet - Enhanced reasoning",
			Capabilities: []string{"text-generation", "code-generation", "analysis", "reasoning"},
		},
		{
			ID:           "us.anthropic.claude-opus-4-20250514-v1:0",
			Name:         "Claude Opus 4",
			Provider:     "aws-bedrock",
			MaxTokens:    200000,
			Description:  "Claude Opus 4 - Next generation Opus",
			Capabilities: []string{"text-generation", "code-generation", "analysis", "reasoning", "vision"},
		},
		{
			ID:           "us.anthropic.claude-opus-4-1-20250805-v1:0",
			Name:         "Claude Opus 4.1",
			Provider:     "aws-bedrock",
			MaxTokens:    200000,
			Description:  "Claude Opus 4.1 - Latest Opus version",
			Capabilities: []string{"text-generation", "code-generation", "analysis", "reasoning", "vision"},
		},
		{
			ID:           "us.anthropic.claude-sonnet-4-20250514-v1:0",
			Name:         "Claude Sonnet 4",
			Provider:     "aws-bedrock",
			MaxTokens:    200000,
			Description:  "Claude Sonnet 4 - Next generation Sonnet",
			Capabilities: []string{"text-generation", "code-generation", "analysis", "reasoning"},
		},

		// Meta Llama Models
		{
			ID:           "us.meta.llama3-2-11b-instruct-v1:0",
			Name:         "Llama 3.2 11B Instruct",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Meta Llama 3.2 11B Instruct",
			Capabilities: []string{"text-generation", "code-generation"},
		},
		{
			ID:           "us.meta.llama3-2-90b-instruct-v1:0",
			Name:         "Llama 3.2 90B Instruct",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Meta Llama 3.2 90B Instruct",
			Capabilities: []string{"text-generation", "code-generation", "analysis"},
		},
		{
			ID:           "us.meta.llama3-2-3b-instruct-v1:0",
			Name:         "Llama 3.2 3B Instruct",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Meta Llama 3.2 3B Instruct",
			Capabilities: []string{"text-generation"},
		},
		{
			ID:           "us.meta.llama3-2-1b-instruct-v1:0",
			Name:         "Llama 3.2 1B Instruct",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Meta Llama 3.2 1B Instruct",
			Capabilities: []string{"text-generation"},
		},
		{
			ID:           "us.meta.llama3-1-8b-instruct-v1:0",
			Name:         "Llama 3.1 8B Instruct",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Meta Llama 3.1 8B Instruct",
			Capabilities: []string{"text-generation", "code-generation"},
		},
		{
			ID:           "us.meta.llama3-1-70b-instruct-v1:0",
			Name:         "Llama 3.1 70B Instruct",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Meta Llama 3.1 70B Instruct",
			Capabilities: []string{"text-generation", "code-generation", "analysis"},
		},
		{
			ID:           "us.meta.llama3-3-70b-instruct-v1:0",
			Name:         "Llama 3.3 70B Instruct",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Meta Llama 3.3 70B Instruct",
			Capabilities: []string{"text-generation", "code-generation", "analysis"},
		},
		{
			ID:           "us.meta.llama4-maverick-17b-instruct-v1:0",
			Name:         "Llama 4 Maverick 17B",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Meta Llama 4 Maverick 17B Instruct",
			Capabilities: []string{"text-generation", "code-generation"},
		},
		{
			ID:           "us.meta.llama4-scout-17b-instruct-v1:0",
			Name:         "Llama 4 Scout 17B",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Meta Llama 4 Scout 17B Instruct",
			Capabilities: []string{"text-generation", "code-generation"},
		},

		// Amazon Nova Models
		{
			ID:           "us.amazon.nova-premier-v1:0",
			Name:         "Nova Premier",
			Provider:     "aws-bedrock",
			MaxTokens:    300000,
			Description:  "Amazon Nova Premier - Most capable Nova model",
			Capabilities: []string{"text-generation", "code-generation", "analysis", "reasoning", "vision"},
		},
		{
			ID:           "us.amazon.nova-pro-v1:0",
			Name:         "Nova Pro",
			Provider:     "aws-bedrock",
			MaxTokens:    300000,
			Description:  "Amazon Nova Pro - Professional grade",
			Capabilities: []string{"text-generation", "code-generation", "analysis", "vision"},
		},
		{
			ID:           "us.amazon.nova-micro-v1:0",
			Name:         "Nova Micro",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Amazon Nova Micro - Fastest Nova model",
			Capabilities: []string{"text-generation"},
		},
		{
			ID:           "us.amazon.nova-lite-v1:0",
			Name:         "Nova Lite",
			Provider:     "aws-bedrock",
			MaxTokens:    300000,
			Description:  "Amazon Nova Lite - Efficient and capable",
			Capabilities: []string{"text-generation", "code-generation", "vision"},
		},

		// DeepSeek Models
		{
			ID:           "us.deepseek.r1-v1:0",
			Name:         "DeepSeek R1",
			Provider:     "aws-bedrock",
			MaxTokens:    65536,
			Description:  "DeepSeek R1 - Advanced reasoning model",
			Capabilities: []string{"text-generation", "code-generation", "reasoning", "analysis"},
		},

		// Mistral Models
		{
			ID:           "us.mistral.pixtral-large-2502-v1:0",
			Name:         "Pixtral Large 25.02",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Mistral Pixtral Large 25.02 - Multimodal model",
			Capabilities: []string{"text-generation", "code-generation", "vision", "analysis"},
		},

		// Writer Models
		{
			ID:           "us.writer.palmyra-x4-v1:0",
			Name:         "Palmyra X4",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Writer Palmyra X4",
			Capabilities: []string{"text-generation", "code-generation"},
		},
		{
			ID:           "us.writer.palmyra-x5-v1:0",
			Name:         "Palmyra X5",
			Provider:     "aws-bedrock",
			MaxTokens:    128000,
			Description:  "Writer Palmyra X5",
			Capabilities: []string{"text-generation", "code-generation", "analysis"},
		},

		// TwelveLabs Models
		{
			ID:           "us.twelvelabs.pegasus-1-2-v1:0",
			Name:         "Pegasus v1.2",
			Provider:     "aws-bedrock",
			MaxTokens:    8192,
			Description:  "TwelveLabs Pegasus v1.2 - Video understanding",
			Capabilities: []string{"video-analysis", "text-generation"},
		},

		// Stability AI Image Models
		{
			ID:           "us.stability.stable-image-control-sketch-v1:0",
			Name:         "Stable Image Control Sketch",
			Provider:     "aws-bedrock",
			MaxTokens:    0,
			Description:  "Stability AI - Sketch to image",
			Capabilities: []string{"image-generation", "sketch-to-image"},
		},
		{
			ID:           "us.stability.stable-image-control-structure-v1:0",
			Name:         "Stable Image Control Structure",
			Provider:     "aws-bedrock",
			MaxTokens:    0,
			Description:  "Stability AI - Structure control",
			Capabilities: []string{"image-generation", "structure-control"},
		},
		{
			ID:           "us.stability.stable-image-erase-object-v1:0",
			Name:         "Stable Image Erase Object",
			Provider:     "aws-bedrock",
			MaxTokens:    0,
			Description:  "Stability AI - Object removal",
			Capabilities: []string{"image-editing", "object-removal"},
		},
		{
			ID:           "us.stability.stable-image-inpaint-v1:0",
			Name:         "Stable Image Inpaint",
			Provider:     "aws-bedrock",
			MaxTokens:    0,
			Description:  "Stability AI - Image inpainting",
			Capabilities: []string{"image-editing", "inpainting"},
		},
		{
			ID:           "us.stability.stable-image-remove-background-v1:0",
			Name:         "Stable Image Remove Background",
			Provider:     "aws-bedrock",
			MaxTokens:    0,
			Description:  "Stability AI - Background removal",
			Capabilities: []string{"image-editing", "background-removal"},
		},
		{
			ID:           "us.stability.stable-image-search-recolor-v1:0",
			Name:         "Stable Image Search and Recolor",
			Provider:     "aws-bedrock",
			MaxTokens:    0,
			Description:  "Stability AI - Search and recolor",
			Capabilities: []string{"image-editing", "recoloring"},
		},
		{
			ID:           "us.stability.stable-image-search-replace-v1:0",
			Name:         "Stable Image Search and Replace",
			Provider:     "aws-bedrock",
			MaxTokens:    0,
			Description:  "Stability AI - Search and replace",
			Capabilities: []string{"image-editing", "object-replacement"},
		},
		{
			ID:           "us.stability.stable-image-style-guide-v1:0",
			Name:         "Stable Image Style Guide",
			Provider:     "aws-bedrock",
			MaxTokens:    0,
			Description:  "Stability AI - Style guide",
			Capabilities: []string{"image-generation", "style-transfer"},
		},
		{
			ID:           "us.stability.stable-style-transfer-v1:0",
			Name:         "Stable Style Transfer",
			Provider:     "aws-bedrock",
			MaxTokens:    0,
			Description:  "Stability AI - Style transfer",
			Capabilities: []string{"image-editing", "style-transfer"},
		},

		// Amazon Titan Models
		{
			ID:           "amazon.titan-text-express-v1",
			Name:         "Titan Text Express",
			Provider:     "aws-bedrock",
			MaxTokens:    8192,
			Description:  "Amazon Titan Text Express",
			Capabilities: []string{"text-generation"},
		},

		// Global Models (cross-region)
		{
			ID:           "global.anthropic.claude-sonnet-4-20250514-v1:0",
			Name:         "Claude Sonnet 4 (Global)",
			Provider:     "aws-bedrock",
			MaxTokens:    200000,
			Description:  "Claude Sonnet 4 - Global cross-region",
			Capabilities: []string{"text-generation", "code-generation", "analysis", "reasoning"},
		},
	}
}

// GetName obtiene el nombre del proveedor
func (p *AWSProvider) GetName() string {
	return "aws-bedrock"
}

// mapModelID mapea el nombre del modelo al ID de AWS
func (p *AWSProvider) mapModelID(model string) string {
	// Normalizar el nombre del modelo (quitar espacios extras, convertir a minúsculas para comparación)
	normalizedModel := strings.ToLower(strings.TrimSpace(model))

	// Mapeo de nombres cortos a IDs completos con inference profiles
	modelMap := map[string]string{
		// Claude models - usando inference profiles
		"claude-3-sonnet":   "us.anthropic.claude-3-sonnet-20240229-v1:0",
		"claude 3 sonnet":   "us.anthropic.claude-3-sonnet-20240229-v1:0",
		"claude-3-haiku":    "us.anthropic.claude-3-haiku-20240307-v1:0",
		"claude 3 haiku":    "us.anthropic.claude-3-haiku-20240307-v1:0",
		"claude-3-opus":     "us.anthropic.claude-3-opus-20240229-v1:0",
		"claude 3 opus":     "us.anthropic.claude-3-opus-20240229-v1:0",
		"claude-3-5-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
		"claude 3.5 sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
		"claude-3-5-haiku":  "us.anthropic.claude-3-5-haiku-20241022-v1:0",
		"claude 3.5 haiku":  "us.anthropic.claude-3-5-haiku-20241022-v1:0",
		"claude-3-7-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
		"claude 3.7 sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
		"claude-opus-4":     "us.anthropic.claude-opus-4-20250514-v1:0",
		"claude opus 4":     "us.anthropic.claude-opus-4-20250514-v1:0",
		"claude-opus-4-1":   "us.anthropic.claude-opus-4-1-20250805-v1:0",
		"claude opus 4.1":   "us.anthropic.claude-opus-4-1-20250805-v1:0",
		"claude-sonnet-4":   "us.anthropic.claude-sonnet-4-20250514-v1:0",
		"claude sonnet 4":   "us.anthropic.claude-sonnet-4-20250514-v1:0",

		// Mapeo por ID exacto del modelo (como viene del endpoint /models)
		"anthropic.claude-3-sonnet-20240229-v1:0": "us.anthropic.claude-3-sonnet-20240229-v1:0",
		"anthropic.claude-3-haiku-20240307-v1:0":  "us.anthropic.claude-3-haiku-20240307-v1:0",
		"anthropic.claude-3-opus-20240229-v1:0":   "us.anthropic.claude-3-opus-20240229-v1:0",

		// Llama models - usando inference profiles
		"llama-3-70b":                   "us.meta.llama3-70b-instruct-v1:0",
		"llama 3 70b":                   "us.meta.llama3-70b-instruct-v1:0",
		"llama-3-1-8b":                  "us.meta.llama3-1-8b-instruct-v1:0",
		"llama 3.1 8b":                  "us.meta.llama3-1-8b-instruct-v1:0",
		"llama-3-1-70b":                 "us.meta.llama3-1-70b-instruct-v1:0",
		"llama 3.1 70b":                 "us.meta.llama3-1-70b-instruct-v1:0",
		"llama-3-3-70b":                 "us.meta.llama3-3-70b-instruct-v1:0",
		"llama 3.3 70b":                 "us.meta.llama3-3-70b-instruct-v1:0",
		"llama-3-2-1b":                  "us.meta.llama3-2-1b-instruct-v1:0",
		"llama 3.2 1b":                  "us.meta.llama3-2-1b-instruct-v1:0",
		"llama-3-2-3b":                  "us.meta.llama3-2-3b-instruct-v1:0",
		"llama 3.2 3b":                  "us.meta.llama3-2-3b-instruct-v1:0",
		"llama-3-2-11b":                 "us.meta.llama3-2-11b-instruct-v1:0",
		"llama 3.2 11b":                 "us.meta.llama3-2-11b-instruct-v1:0",
		"llama-3-2-90b":                 "us.meta.llama3-2-90b-instruct-v1:0",
		"llama 3.2 90b":                 "us.meta.llama3-2-90b-instruct-v1:0",
		"meta.llama3-70b-instruct-v1:0": "us.meta.llama3-70b-instruct-v1:0",

		// Amazon models - usando inference profiles
		"titan-express":                "amazon.titan-text-express-v1",
		"titan text express":           "amazon.titan-text-express-v1",
		"nova-micro":                   "us.amazon.nova-micro-v1:0",
		"nova micro":                   "us.amazon.nova-micro-v1:0",
		"nova-lite":                    "us.amazon.nova-lite-v1:0",
		"nova lite":                    "us.amazon.nova-lite-v1:0",
		"nova-pro":                     "us.amazon.nova-pro-v1:0",
		"nova pro":                     "us.amazon.nova-pro-v1:0",
		"nova-premier":                 "us.amazon.nova-premier-v1:0",
		"nova premier":                 "us.amazon.nova-premier-v1:0",
		"amazon.titan-text-express-v1": "amazon.titan-text-express-v1",

		// DeepSeek models - usando inference profiles
		"deepseek-r1": "us.deepseek.r1-v1:0",
		"deepseek r1": "us.deepseek.r1-v1:0",

		// Mistral models - usando inference profiles
		"pixtral-large": "us.mistral.pixtral-large-2502-v1:0",
		"pixtral large": "us.mistral.pixtral-large-2502-v1:0",
	}

	// Buscar en el mapa con el nombre normalizado
	if id, ok := modelMap[normalizedModel]; ok {
		return id
	}

	// Si ya es un ID completo con "us." o un ARN, usarlo tal cual
	if strings.HasPrefix(model, "us.") || strings.HasPrefix(model, "arn:") {
		return model
	}

	// Si es un ID sin el prefijo "us.", intentar agregarlo
	if strings.Contains(model, ".") && !strings.HasPrefix(model, "us.") {
		// Verificar si es un modelo de Anthropic, Meta, Amazon, etc.
		if strings.HasPrefix(model, "anthropic.") ||
			strings.HasPrefix(model, "meta.") ||
			strings.HasPrefix(model, "deepseek.") ||
			strings.HasPrefix(model, "mistral.") {
			return "us." + model
		}
	}

	// Si no se encuentra, devolver el modelo tal cual
	return model
}
