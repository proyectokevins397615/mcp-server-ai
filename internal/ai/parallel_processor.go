package ai

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"kevins-tools/mcp-server-ai/internal/pool"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"
)

// Métricas para el procesador paralelo
var (
	parallelRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "ai_parallel_requests_total",
		Help: "Total number of parallel AI requests",
	}, []string{"provider", "model", "status"})

	parallelRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "ai_parallel_request_duration_seconds",
		Help:    "Duration of parallel AI requests",
		Buckets: prometheus.ExponentialBuckets(0.1, 2, 10),
	}, []string{"provider", "model"})

	parallelBatchSize = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "ai_parallel_batch_size",
		Help:    "Size of parallel processing batches",
		Buckets: []float64{1, 10, 50, 100, 500, 1000, 5000, 10000},
	}, []string{"provider"})

	parallelConcurrentRequests = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "ai_parallel_concurrent_requests",
		Help: "Number of concurrent AI requests",
	}, []string{"provider"})
)

// ParallelProcessor procesa requests de IA en paralelo usando worker pools
type ParallelProcessor struct {
	manager    *Manager
	workerPool *pool.WorkerPool
	logger     *zap.Logger

	// Control de concurrencia
	maxConcurrent int32
	current       int32

	// Buffers reutilizables
	responsePool *sync.Pool
	bufferPool   *sync.Pool

	// Progress tracking
	progressCallbacks []func(completed, total int, details map[string]interface{})
	mu                sync.RWMutex

	// Estadísticas
	totalProcessed uint64
	totalFailed    uint64
	totalDuration  int64
}

// AITask implementa pool.Task para procesar requests de IA
type AITask struct {
	id        string
	request   *GenerateRequest
	response  chan *GenerateResponse
	error     chan error
	processor *ParallelProcessor
	startTime time.Time
	priority  int
}

func (t *AITask) Execute(ctx context.Context) error {
	// Incrementar contador de requests concurrentes
	current := atomic.AddInt32(&t.processor.current, 1)
	parallelConcurrentRequests.WithLabelValues(t.request.Provider).Set(float64(current))
	defer func() {
		atomic.AddInt32(&t.processor.current, -1)
		parallelConcurrentRequests.WithLabelValues(t.request.Provider).Dec()
	}()

	// Obtener buffer del pool
	buffer := t.processor.bufferPool.Get().([]byte)
	defer t.processor.bufferPool.Put(buffer)

	// Ejecutar request
	resp, err := t.processor.manager.Generate(ctx, t.request)

	// Registrar métricas
	duration := time.Since(t.startTime)
	status := "success"
	if err != nil {
		status = "error"
		atomic.AddUint64(&t.processor.totalFailed, 1)
	} else {
		atomic.AddUint64(&t.processor.totalProcessed, 1)
	}

	parallelRequestsTotal.WithLabelValues(t.request.Provider, t.request.Model, status).Inc()
	parallelRequestDuration.WithLabelValues(t.request.Provider, t.request.Model).Observe(duration.Seconds())

	// Enviar resultado
	if err != nil {
		select {
		case t.error <- err:
		case <-ctx.Done():
			return ctx.Err()
		}
	} else {
		select {
		case t.response <- resp:
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return nil
}

func (t *AITask) GetID() string {
	return t.id
}

func (t *AITask) GetPriority() int {
	return t.priority
}

// NewParallelProcessor crea un nuevo procesador paralelo
func NewParallelProcessor(manager *Manager, logger *zap.Logger) *ParallelProcessor {
	// Leer configuración del entorno
	minWorkers := getEnvAsInt("POOL_MIN_WORKERS", 32)
	maxWorkers := getEnvAsInt("POOL_MAX_WORKERS", 512)
	queueSize := getEnvAsInt("POOL_QUEUE_SIZE", 100000)
	scaleInterval := getEnvAsDuration("POOL_SCALE_INTERVAL", 5*time.Second)
	utilizationTarget := getEnvAsFloat("POOL_UTILIZATION_TARGET", 0.75)
	maxConcurrent := getEnvAsInt("POOL_MAX_CONCURRENT", 10000)
	bufferSize := getEnvAsInt("POOL_BUFFER_SIZE", 65536)

	workerPool := pool.NewWorkerPool(pool.Config{
		Name:              "ai-parallel-processor",
		MinWorkers:        minWorkers,
		MaxWorkers:        maxWorkers,
		QueueSize:         queueSize,
		ScaleInterval:     scaleInterval,
		UtilizationTarget: utilizationTarget,
		Logger:            logger,
	})

	processor := &ParallelProcessor{
		manager:       manager,
		workerPool:    workerPool,
		logger:        logger,
		maxConcurrent: int32(maxConcurrent),
		responsePool: &sync.Pool{
			New: func() interface{} {
				return &GenerateResponse{}
			},
		},
		bufferPool: &sync.Pool{
			New: func() interface{} {
				return make([]byte, bufferSize)
			},
		},
		progressCallbacks: make([]func(int, int, map[string]interface{}), 0),
	}

	logger.Info("ParallelProcessor initialized with WorkerPool",
		zap.Int("min_workers", minWorkers),
		zap.Int("max_workers", maxWorkers),
		zap.Int("queue_size", queueSize),
		zap.Int("max_concurrent", maxConcurrent),
		zap.Float64("utilization_target", utilizationTarget))

	// Registrar callback de progreso con el pool
	workerPool.RegisterProgressCallback(func(completed, total int) {
		processor.notifyProgress(completed, total)
	})

	return processor
}

// ProcessBatch procesa un batch de requests en paralelo
func (p *ParallelProcessor) ProcessBatch(ctx context.Context, requests []*GenerateRequest) ([]*GenerateResponse, error) {
	batchSize := len(requests)
	if batchSize == 0 {
		return nil, fmt.Errorf("empty batch")
	}

	p.logger.Info("Processing batch",
		zap.Int("batch_size", batchSize))

	// Registrar tamaño del batch
	parallelBatchSize.WithLabelValues("batch").Observe(float64(batchSize))

	// Crear canales para respuestas
	responses := make([]*GenerateResponse, batchSize)
	errors := make([]error, batchSize)

	// Usar errgroup para control de errores
	g, ctx := errgroup.WithContext(ctx)

	// Crear y enviar tareas al pool
	tasks := make([]pool.Task, batchSize)
	for i, req := range requests {
		idx := i // Captura para el closure

		respChan := make(chan *GenerateResponse, 1)
		errChan := make(chan error, 1)

		task := &AITask{
			id:        fmt.Sprintf("batch_%d_task_%d", time.Now().UnixNano(), idx),
			request:   req,
			response:  respChan,
			error:     errChan,
			processor: p,
			startTime: time.Now(),
			priority:  1,
		}

		tasks[idx] = task

		// Goroutine para recolectar resultados
		g.Go(func() error {
			select {
			case resp := <-respChan:
				responses[idx] = resp
				return nil
			case err := <-errChan:
				errors[idx] = err
				return err
			case <-ctx.Done():
				return ctx.Err()
			}
		})
	}

	// Enviar todas las tareas al pool
	if err := p.workerPool.SubmitBatch(tasks); err != nil {
		return nil, fmt.Errorf("failed to submit batch: %w", err)
	}

	// Esperar a que todas las tareas terminen
	err := g.Wait()

	// Verificar si hubo errores
	if err != nil {
		// Recolectar todos los errores
		var failedCount int
		for _, e := range errors {
			if e != nil {
				failedCount++
			}
		}
		p.logger.Error("Batch processing failed",
			zap.Error(err),
			zap.Int("failed_tasks", failedCount),
			zap.Int("total_tasks", batchSize))

		// Devolver respuestas parciales si las hay
		return responses, fmt.Errorf("batch processing failed: %d/%d tasks failed", failedCount, batchSize)
	}

	p.logger.Info("Batch processing completed",
		zap.Int("batch_size", batchSize))

	return responses, nil
}

// ProcessStream procesa requests en streaming paralelo
func (p *ParallelProcessor) ProcessStream(ctx context.Context, requests <-chan *GenerateRequest, responses chan<- *GenerateResponse) error {
	g, ctx := errgroup.WithContext(ctx)

	// Worker para procesar requests
	for i := 0; i < 10; i++ { // 10 workers concurrentes para el stream
		g.Go(func() error {
			for {
				select {
				case req, ok := <-requests:
					if !ok {
						return nil // Canal cerrado
					}

					// Crear tarea
					respChan := make(chan *GenerateResponse, 1)
					errChan := make(chan error, 1)

					task := &AITask{
						id:        fmt.Sprintf("stream_%d", time.Now().UnixNano()),
						request:   req,
						response:  respChan,
						error:     errChan,
						processor: p,
						startTime: time.Now(),
						priority:  1,
					}

					// Enviar al pool
					if err := p.workerPool.Submit(task); err != nil {
						p.logger.Error("Failed to submit stream task", zap.Error(err))
						continue
					}

					// Esperar resultado
					select {
					case resp := <-respChan:
						select {
						case responses <- resp:
						case <-ctx.Done():
							return ctx.Err()
						}
					case err := <-errChan:
						p.logger.Error("Stream task failed", zap.Error(err))
					case <-ctx.Done():
						return ctx.Err()
					}

				case <-ctx.Done():
					return ctx.Err()
				}
			}
		})
	}

	return g.Wait()
}

// ProcessConcurrent procesa múltiples requests concurrentemente con límite
func (p *ParallelProcessor) ProcessConcurrent(ctx context.Context, requests []*GenerateRequest, maxConcurrent int) ([]*GenerateResponse, error) {
	if maxConcurrent <= 0 {
		maxConcurrent = int(p.maxConcurrent)
	}

	batchSize := len(requests)
	responses := make([]*GenerateResponse, batchSize)
	errors := make([]error, batchSize)

	// Semáforo para limitar concurrencia
	sem := make(chan struct{}, maxConcurrent)

	g, ctx := errgroup.WithContext(ctx)

	for i, req := range requests {
		idx := i
		request := req

		g.Go(func() error {
			// Adquirir semáforo
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				return ctx.Err()
			}

			// Procesar request
			resp, err := p.manager.Generate(ctx, request)
			if err != nil {
				errors[idx] = err
				return nil // No fallar todo el batch por un error
			}

			responses[idx] = resp
			return nil
		})
	}

	// Esperar a que terminen todos
	if err := g.Wait(); err != nil {
		return responses, err
	}

	// Contar errores
	var failedCount int
	for _, err := range errors {
		if err != nil {
			failedCount++
		}
	}

	if failedCount > 0 {
		return responses, fmt.Errorf("%d/%d requests failed", failedCount, batchSize)
	}

	return responses, nil
}

// RegisterProgressCallback registra un callback de progreso
func (p *ParallelProcessor) RegisterProgressCallback(callback func(completed, total int, details map[string]interface{})) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.progressCallbacks = append(p.progressCallbacks, callback)
}

// notifyProgress notifica el progreso a todos los callbacks
func (p *ParallelProcessor) notifyProgress(completed, total int) {
	p.mu.RLock()
	callbacks := p.progressCallbacks
	p.mu.RUnlock()

	details := map[string]interface{}{
		"processed":  atomic.LoadUint64(&p.totalProcessed),
		"failed":     atomic.LoadUint64(&p.totalFailed),
		"current":    atomic.LoadInt32(&p.current),
		"pool_stats": p.workerPool.GetStats(),
	}

	for _, callback := range callbacks {
		go callback(completed, total, details)
	}
}

// GetStats obtiene estadísticas del procesador
func (p *ParallelProcessor) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"total_processed":    atomic.LoadUint64(&p.totalProcessed),
		"total_failed":       atomic.LoadUint64(&p.totalFailed),
		"current_concurrent": atomic.LoadInt32(&p.current),
		"max_concurrent":     p.maxConcurrent,
		"pool_stats":         p.workerPool.GetStats(),
	}
}

// SetMaxConcurrent establece el límite máximo de concurrencia
func (p *ParallelProcessor) SetMaxConcurrent(max int32) {
	atomic.StoreInt32(&p.maxConcurrent, max)

	// Ajustar el pool si es necesario
	if max > 256 {
		p.workerPool.Resize(32, max/4)
	}
}

// Shutdown cierra el procesador paralelo
func (p *ParallelProcessor) Shutdown(timeout time.Duration) error {
	p.logger.Info("Shutting down parallel processor")
	return p.workerPool.Shutdown(timeout)
}

// ProcessMillionRequests procesa millones de requests optimizadamente
func (p *ParallelProcessor) ProcessMillionRequests(ctx context.Context, requestGenerator func(int) *GenerateRequest, count int) error {
	p.logger.Info("Starting million request processing",
		zap.Int("total_requests", count))

	// Ajustar pool para carga masiva
	p.workerPool.Resize(64, 512)

	// Procesar en chunks para no saturar memoria
	chunkSize := 10000
	completed := 0

	startTime := time.Now()

	for completed < count {
		remaining := count - completed
		if remaining > chunkSize {
			remaining = chunkSize
		}

		// Generar chunk de requests
		requests := make([]*GenerateRequest, remaining)
		for i := 0; i < remaining; i++ {
			requests[i] = requestGenerator(completed + i)
		}

		// Procesar chunk
		_, err := p.ProcessBatch(ctx, requests)
		if err != nil {
			p.logger.Error("Chunk processing failed",
				zap.Error(err),
				zap.Int("chunk_start", completed),
				zap.Int("chunk_size", remaining))
		}

		completed += remaining

		// Log de progreso
		if completed%100000 == 0 {
			elapsed := time.Since(startTime)
			rate := float64(completed) / elapsed.Seconds()
			p.logger.Info("Processing progress",
				zap.Int("completed", completed),
				zap.Int("total", count),
				zap.Float64("rate_per_sec", rate),
				zap.Duration("elapsed", elapsed))
		}
	}

	totalElapsed := time.Since(startTime)
	finalRate := float64(count) / totalElapsed.Seconds()

	p.logger.Info("Million request processing completed",
		zap.Int("total_processed", count),
		zap.Duration("total_time", totalElapsed),
		zap.Float64("avg_rate_per_sec", finalRate))

	return nil
}

// Helper functions para leer configuración del entorno
func getEnvAsInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intVal, err := strconv.Atoi(value); err == nil {
			return intVal
		}
	}
	return defaultValue
}

func getEnvAsFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if floatVal, err := strconv.ParseFloat(value, 64); err == nil {
			return floatVal
		}
	}
	return defaultValue
}

func getEnvAsDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}
