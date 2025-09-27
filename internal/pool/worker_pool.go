package pool

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"
)

// Métricas de Prometheus para el pool
var (
	poolActiveWorkers = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "worker_pool_active_workers",
		Help: "Number of active workers in the pool",
	}, []string{"pool_name"})

	poolQueueSize = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "worker_pool_queue_size",
		Help: "Current size of the work queue",
	}, []string{"pool_name"})

	poolTasksProcessed = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "worker_pool_tasks_processed_total",
		Help: "Total number of tasks processed",
	}, []string{"pool_name", "status"})

	poolTaskDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "worker_pool_task_duration_seconds",
		Help:    "Duration of task processing",
		Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
	}, []string{"pool_name"})

	poolWorkerUtilization = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "worker_pool_utilization_ratio",
		Help: "Worker utilization ratio (0-1)",
	}, []string{"pool_name"})
)

// Task representa una tarea a ejecutar
type Task interface {
	Execute(ctx context.Context) error
	GetID() string
	GetPriority() int
}

// BufferedTask es una tarea que puede usar un buffer pre-asignado
type BufferedTask interface {
	Task
	ExecuteWithBuffer(ctx context.Context, buffer []byte) error
}

// WorkerPool representa un pool de workers escalable
type WorkerPool struct {
	name           string
	minWorkers     int32
	maxWorkers     int32
	currentWorkers int32
	taskQueue      chan Task
	workerWG       sync.WaitGroup
	ctx            context.Context
	cancel         context.CancelFunc
	logger         *zap.Logger

	// Métricas y estadísticas
	tasksProcessed uint64
	tasksFailed    uint64
	totalDuration  int64

	// Pool de buffers reutilizables
	bufferPool *sync.Pool

	// Control de redimensionamiento dinámico
	scaleTicker       *time.Ticker
	scaleInterval     time.Duration
	utilizationTarget float64

	// Progress tracking
	progressCallbacks []func(completed, total int)
	totalTasks        int32
	completedTasks    int32

	// Control de concurrencia con errgroup
	errGroup *errgroup.Group

	// Mutex para operaciones críticas
	mu sync.RWMutex
}

// Config configuración del pool
type Config struct {
	Name              string
	MinWorkers        int
	MaxWorkers        int
	QueueSize         int
	ScaleInterval     time.Duration
	UtilizationTarget float64
	BufferSize        int
	Logger            *zap.Logger
}

// NewWorkerPool crea un nuevo pool de workers
func NewWorkerPool(config Config) *WorkerPool {
	if config.MinWorkers <= 0 {
		config.MinWorkers = runtime.NumCPU()
	}
	if config.MaxWorkers <= 0 {
		config.MaxWorkers = runtime.NumCPU() * 4
	}
	if config.QueueSize <= 0 {
		config.QueueSize = config.MaxWorkers * 100
	}
	if config.ScaleInterval <= 0 {
		config.ScaleInterval = 10 * time.Second
	}
	if config.UtilizationTarget <= 0 {
		config.UtilizationTarget = 0.8
	}
	if config.Logger == nil {
		config.Logger, _ = zap.NewProduction()
	}

	ctx, cancel := context.WithCancel(context.Background())

	pool := &WorkerPool{
		name:              config.Name,
		minWorkers:        int32(config.MinWorkers),
		maxWorkers:        int32(config.MaxWorkers),
		currentWorkers:    0,
		taskQueue:         make(chan Task, config.QueueSize),
		ctx:               ctx,
		cancel:            cancel,
		logger:            config.Logger,
		scaleInterval:     config.ScaleInterval,
		utilizationTarget: config.UtilizationTarget,
		bufferPool: &sync.Pool{
			New: func() interface{} {
				// Crear buffers reutilizables (por defecto 64KB)
				size := 64 * 1024
				if config.BufferSize > 0 {
					size = config.BufferSize
				}
				return make([]byte, size)
			},
		},
		progressCallbacks: make([]func(int, int), 0),
	}

	// Iniciar workers mínimos
	pool.scale(int32(config.MinWorkers))

	// Iniciar auto-scaling
	pool.startAutoScaling()

	// Iniciar recolector de métricas
	pool.startMetricsCollector()

	pool.logger.Info("Worker pool created",
		zap.String("name", pool.name),
		zap.Int32("min_workers", pool.minWorkers),
		zap.Int32("max_workers", pool.maxWorkers))

	return pool
}

// Submit envía una tarea al pool
func (p *WorkerPool) Submit(task Task) error {
	select {
	case <-p.ctx.Done():
		return fmt.Errorf("pool is shutting down")
	case p.taskQueue <- task:
		atomic.AddInt32(&p.totalTasks, 1)
		poolQueueSize.WithLabelValues(p.name).Set(float64(len(p.taskQueue)))
		return nil
	default:
		return fmt.Errorf("task queue is full")
	}
}

// SubmitBatch envía múltiples tareas al pool
func (p *WorkerPool) SubmitBatch(tasks []Task) error {
	g, ctx := errgroup.WithContext(p.ctx)

	for _, task := range tasks {
		task := task // Captura para el closure
		g.Go(func() error {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case p.taskQueue <- task:
				atomic.AddInt32(&p.totalTasks, 1)
				return nil
			}
		})
	}

	return g.Wait()
}

// worker función del worker
func (p *WorkerPool) worker(id int) {
	defer p.workerWG.Done()

	p.logger.Debug("Worker started", zap.Int("worker_id", id))

	for {
		select {
		case <-p.ctx.Done():
			p.logger.Debug("Worker stopping", zap.Int("worker_id", id))
			return

		case task, ok := <-p.taskQueue:
			if !ok {
				return
			}

			// Obtener buffer del pool
			buffer := p.bufferPool.Get().([]byte)

			// Ejecutar tarea con medición de tiempo
			startTime := time.Now()
			err := p.executeTask(task, buffer)
			duration := time.Since(startTime)

			// Devolver buffer al pool
			p.bufferPool.Put(buffer)

			// Actualizar métricas
			atomic.AddUint64(&p.tasksProcessed, 1)
			atomic.AddInt64(&p.totalDuration, duration.Nanoseconds())

			if err != nil {
				atomic.AddUint64(&p.tasksFailed, 1)
				poolTasksProcessed.WithLabelValues(p.name, "failed").Inc()
				p.logger.Error("Task failed",
					zap.String("task_id", task.GetID()),
					zap.Error(err))
			} else {
				poolTasksProcessed.WithLabelValues(p.name, "success").Inc()

				// Progress tracking
				completed := atomic.AddInt32(&p.completedTasks, 1)
				p.notifyProgress(int(completed), int(atomic.LoadInt32(&p.totalTasks)))
			}

			poolTaskDuration.WithLabelValues(p.name).Observe(duration.Seconds())
			poolQueueSize.WithLabelValues(p.name).Set(float64(len(p.taskQueue)))
		}
	}
}

// executeTask ejecuta una tarea con timeout y recuperación de pánico
func (p *WorkerPool) executeTask(task Task, buffer []byte) (err error) {
	// Recuperación de pánico
	defer func() {
		if r := recover(); r != nil {
			// Usar el buffer para construir el mensaje de error
			errMsg := fmt.Sprintf("task panic: %v", r)
			copy(buffer, []byte(errMsg))
			err = fmt.Errorf("%s", string(buffer[:len(errMsg)]))

			p.logger.Error("Task panicked",
				zap.String("task_id", task.GetID()),
				zap.Any("panic", r),
				zap.Int("buffer_used", len(errMsg)))
		}
	}()

	// Crear contexto con timeout y pasar el buffer como valor
	ctx, cancel := context.WithTimeout(p.ctx, 30*time.Second)
	defer cancel()

	// Adjuntar el buffer al contexto para que las tareas puedan usarlo
	type bufferKey struct{}
	ctx = context.WithValue(ctx, bufferKey{}, buffer)

	// Si la tarea soporta buffer, usarlo directamente
	if bufferedTask, ok := task.(BufferedTask); ok {
		return bufferedTask.ExecuteWithBuffer(ctx, buffer)
	}

	// Ejecutar tarea normal
	return task.Execute(ctx)
}

// scale ajusta el número de workers
func (p *WorkerPool) scale(targetWorkers int32) {
	p.mu.Lock()
	defer p.mu.Unlock()

	current := atomic.LoadInt32(&p.currentWorkers)

	if targetWorkers > p.maxWorkers {
		targetWorkers = p.maxWorkers
	} else if targetWorkers < p.minWorkers {
		targetWorkers = p.minWorkers
	}

	diff := targetWorkers - current

	if diff > 0 {
		// Agregar workers
		for i := int32(0); i < diff; i++ {
			p.workerWG.Add(1)
			workerID := int(current + i)
			go p.worker(workerID)
		}
		atomic.StoreInt32(&p.currentWorkers, targetWorkers)
		p.logger.Info("Scaled up workers",
			zap.Int32("added", diff),
			zap.Int32("total", targetWorkers))
	} else if diff < 0 {
		// Reducir workers (se implementaría con señales)
		p.logger.Info("Scale down requested",
			zap.Int32("current", current),
			zap.Int32("target", targetWorkers))
	}

	poolActiveWorkers.WithLabelValues(p.name).Set(float64(targetWorkers))
}

// startAutoScaling inicia el auto-scaling basado en utilización
func (p *WorkerPool) startAutoScaling() {
	p.scaleTicker = time.NewTicker(p.scaleInterval)

	go func() {
		for {
			select {
			case <-p.ctx.Done():
				p.scaleTicker.Stop()
				return

			case <-p.scaleTicker.C:
				p.autoScale()
			}
		}
	}()
}

// autoScale ajusta automáticamente el número de workers
func (p *WorkerPool) autoScale() {
	queueSize := len(p.taskQueue)
	queueCapacity := cap(p.taskQueue)
	utilization := float64(queueSize) / float64(queueCapacity)

	currentWorkers := atomic.LoadInt32(&p.currentWorkers)

	poolWorkerUtilization.WithLabelValues(p.name).Set(utilization)

	if utilization > p.utilizationTarget && currentWorkers < p.maxWorkers {
		// Scale up
		newWorkers := currentWorkers + int32(float64(currentWorkers)*0.2) + 1
		if newWorkers > p.maxWorkers {
			newWorkers = p.maxWorkers
		}
		p.scale(newWorkers)
	} else if utilization < p.utilizationTarget*0.5 && currentWorkers > p.minWorkers {
		// Scale down
		newWorkers := currentWorkers - int32(float64(currentWorkers)*0.1)
		if newWorkers < p.minWorkers {
			newWorkers = p.minWorkers
		}
		p.scale(newWorkers)
	}
}

// startMetricsCollector inicia el recolector de métricas
func (p *WorkerPool) startMetricsCollector() {
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-p.ctx.Done():
				return
			case <-ticker.C:
				p.collectMetrics()
			}
		}
	}()
}

// collectMetrics recolecta y actualiza métricas
func (p *WorkerPool) collectMetrics() {
	processed := atomic.LoadUint64(&p.tasksProcessed)
	failed := atomic.LoadUint64(&p.tasksFailed)
	totalDuration := atomic.LoadInt64(&p.totalDuration)

	if processed > 0 {
		avgDuration := time.Duration(totalDuration / int64(processed))
		p.logger.Debug("Pool metrics",
			zap.String("pool", p.name),
			zap.Uint64("processed", processed),
			zap.Uint64("failed", failed),
			zap.Duration("avg_duration", avgDuration),
			zap.Int32("workers", atomic.LoadInt32(&p.currentWorkers)),
			zap.Int("queue_size", len(p.taskQueue)))
	}
}

// RegisterProgressCallback registra un callback de progreso
func (p *WorkerPool) RegisterProgressCallback(callback func(completed, total int)) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.progressCallbacks = append(p.progressCallbacks, callback)
}

// notifyProgress notifica el progreso a todos los callbacks
func (p *WorkerPool) notifyProgress(completed, total int) {
	p.mu.RLock()
	callbacks := p.progressCallbacks
	p.mu.RUnlock()

	for _, callback := range callbacks {
		go callback(completed, total)
	}
}

// GetStats obtiene estadísticas del pool
func (p *WorkerPool) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"name":            p.name,
		"current_workers": atomic.LoadInt32(&p.currentWorkers),
		"min_workers":     p.minWorkers,
		"max_workers":     p.maxWorkers,
		"queue_size":      len(p.taskQueue),
		"queue_capacity":  cap(p.taskQueue),
		"tasks_processed": atomic.LoadUint64(&p.tasksProcessed),
		"tasks_failed":    atomic.LoadUint64(&p.tasksFailed),
		"completed_tasks": atomic.LoadInt32(&p.completedTasks),
		"total_tasks":     atomic.LoadInt32(&p.totalTasks),
	}
}

// Resize cambia dinámicamente el tamaño del pool
func (p *WorkerPool) Resize(minWorkers, maxWorkers int32) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if minWorkers > 0 {
		p.minWorkers = minWorkers
	}
	if maxWorkers > 0 && maxWorkers >= p.minWorkers {
		p.maxWorkers = maxWorkers
	}

	// Aplicar cambios inmediatamente
	current := atomic.LoadInt32(&p.currentWorkers)
	if current < p.minWorkers {
		p.scale(p.minWorkers)
	} else if current > p.maxWorkers {
		p.scale(p.maxWorkers)
	}

	p.logger.Info("Pool resized",
		zap.Int32("min_workers", p.minWorkers),
		zap.Int32("max_workers", p.maxWorkers))
}

// Shutdown cierra el pool de manera ordenada
func (p *WorkerPool) Shutdown(timeout time.Duration) error {
	p.logger.Info("Shutting down worker pool", zap.String("name", p.name))

	// Señalar shutdown
	p.cancel()

	// Esperar con timeout
	done := make(chan struct{})
	go func() {
		p.workerWG.Wait()
		close(done)
	}()

	select {
	case <-done:
		p.logger.Info("Worker pool shutdown complete", zap.String("name", p.name))
		return nil
	case <-time.After(timeout):
		return fmt.Errorf("shutdown timeout after %v", timeout)
	}
}

// Wait espera a que todas las tareas se completen
func (p *WorkerPool) Wait() {
	for len(p.taskQueue) > 0 || atomic.LoadInt32(&p.completedTasks) < atomic.LoadInt32(&p.totalTasks) {
		time.Sleep(100 * time.Millisecond)
	}
}
