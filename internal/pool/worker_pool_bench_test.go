package pool

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.uber.org/zap"
)

// MockTask implementa la interfaz Task para benchmarks
type MockTask struct {
	id       string
	priority int
	workTime time.Duration
	executed int32
}

func (t *MockTask) Execute(ctx context.Context) error {
	// Simular trabajo
	if t.workTime > 0 {
		time.Sleep(t.workTime)
	}
	atomic.AddInt32(&t.executed, 1)
	return nil
}

func (t *MockTask) GetID() string {
	return t.id
}

func (t *MockTask) GetPriority() int {
	return t.priority
}

// BenchmarkWorkerPool_Submit prueba el rendimiento de submit individual
func BenchmarkWorkerPool_Submit(b *testing.B) {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	configs := []struct {
		name       string
		minWorkers int
		maxWorkers int
		queueSize  int
	}{
		{"Small", 4, 8, 1000},
		{"Medium", 8, 32, 10000},
		{"Large", 32, 128, 100000},
		{"XLarge", 64, 256, 1000000},
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			pool := NewWorkerPool(Config{
				Name:       fmt.Sprintf("bench_%s", cfg.name),
				MinWorkers: cfg.minWorkers,
				MaxWorkers: cfg.maxWorkers,
				QueueSize:  cfg.queueSize,
				Logger:     logger,
			})
			defer pool.Shutdown(5 * time.Second)

			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					task := &MockTask{
						id:       fmt.Sprintf("task_%d", i),
						priority: 1,
						workTime: 0,
					}
					_ = pool.Submit(task)
					i++
				}
			})

			b.ReportMetric(float64(b.N), "tasks")
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tasks/sec")
		})
	}
}

// BenchmarkWorkerPool_SubmitBatch prueba el rendimiento de submit en batch
func BenchmarkWorkerPool_SubmitBatch(b *testing.B) {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	batchSizes := []int{10, 100, 1000, 10000}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("Batch_%d", batchSize), func(b *testing.B) {
			pool := NewWorkerPool(Config{
				Name:       fmt.Sprintf("bench_batch_%d", batchSize),
				MinWorkers: 16,
				MaxWorkers: 64,
				QueueSize:  batchSize * 100,
				Logger:     logger,
			})
			defer pool.Shutdown(5 * time.Second)

			// Preparar batch de tareas
			tasks := make([]Task, batchSize)
			for i := 0; i < batchSize; i++ {
				tasks[i] = &MockTask{
					id:       fmt.Sprintf("task_%d", i),
					priority: 1,
					workTime: 0,
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = pool.SubmitBatch(tasks)
			}

			totalTasks := b.N * batchSize
			b.ReportMetric(float64(totalTasks), "total_tasks")
			b.ReportMetric(float64(totalTasks)/b.Elapsed().Seconds(), "tasks/sec")
		})
	}
}

// BenchmarkWorkerPool_Throughput mide el throughput con diferentes cargas
func BenchmarkWorkerPool_Throughput(b *testing.B) {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	workloads := []struct {
		name     string
		workTime time.Duration
		workers  int
	}{
		{"NoWork", 0, 32},
		{"Light_1us", time.Microsecond, 32},
		{"Medium_100us", 100 * time.Microsecond, 64},
		{"Heavy_1ms", time.Millisecond, 128},
	}

	for _, wl := range workloads {
		b.Run(wl.name, func(b *testing.B) {
			pool := NewWorkerPool(Config{
				Name:       fmt.Sprintf("bench_throughput_%s", wl.name),
				MinWorkers: wl.workers,
				MaxWorkers: wl.workers,
				QueueSize:  100000,
				Logger:     logger,
			})
			defer pool.Shutdown(10 * time.Second)

			var completed int64

			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					task := &MockTask{
						id:       fmt.Sprintf("task_%d", atomic.AddInt64(&completed, 1)),
						priority: 1,
						workTime: wl.workTime,
					}
					_ = pool.Submit(task)
				}
			})

			pool.Wait()

			b.ReportMetric(float64(b.N), "tasks_completed")
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tasks/sec")

			stats := pool.GetStats()
			b.ReportMetric(float64(stats["tasks_processed"].(uint64)), "processed")
			b.ReportMetric(float64(stats["tasks_failed"].(uint64)), "failed")
		})
	}
}

// BenchmarkWorkerPool_AutoScaling prueba el auto-scaling
func BenchmarkWorkerPool_AutoScaling(b *testing.B) {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	b.Run("AutoScale", func(b *testing.B) {
		pool := NewWorkerPool(Config{
			Name:              "bench_autoscale",
			MinWorkers:        4,
			MaxWorkers:        128,
			QueueSize:         100000,
			ScaleInterval:     100 * time.Millisecond, // Escalar más rápido para el test
			UtilizationTarget: 0.7,
			Logger:            logger,
		})
		defer pool.Shutdown(10 * time.Second)

		// Generar carga variable
		var wg sync.WaitGroup
		b.ResetTimer()

		// Fase 1: Carga baja
		for i := 0; i < 1000; i++ {
			task := &MockTask{
				id:       fmt.Sprintf("low_%d", i),
				workTime: time.Millisecond,
			}
			pool.Submit(task)
		}
		time.Sleep(500 * time.Millisecond)

		// Fase 2: Carga alta
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50000; i++ {
				task := &MockTask{
					id:       fmt.Sprintf("high_%d", i),
					workTime: 100 * time.Microsecond,
				}
				pool.Submit(task)
			}
		}()

		// Fase 3: Pico de carga
		wg.Add(1)
		go func() {
			defer wg.Done()
			time.Sleep(1 * time.Second)
			for i := 0; i < 100000; i++ {
				task := &MockTask{
					id:       fmt.Sprintf("peak_%d", i),
					workTime: 10 * time.Microsecond,
				}
				pool.Submit(task)
			}
		}()

		wg.Wait()
		pool.Wait()

		stats := pool.GetStats()
		b.ReportMetric(float64(stats["tasks_processed"].(uint64)), "total_processed")
		b.ReportMetric(float64(stats["current_workers"].(int32)), "final_workers")
		b.ReportMetric(float64(stats["max_workers"].(int32)), "max_workers")
	})
}

// BenchmarkWorkerPool_BufferPool prueba la eficiencia del sync.Pool
func BenchmarkWorkerPool_BufferPool(b *testing.B) {
	b.Run("WithPool", func(b *testing.B) {
		pool := &sync.Pool{
			New: func() interface{} {
				return make([]byte, 64*1024)
			},
		}

		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				buf := pool.Get().([]byte)
				// Simular uso del buffer
				_ = buf[0]
				pool.Put(buf)
			}
		})
	})

	b.Run("WithoutPool", func(b *testing.B) {
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				buf := make([]byte, 64*1024)
				// Simular uso del buffer
				_ = buf[0]
			}
		})
	})

	// Reportar diferencia de memoria
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	b.ReportMetric(float64(m.Alloc)/1024/1024, "MB_allocated")
}

// BenchmarkWorkerPool_Concurrent prueba concurrencia extrema
func BenchmarkWorkerPool_Concurrent(b *testing.B) {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	concurrencyLevels := []int{100, 1000, 10000}

	for _, level := range concurrencyLevels {
		b.Run(fmt.Sprintf("Concurrent_%d", level), func(b *testing.B) {
			pool := NewWorkerPool(Config{
				Name:       fmt.Sprintf("bench_concurrent_%d", level),
				MinWorkers: 32,
				MaxWorkers: 256,
				QueueSize:  level * 10,
				Logger:     logger,
			})
			defer pool.Shutdown(10 * time.Second)

			b.ResetTimer()

			var wg sync.WaitGroup
			wg.Add(level)

			start := time.Now()
			for i := 0; i < level; i++ {
				go func(id int) {
					defer wg.Done()
					for j := 0; j < b.N/level; j++ {
						task := &MockTask{
							id:       fmt.Sprintf("task_%d_%d", id, j),
							workTime: 10 * time.Microsecond,
						}
						for {
							if err := pool.Submit(task); err == nil {
								break
							}
							time.Sleep(time.Microsecond)
						}
					}
				}(i)
			}

			wg.Wait()
			pool.Wait()
			elapsed := time.Since(start)

			b.ReportMetric(float64(b.N), "total_tasks")
			b.ReportMetric(float64(b.N)/elapsed.Seconds(), "tasks/sec")
			b.ReportMetric(float64(level), "concurrent_producers")
		})
	}
}

// BenchmarkWorkerPool_ProgressTracking prueba el overhead del progress tracking
func BenchmarkWorkerPool_ProgressTracking(b *testing.B) {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	b.Run("WithProgress", func(b *testing.B) {
		pool := NewWorkerPool(Config{
			Name:       "bench_progress",
			MinWorkers: 32,
			MaxWorkers: 32,
			QueueSize:  10000,
			Logger:     logger,
		})
		defer pool.Shutdown(5 * time.Second)

		var progressUpdates int64
		pool.RegisterProgressCallback(func(completed, total int) {
			atomic.AddInt64(&progressUpdates, 1)
		})

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			task := &MockTask{
				id:       fmt.Sprintf("task_%d", i),
				workTime: 0,
			}
			pool.Submit(task)
		}
		pool.Wait()

		b.ReportMetric(float64(progressUpdates), "progress_updates")
	})

	b.Run("WithoutProgress", func(b *testing.B) {
		pool := NewWorkerPool(Config{
			Name:       "bench_no_progress",
			MinWorkers: 32,
			MaxWorkers: 32,
			QueueSize:  10000,
			Logger:     logger,
		})
		defer pool.Shutdown(5 * time.Second)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			task := &MockTask{
				id:       fmt.Sprintf("task_%d", i),
				workTime: 0,
			}
			pool.Submit(task)
		}
		pool.Wait()
	})
}

// BenchmarkWorkerPool_Resize prueba el rendimiento del redimensionamiento dinámico
func BenchmarkWorkerPool_Resize(b *testing.B) {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	pool := NewWorkerPool(Config{
		Name:       "bench_resize",
		MinWorkers: 4,
		MaxWorkers: 256,
		QueueSize:  100000,
		Logger:     logger,
	})
	defer pool.Shutdown(10 * time.Second)

	b.ResetTimer()

	// Simular redimensionamiento durante carga
	go func() {
		sizes := []int32{8, 16, 32, 64, 128, 64, 32, 16, 8}
		for _, size := range sizes {
			time.Sleep(time.Duration(b.N/len(sizes)) * time.Microsecond)
			pool.Resize(size/2, size)
		}
	}()

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			task := &MockTask{
				id:       fmt.Sprintf("task_%d", i),
				workTime: 10 * time.Microsecond,
			}
			pool.Submit(task)
			i++
		}
	})

	stats := pool.GetStats()
	b.ReportMetric(float64(stats["tasks_processed"].(uint64)), "processed_during_resize")
}

// BenchmarkWorkerPool_MillionTasks prueba con millones de tareas
func BenchmarkWorkerPool_MillionTasks(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping million tasks benchmark in short mode")
	}

	logger, _ := zap.NewProduction()
	defer logger.Sync()

	taskCounts := []int{100000, 500000, 1000000}

	for _, count := range taskCounts {
		b.Run(fmt.Sprintf("Tasks_%d", count), func(b *testing.B) {
			pool := NewWorkerPool(Config{
				Name:       fmt.Sprintf("bench_million_%d", count),
				MinWorkers: 64,
				MaxWorkers: 256,
				QueueSize:  100000,
				Logger:     logger,
			})
			defer pool.Shutdown(30 * time.Second)

			tasks := make([]Task, count)
			for i := 0; i < count; i++ {
				tasks[i] = &MockTask{
					id:       fmt.Sprintf("task_%d", i),
					workTime: time.Microsecond,
				}
			}

			b.ResetTimer()
			start := time.Now()

			// Submit en batches para mejor rendimiento
			batchSize := 1000
			for i := 0; i < count; i += batchSize {
				end := i + batchSize
				if end > count {
					end = count
				}
				pool.SubmitBatch(tasks[i:end])
			}

			pool.Wait()
			elapsed := time.Since(start)

			b.ReportMetric(float64(count), "total_tasks")
			b.ReportMetric(float64(count)/elapsed.Seconds(), "tasks/sec")
			b.ReportMetric(elapsed.Seconds(), "total_seconds")

			stats := pool.GetStats()
			b.ReportMetric(float64(stats["tasks_processed"].(uint64)), "verified_processed")
		})
	}
}
