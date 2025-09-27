package main

import (
	"bufio"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

type BenchmarkEvent struct {
	Type      string      `json:"type"`
	Timestamp string      `json:"timestamp"`
	Data      interface{} `json:"data"`
}

type BenchmarkResult struct {
	Name        string  `json:"name"`
	Iterations  int64   `json:"iterations"`
	NsPerOp     float64 `json:"nsPerOp"`
	OpsPerSec   float64 `json:"opsPerSec"`
	BytesPerOp  int64   `json:"bytesPerOp"`
	AllocsPerOp int64   `json:"allocsPerOp"`
	Tasks       int64   `json:"tasks"`
	TasksPerSec int64   `json:"tasksPerSec"`
}

type SystemInfo struct {
	CPU   string `json:"cpu"`
	Cores int    `json:"cores"`
	RAM   string `json:"ram"`
	GoVer string `json:"goVersion"`
}

type Server struct {
	clients    map[*websocket.Conn]bool
	broadcast  chan BenchmarkEvent
	register   chan *websocket.Conn
	unregister chan *websocket.Conn
	mu         sync.RWMutex
	results    []BenchmarkResult
	systemInfo SystemInfo
	projectDir string
}

func NewServer() *Server {
	// Detectar el directorio del proyecto
	scriptDir, _ := filepath.Abs(filepath.Dir(os.Args[0]))
	projectDir := filepath.Dir(scriptDir)

	log.Printf("📁 Directorio del proyecto: %s", projectDir)

	return &Server{
		clients:    make(map[*websocket.Conn]bool),
		broadcast:  make(chan BenchmarkEvent, 100),
		register:   make(chan *websocket.Conn),
		unregister: make(chan *websocket.Conn),
		results:    []BenchmarkResult{},
		projectDir: projectDir,
	}
}

func (s *Server) run() {
	for {
		select {
		case client := <-s.register:
			s.mu.Lock()
			s.clients[client] = true
			s.mu.Unlock()
			log.Println("✅ Cliente conectado")

			// Enviar resultados existentes al nuevo cliente
			s.sendExistingResults(client)

		case client := <-s.unregister:
			s.mu.Lock()
			if _, ok := s.clients[client]; ok {
				delete(s.clients, client)
				client.Close()
			}
			s.mu.Unlock()
			log.Println("❌ Cliente desconectado")

		case event := <-s.broadcast:
			s.mu.RLock()
			clientsCopy := make([]*websocket.Conn, 0, len(s.clients))
			for client := range s.clients {
				clientsCopy = append(clientsCopy, client)
			}
			s.mu.RUnlock()

			for _, client := range clientsCopy {
				err := client.WriteJSON(event)
				if err != nil {
					log.Printf("⚠️ Error enviando a cliente: %v", err)
					s.mu.Lock()
					delete(s.clients, client)
					s.mu.Unlock()
					client.Close()
				}
			}
		}
	}
}

func (s *Server) sendExistingResults(client *websocket.Conn) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Enviar info del sistema
	if s.systemInfo.CPU != "" {
		client.WriteJSON(BenchmarkEvent{
			Type:      "system_info",
			Timestamp: time.Now().Format(time.RFC3339),
			Data:      s.systemInfo,
		})
	}

	// Enviar resultados existentes
	for _, result := range s.results {
		client.WriteJSON(BenchmarkEvent{
			Type:      "benchmark_result",
			Timestamp: time.Now().Format(time.RFC3339),
			Data:      result,
		})
	}
}

func (s *Server) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("❌ Error en upgrade: %v", err)
		return
	}

	s.register <- conn

	defer func() {
		s.unregister <- conn
	}()

	// Mantener la conexión abierta y escuchar mensajes
	for {
		var msg map[string]interface{}
		err := conn.ReadJSON(&msg)
		if err != nil {
			break
		}

		// Manejar mensajes del cliente
		if msgType, ok := msg["type"].(string); ok {
			switch msgType {
			case "start_benchmarks":
				log.Println("🎯 Iniciando benchmarks desde interfaz web...")
				go s.runBenchmarks()
			}
		}
	}
}

func (s *Server) runBenchmarks() {
	log.Println("🚀 Iniciando suite de benchmarks...")

	// Obtener información del sistema
	s.getSystemInfo()

	// Verificar que el directorio de tests existe
	testDir := filepath.Join(s.projectDir, "internal", "pool")
	if _, err := os.Stat(testDir); os.IsNotExist(err) {
		log.Printf("❌ Error: No se encuentra el directorio de tests: %s", testDir)
		s.broadcast <- BenchmarkEvent{
			Type:      "error",
			Timestamp: time.Now().Format(time.RFC3339),
			Data: map[string]string{
				"message": fmt.Sprintf("No se encuentra el directorio de tests: %s", testDir),
			},
		}
		return
	}

	benchmarks := []struct {
		name    string
		pattern string
		timeout string
	}{
		{"Submit Individual", "BenchmarkWorkerPool_Submit", "120s"},
		{"Submit Batch", "BenchmarkWorkerPool_SubmitBatch", "60s"},
		{"Throughput", "BenchmarkWorkerPool_Throughput", "60s"},
		{"Auto-Scaling", "BenchmarkWorkerPool_AutoScaling", "30s"},
		{"Buffer Pool", "BenchmarkWorkerPool_BufferPool", "30s"},
		{"Concurrent Load", "BenchmarkWorkerPool_ConcurrentLoad", "60s"},
		{"Progress Tracking", "BenchmarkWorkerPool_ProgressTracking", "30s"},
		{"Dynamic Resize", "BenchmarkWorkerPool_DynamicResize", "30s"},
	}

	s.broadcast <- BenchmarkEvent{
		Type:      "start",
		Timestamp: time.Now().Format(time.RFC3339),
		Data:      map[string]interface{}{"total": len(benchmarks)},
	}

	for i, bench := range benchmarks {
		log.Printf("📊 Ejecutando benchmark %d/%d: %s", i+1, len(benchmarks), bench.name)

		s.broadcast <- BenchmarkEvent{
			Type:      "benchmark_start",
			Timestamp: time.Now().Format(time.RFC3339),
			Data: map[string]interface{}{
				"name":    bench.name,
				"index":   i + 1,
				"total":   len(benchmarks),
				"pattern": bench.pattern,
			},
		}

		s.runSingleBenchmark(bench.name, bench.pattern, bench.timeout)

		s.broadcast <- BenchmarkEvent{
			Type:      "benchmark_end",
			Timestamp: time.Now().Format(time.RFC3339),
			Data: map[string]interface{}{
				"name":  bench.name,
				"index": i + 1,
				"total": len(benchmarks),
			},
		}

		// Pequeña pausa entre benchmarks
		time.Sleep(1 * time.Second)
	}

	s.broadcast <- BenchmarkEvent{
		Type:      "complete",
		Timestamp: time.Now().Format(time.RFC3339),
		Data: map[string]interface{}{
			"totalResults": len(s.results),
		},
	}

	log.Printf("✅ Suite de benchmarks completada. Total de resultados: %d", len(s.results))
}

func (s *Server) getSystemInfo() {
	log.Println("📋 Obteniendo información del sistema...")

	// CPU info
	cpuCmd := exec.Command("sh", "-c", "lscpu | grep 'Model name' | cut -d':' -f2 | xargs")
	cpuOut, _ := cpuCmd.Output()
	s.systemInfo.CPU = strings.TrimSpace(string(cpuOut))

	// Cores
	coresCmd := exec.Command("nproc")
	coresOut, _ := coresCmd.Output()
	fmt.Sscanf(string(coresOut), "%d", &s.systemInfo.Cores)

	// RAM
	ramCmd := exec.Command("sh", "-c", "free -h | grep Mem | awk '{print $2}'")
	ramOut, _ := ramCmd.Output()
	s.systemInfo.RAM = strings.TrimSpace(string(ramOut))

	// Go version
	goCmd := exec.Command("go", "version")
	goOut, _ := goCmd.Output()
	s.systemInfo.GoVer = strings.TrimSpace(string(goOut))

	log.Printf("💻 Sistema: CPU=%s, Cores=%d, RAM=%s", s.systemInfo.CPU, s.systemInfo.Cores, s.systemInfo.RAM)

	s.broadcast <- BenchmarkEvent{
		Type:      "system_info",
		Timestamp: time.Now().Format(time.RFC3339),
		Data:      s.systemInfo,
	}
}

func (s *Server) runSingleBenchmark(benchName, pattern, timeout string) {
	// Construir comando para ejecutar benchmark
	cmd := exec.Command("go", "test",
		"-bench="+pattern,
		"-benchtime=5s",
		"-benchmem",
		"-timeout="+timeout,
		"-run=^$",
		"./internal/pool")

	// Establecer el directorio de trabajo
	cmd.Dir = s.projectDir

	log.Printf("🔧 Ejecutando comando: %s", strings.Join(cmd.Args, " "))
	log.Printf("📂 En directorio: %s", cmd.Dir)

	// Enviar evento de inicio
	s.broadcast <- BenchmarkEvent{
		Type:      "benchmark_running",
		Timestamp: time.Now().Format(time.RFC3339),
		Data: map[string]interface{}{
			"name":    benchName,
			"pattern": pattern,
			"timeout": timeout,
			"command": strings.Join(cmd.Args, " "),
		},
	}

	// Capturar salida
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Printf("❌ Error creando pipe para %s: %v", benchName, err)
		s.broadcast <- BenchmarkEvent{
			Type:      "error",
			Timestamp: time.Now().Format(time.RFC3339),
			Data: map[string]string{
				"benchmark": benchName,
				"error":     err.Error(),
			},
		}
		return
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Printf("❌ Error creando stderr pipe: %v", err)
	}

	if err := cmd.Start(); err != nil {
		log.Printf("❌ Error iniciando benchmark %s: %v", benchName, err)
		s.broadcast <- BenchmarkEvent{
			Type:      "error",
			Timestamp: time.Now().Format(time.RFC3339),
			Data: map[string]string{
				"benchmark": benchName,
				"error":     err.Error(),
			},
		}
		return
	}

	// Leer stderr en goroutine separada
	go func() {
		if stderr != nil {
			scanner := bufio.NewScanner(stderr)
			for scanner.Scan() {
				line := scanner.Text()
				log.Printf("⚠️ STDERR: %s", line)
				s.broadcast <- BenchmarkEvent{
					Type:      "log",
					Timestamp: time.Now().Format(time.RFC3339),
					Data:      map[string]string{"line": "[ERROR] " + line},
				}
			}
		}
	}()

	// Procesar salida línea por línea
	scanner := bufio.NewScanner(stdout)
	benchRegex := regexp.MustCompile(`^Benchmark\S+\s+(\d+)\s+([\d.]+)\s+ns/op`)
	extendedRegex := regexp.MustCompile(`(\d+)\s+tasks\s+(\d+)\s+tasks/sec`)
	memRegex := regexp.MustCompile(`([\d.]+)\s+B/op\s+(\d+)\s+allocs/op`)

	var currentResult *BenchmarkResult

	for scanner.Scan() {
		line := scanner.Text()

		// Filtrar líneas de log de zap
		if !strings.Contains(line, `"level"`) {
			// Enviar línea en tiempo real (solo si no es log de zap)
			s.broadcast <- BenchmarkEvent{
				Type:      "log",
				Timestamp: time.Now().Format(time.RFC3339),
				Data:      map[string]string{"line": line},
			}
		}

		// Parsear resultados de benchmark
		if matches := benchRegex.FindStringSubmatch(line); matches != nil {
			currentResult = &BenchmarkResult{
				Name: strings.Fields(line)[0],
			}

			fmt.Sscanf(matches[1], "%d", &currentResult.Iterations)
			fmt.Sscanf(matches[2], "%f", &currentResult.NsPerOp)
			currentResult.OpsPerSec = 1000000000.0 / currentResult.NsPerOp

			// Buscar información adicional en la misma línea
			if extMatches := extendedRegex.FindStringSubmatch(line); extMatches != nil {
				fmt.Sscanf(extMatches[1], "%d", &currentResult.Tasks)
				fmt.Sscanf(extMatches[2], "%d", &currentResult.TasksPerSec)
			}

			if memMatches := memRegex.FindStringSubmatch(line); memMatches != nil {
				fmt.Sscanf(memMatches[1], "%d", &currentResult.BytesPerOp)
				fmt.Sscanf(memMatches[2], "%d", &currentResult.AllocsPerOp)
			}

			// Enviar el resultado
			s.mu.Lock()
			s.results = append(s.results, *currentResult)
			s.mu.Unlock()

			log.Printf("✅ Resultado: %s - %.2f ns/op, %.0f ops/sec",
				currentResult.Name, currentResult.NsPerOp, currentResult.OpsPerSec)

			s.broadcast <- BenchmarkEvent{
				Type:      "benchmark_result",
				Timestamp: time.Now().Format(time.RFC3339),
				Data:      currentResult,
			}

			currentResult = nil
		}
	}

	if err := cmd.Wait(); err != nil {
		log.Printf("⚠️ Benchmark %s terminó con error: %v", benchName, err)
	} else {
		log.Printf("✅ Benchmark %s completado", benchName)
	}
}

func (s *Server) handleHome(w http.ResponseWriter, r *http.Request) {
	tmpl := template.Must(template.New("index").Parse(htmlTemplate))
	tmpl.Execute(w, nil)
}

const htmlTemplate = `<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Server AI - Benchmarks en Tiempo Real</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        
        h1 {
            color: #333;
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        
        .status-connected {
            background: #d4edda;
            color: #155724;
        }
        
        .status-disconnected {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status-running {
            background: #fff3cd;
            color: #856404;
        }
        
        .pulse {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: currentColor;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        
        .system-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .info-item {
            display: flex;
            flex-direction: column;
        }
        
        .info-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .info-value {
            font-size: 18px;
            font-weight: 600;
            color: #333;
        }
        
        .progress-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
        
        .current-benchmark {
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            font-weight: 500;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .result-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result-header {
            font-size: 16px;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        
        .result-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .metric {
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .metric-value {
            font-size: 20px;
            font-weight: 700;
            color: #667eea;
        }
        
        .metric-label {
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
            margin-top: 2px;
        }
        
        .log-container {
            background: #1e1e1e;
            color: #d4d4d4;
            border-radius: 15px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            line-height: 1.5;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }
        
        .log-line {
            margin: 2px 0;
            padding: 2px 5px;
            border-radius: 3px;
            white-space: pre-wrap;
            word-break: break-all;
        }
        
        .log-line:hover {
            background: rgba(255,255,255,0.05);
        }
        
        .log-benchmark {
            color: #4ec9b0;
            font-weight: 600;
        }
        
        .log-info {
            color: #569cd6;
        }
        
        .log-error {
            color: #f48771;
        }
        
        .stats-summary {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-top: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .stat-card {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
        }
        
        .stat-value {
            font-size: 36px;
            font-weight: 700;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        
        .control-buttons {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: #667eea;
            color: white;
        }
        
        .btn-primary:hover {
            background: #5a67d8;
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #5a6268;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                🚀 MCP Server AI - Benchmarks en Tiempo Real
                <span id="status" class="status-badge status-disconnected">
                    <span class="pulse"></span>
                    Desconectado
                </span>
            </h1>
            
            <div id="systemInfo" class="system-info" style="display: none;">
                <div class="info-item">
                    <span class="info-label">CPU</span>
                    <span class="info-value" id="cpuInfo">-</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Cores</span>
                    <span class="info-value" id="coresInfo">-</span>
                </div>
                <div class="info-item">
                    <span class="info-label">RAM</span>
                    <span class="info-value" id="ramInfo">-</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Go Version</span>
                    <span class="info-value" id="goInfo">-</span>
                </div>
            </div>
            
            <div class="control-buttons">
                <button class="btn btn-primary" onclick="startBenchmarks()">▶️ Iniciar Benchmarks</button>
                <button class="btn btn-secondary" onclick="clearResults()">🗑️ Limpiar Resultados</button>
            </div>
        </div>
        
        <div id="progressContainer" class="progress-container" style="display: none;">
            <h2>📊 Progreso de Benchmarks</h2>
            <div class="progress-bar">
                <div id="progressFill" class="progress-fill" style="width: 0%">
                    0%
                </div>
            </div>
            <div id="currentBenchmark" class="current-benchmark">
                Esperando inicio...
            </div>
        </div>
        
        <div id="resultsGrid" class="results-grid"></div>
        
        <div id="statsSummary" class="stats-summary" style="display: none;">
            <h2>📈 Resumen de Rendimiento</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="avgOps">0</div>
                    <div class="stat-label">Ops/seg Promedio</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="maxOps">0</div>
                    <div class="stat-label">Ops/seg Máximo</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="totalTasks">0</div>
                    <div class="stat-label">Tareas Totales</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="avgLatency">0</div>
                    <div class="stat-label">Latencia Promedio (ns)</div>
                </div>
            </div>
        </div>
        
        <div class="log-container">
            <div id="logContent"></div>
        </div>
    </div>
    
    <script>
        let ws;
        let results = [];
        let currentBenchmark = null;
        let totalBenchmarks = 0;
        let completedBenchmarks = 0;
        const wsPort = window.location.port || '8095';
        
        function connect() {
            const wsUrl = 'ws://localhost:' + wsPort + '/ws';
            console.log('Conectando a:', wsUrl);
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('✅ Conectado al servidor');
                updateStatus('connected');
                addLog('✅ Conectado al servidor de benchmarks', 'log-info');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleEvent(data);
            };
            
            ws.onclose = () => {
                console.log('❌ Desconectado del servidor');
                updateStatus('disconnected');
                addLog('❌ Desconectado del servidor', 'log-error');
                setTimeout(connect, 3000);
            };
            
            ws.onerror = (error) => {
                console.error('Error WebSocket:', error);
                addLog('❌ Error de conexión WebSocket', 'log-error');
            };
        }
        
        function updateStatus(status) {
            const statusEl = document.getElementById('status');
            statusEl.className = 'status-badge';
            
            switch(status) {
                case 'connected':
                    statusEl.classList.add('status-connected');
                    statusEl.innerHTML = '<span class="pulse"></span> Conectado';
                    break;
                case 'running':
                    statusEl.classList.add('status-running');
                    statusEl.innerHTML = '<span class="pulse"></span> Ejecutando';
                    break;
                case 'disconnected':
                    statusEl.classList.add('status-disconnected');
                    statusEl.innerHTML = '<span class="pulse"></span> Desconectado';
                    break;
            }
        }
        
        function handleEvent(event) {
            console.log('Evento recibido:', event.type, event.data);
            
            switch(event.type) {
                case 'system_info':
                    displaySystemInfo(event.data);
                    break;
                case 'start':
                    startBenchmarksUI(event.data);
                    break;
                case 'benchmark_start':
                    updateCurrentBenchmark(event.data);
                    break;
                case 'benchmark_running':
                    addLog('🔧 ' + event.data.name + ': ' + event.data.command, 'log-info');
                    break;
                case 'benchmark_result':
                    addResult(event.data);
                    break;
                case 'benchmark_end':
                    completeBenchmark(event.data);
                    break;
                case 'log':
                    addLog(event.data.line);
                    break;
                case 'error':
                    addLog('❌ Error: ' + event.data.message || event.data.error, 'log-error');
                    break;
                case 'complete':
                    completeBenchmarks();
                    break;
            }
        }
        
        function displaySystemInfo(info) {
            document.getElementById('systemInfo').style.display = 'grid';
            document.getElementById('cpuInfo').textContent = info.cpu || 'N/A';
            document.getElementById('coresInfo').textContent = info.cores || 'N/A';
            document.getElementById('ramInfo').textContent = info.ram || 'N/A';
            document.getElementById('goInfo').textContent = info.goVersion || 'N/A';
        }
        
        function startBenchmarksUI(data) {
            totalBenchmarks = data.total;
            completedBenchmarks = 0;
            document.getElementById('progressContainer').style.display = 'block';
            updateStatus('running');
            updateProgress();
        }
        
        function updateCurrentBenchmark(data) {
            currentBenchmark = data.name;
            document.getElementById('currentBenchmark').innerHTML =
                '<strong>Ejecutando:</strong> ' + data.name +
                ' (' + data.index + '/' + data.total + ')';
        }
        
        function completeBenchmark(data) {
            completedBenchmarks++;
            updateProgress();
        }
        
        function updateProgress() {
            const percent = totalBenchmarks > 0
                ? Math.round((completedBenchmarks / totalBenchmarks) * 100)
                : 0;
            const fill = document.getElementById('progressFill');
            fill.style.width = percent + '%';
            fill.textContent = percent + '%';
        }
        
        function addResult(result) {
            results.push(result);
            
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML =
                '<div class="result-header">' + result.name + '</div>' +
                '<div class="result-metrics">' +
                    '<div class="metric">' +
                        '<div class="metric-value">' + formatNumber(result.opsPerSec) + '</div>' +
                        '<div class="metric-label">Ops/seg</div>' +
                    '</div>' +
                    '<div class="metric">' +
                        '<div class="metric-value">' + formatNumber(result.tasksPerSec) + '</div>' +
                        '<div class="metric-label">Tareas/seg</div>' +
                    '</div>' +
                    '<div class="metric">' +
                        '<div class="metric-value">' + formatNumber(result.nsPerOp) + '</div>' +
                        '<div class="metric-label">ns/op</div>' +
                    '</div>' +
                    '<div class="metric">' +
                        '<div class="metric-value">' + formatNumber(result.tasks) + '</div>' +
                        '<div class="metric-label">Tareas Totales</div>' +
                    '</div>' +
                    '<div class="metric">' +
                        '<div class="metric-value">' + (result.bytesPerOp || 0) + '</div>' +
                        '<div class="metric-label">B/op</div>' +
                    '</div>' +
                    '<div class="metric">' +
                        '<div class="metric-value">' + (result.allocsPerOp || 0) + '</div>' +
                        '<div class="metric-label">Allocs/op</div>' +
                    '</div>' +
                '</div>';
            
            document.getElementById('resultsGrid').appendChild(card);
            updateStats();
        }
        
        function updateStats() {
            if (results.length === 0) return;
            
            document.getElementById('statsSummary').style.display = 'block';
            
            const avgOps = results.reduce((sum, r) => sum + r.opsPerSec, 0) / results.length;
            const maxOps = Math.max(...results.map(r => r.opsPerSec));
            const totalTasks = results.reduce((sum, r) => sum + r.tasks, 0);
            const avgLatency = results.reduce((sum, r) => sum + r.nsPerOp, 0) / results.length;
            
            document.getElementById('avgOps').textContent = formatNumber(avgOps);
            document.getElementById('maxOps').textContent = formatNumber(maxOps);
            document.getElementById('totalTasks').textContent = formatNumber(totalTasks);
            document.getElementById('avgLatency').textContent = formatNumber(avgLatency);
        }
        
        function completeBenchmarks() {
            updateStatus('connected');
            document.getElementById('currentBenchmark').innerHTML =
                '<strong>✅ Benchmarks completados!</strong> Total de resultados: ' + results.length;
        }
        
        function addLog(line, className = '') {
            const logContent = document.getElementById('logContent');
            const logLine = document.createElement('div');
            logLine.className = 'log-line';
            
            if (className) {
                logLine.classList.add(className);
            } else if (line.includes('Benchmark')) {
                logLine.classList.add('log-benchmark');
            } else if (line.includes('INFO') || line.includes('info') || line.includes('✅')) {
                logLine.classList.add('log-info');
            } else if (line.includes('ERROR') || line.includes('❌')) {
                logLine.classList.add('log-error');
            }
            
            logLine.textContent = line;
            logContent.appendChild(logLine);
            
            // Auto-scroll
            logContent.parentElement.scrollTop = logContent.parentElement.scrollHeight;
            
            // Limitar a 1000 líneas
            while (logContent.children.length > 1000) {
                logContent.removeChild(logContent.firstChild);
            }
        }
        
        function formatNumber(num) {
            if (num >= 1000000) {
                return (num / 1000000).toFixed(2) + 'M';
            } else if (num >= 1000) {
                return (num / 1000).toFixed(2) + 'K';
            }
            return Math.round(num).toLocaleString();
        }
        
        function startBenchmarks() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'start_benchmarks'}));
            }
        }
        
        function clearResults() {
            results = [];
            document.getElementById('resultsGrid').innerHTML = '';
            document.getElementById('statsSummary').style.display = 'none';
            document.getElementById('logContent').innerHTML = '';
            addLog('🗑️ Resultados limpiados', 'log-info');
        }
        
        // Conectar al cargar la página
        connect();
    </script>
</body>
</html>\`

func main() {
	server := NewServer()

	// Iniciar el procesador de eventos
	go server.run()

	// Configurar rutas HTTP
	http.HandleFunc("/", server.handleHome)
	http.HandleFunc("/ws", server.handleWebSocket)

	// No iniciar benchmarks automáticamente, esperar comando desde interfaz web
	log.Println("⏳ Servidor listo. Usa la interfaz web para iniciar benchmarks.")

	// Iniciar servidor HTTP en puerto 8095 para evitar conflictos
	port := ":8095"
	log.Printf("🚀 Servidor de benchmarks en tiempo real iniciado en http://localhost%s", port)
	log.Printf("📊 Los benchmarks comenzarán automáticamente en 2 segundos...")
	log.Printf("📌 Nota: El servidor MCP usa puertos 8090-8091, este servidor usa 8095")
	log.Fatal(http.ListenAndServe(port, nil))
}
