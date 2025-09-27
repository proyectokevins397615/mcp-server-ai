package session

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"kevins-tools/mcp-server-ai/internal/cache"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

// Message representa un mensaje en el historial
type Message struct {
	Role      string    `json:"role"` // "user", "assistant", "system"
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
	Model     string    `json:"model,omitempty"`
	Tokens    int       `json:"tokens,omitempty"`
}

// Session representa una sesión de chat
type Session struct {
	ID           string                 `json:"id"`
	UserID       string                 `json:"user_id"`
	Messages     []Message              `json:"messages"`
	LastContext  map[string]interface{} `json:"last_context,omitempty"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
	MessageCount int                    `json:"message_count"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// Manager gestiona las sesiones de chat
type Manager struct {
	db            *sql.DB
	redis         *cache.RedisClient
	logger        *zap.Logger
	enabled       bool
	useRedis      bool
	useDB         bool
	maxMessages   int
	cacheTTL      time.Duration
	contextWindow int
}

// Config configuración del SessionManager
type Config struct {
	DB            *sql.DB
	Redis         *cache.RedisClient
	Logger        *zap.Logger
	Enabled       bool
	UseRedis      bool
	UseDB         bool
	MaxMessages   int
	CacheTTL      time.Duration
	ContextWindow int
}

// NewManager crea un nuevo SessionManager
func NewManager(config Config) *Manager {
	if config.MaxMessages == 0 {
		config.MaxMessages = 50
	}
	if config.CacheTTL == 0 {
		config.CacheTTL = 24 * time.Hour
	}
	if config.ContextWindow == 0 {
		config.ContextWindow = 10
	}

	manager := &Manager{
		db:            config.DB,
		redis:         config.Redis,
		logger:        config.Logger,
		enabled:       config.Enabled,
		useRedis:      config.UseRedis && config.Redis != nil,
		useDB:         config.UseDB && config.DB != nil,
		maxMessages:   config.MaxMessages,
		cacheTTL:      config.CacheTTL,
		contextWindow: config.ContextWindow,
	}

	if manager.enabled {
		manager.logger.Info("SessionManager initialized",
			zap.Bool("enabled", manager.enabled),
			zap.Bool("redis", manager.useRedis),
			zap.Bool("database", manager.useDB),
			zap.Int("max_messages", manager.maxMessages),
			zap.Duration("cache_ttl", manager.cacheTTL))
	} else {
		manager.logger.Info("SessionManager disabled - running in stateless mode")
	}

	return manager
}

// IsEnabled retorna si el SessionManager está habilitado
func (m *Manager) IsEnabled() bool {
	return m.enabled
}

// GetSession obtiene una sesión por ID
func (m *Manager) GetSession(ctx context.Context, userID, sessionID string) (*Session, error) {
	if !m.enabled {
		return nil, nil
	}

	// Validar IDs
	if userID == "" || sessionID == "" {
		return nil, fmt.Errorf("user_id and session_id are required when session management is enabled")
	}

	// Intentar obtener de Redis primero
	if m.useRedis {
		session, err := m.getFromRedis(ctx, sessionID)
		if err == nil && session != nil {
			// Validar que el usuario coincida
			if session.UserID != userID {
				return nil, fmt.Errorf("session does not belong to user")
			}
			return session, nil
		}
	}

	// Si no está en Redis, buscar en DB
	if m.useDB {
		session, err := m.getFromDB(ctx, userID, sessionID)
		if err != nil {
			return nil, err
		}

		// Guardar en Redis para próximas consultas
		if session != nil && m.useRedis {
			_ = m.saveToRedis(ctx, session)
		}

		return session, nil
	}

	// Si no hay persistencia, crear sesión temporal
	return &Session{
		ID:        sessionID,
		UserID:    userID,
		Messages:  []Message{},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}, nil
}

// AddMessage agrega un mensaje al historial
func (m *Manager) AddMessage(ctx context.Context, userID, sessionID string, message Message) error {
	if !m.enabled {
		return nil
	}

	// Obtener sesión actual
	session, err := m.GetSession(ctx, userID, sessionID)
	if err != nil {
		return err
	}

	if session == nil {
		// Crear nueva sesión si no existe
		session = &Session{
			ID:        sessionID,
			UserID:    userID,
			Messages:  []Message{},
			CreatedAt: time.Now(),
		}
	}

	// Agregar mensaje
	message.Timestamp = time.Now()
	session.Messages = append(session.Messages, message)
	session.MessageCount = len(session.Messages)
	session.UpdatedAt = time.Now()

	// Limitar cantidad de mensajes
	if len(session.Messages) > m.maxMessages {
		// Mantener solo los últimos N mensajes
		session.Messages = session.Messages[len(session.Messages)-m.maxMessages:]
	}

	// Guardar en ambos lugares
	if m.useRedis {
		if err := m.saveToRedis(ctx, session); err != nil {
			m.logger.Error("Failed to save session to Redis", zap.Error(err))
		}
	}

	if m.useDB {
		if err := m.saveToDB(ctx, session); err != nil {
			m.logger.Error("Failed to save session to DB", zap.Error(err))
			return err
		}
	}

	return nil
}

// GetContext obtiene el contexto para el modelo (últimos N mensajes)
func (m *Manager) GetContext(ctx context.Context, userID, sessionID string) ([]Message, error) {
	if !m.enabled {
		return nil, nil
	}

	session, err := m.GetSession(ctx, userID, sessionID)
	if err != nil {
		return nil, err
	}

	if session == nil || len(session.Messages) == 0 {
		return []Message{}, nil
	}

	// Retornar últimos N mensajes para el contexto
	start := 0
	if len(session.Messages) > m.contextWindow {
		start = len(session.Messages) - m.contextWindow
	}

	return session.Messages[start:], nil
}

// getFromRedis obtiene sesión de Redis
func (m *Manager) getFromRedis(ctx context.Context, sessionID string) (*Session, error) {
	key := fmt.Sprintf("session:%s", sessionID)

	data, err := m.redis.Get(ctx, key)
	if err != nil || data == "" {
		return nil, err
	}

	var session Session
	if err := json.Unmarshal([]byte(data), &session); err != nil {
		return nil, err
	}

	return &session, nil
}

// saveToRedis guarda sesión en Redis
func (m *Manager) saveToRedis(ctx context.Context, session *Session) error {
	key := fmt.Sprintf("session:%s", session.ID)

	data, err := json.Marshal(session)
	if err != nil {
		return err
	}

	return m.redis.Set(ctx, key, string(data), m.cacheTTL)
}

// getFromDB obtiene sesión de PostgreSQL
func (m *Manager) getFromDB(ctx context.Context, userID, sessionID string) (*Session, error) {
	query := `
		SELECT id, "userId", messages, "lastContext", "createdAt"
		FROM "Chat"
		WHERE id = $1 AND "userId" = $2
	`

	var messagesJSON []byte
	var lastContextJSON []byte
	var createdAt time.Time

	err := m.db.QueryRowContext(ctx, query, sessionID, userID).Scan(
		&sessionID,
		&userID,
		&messagesJSON,
		&lastContextJSON,
		&createdAt,
	)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	session := &Session{
		ID:        sessionID,
		UserID:    userID,
		CreatedAt: createdAt,
		UpdatedAt: time.Now(),
	}

	// Parsear mensajes
	if len(messagesJSON) > 0 {
		if err := json.Unmarshal(messagesJSON, &session.Messages); err != nil {
			m.logger.Error("Failed to unmarshal messages", zap.Error(err))
			session.Messages = []Message{}
		}
	}

	// Parsear contexto
	if len(lastContextJSON) > 0 {
		if err := json.Unmarshal(lastContextJSON, &session.LastContext); err != nil {
			m.logger.Error("Failed to unmarshal context", zap.Error(err))
		}
	}

	session.MessageCount = len(session.Messages)

	return session, nil
}

// saveToDB guarda sesión en PostgreSQL
func (m *Manager) saveToDB(ctx context.Context, session *Session) error {
	messagesJSON, err := json.Marshal(session.Messages)
	if err != nil {
		return err
	}

	var lastContextJSON []byte
	if session.LastContext != nil {
		lastContextJSON, err = json.Marshal(session.LastContext)
		if err != nil {
			return err
		}
	}

	// Intentar actualizar primero
	updateQuery := `
		UPDATE "Chat"
		SET messages = $1, "lastContext" = $2
		WHERE id = $3 AND "userId" = $4
	`

	result, err := m.db.ExecContext(ctx, updateQuery, messagesJSON, lastContextJSON, session.ID, session.UserID)
	if err != nil {
		return err
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected > 0 {
		return nil
	}

	// Si no existe, insertar
	insertQuery := `
		INSERT INTO "Chat" (id, "userId", messages, "lastContext", "createdAt")
		VALUES ($1, $2, $3, $4, $5)
	`

	// Generar UUID si no existe
	if session.ID == "" {
		session.ID = uuid.New().String()
	}

	_, err = m.db.ExecContext(ctx, insertQuery,
		session.ID,
		session.UserID,
		messagesJSON,
		lastContextJSON,
		session.CreatedAt,
	)

	return err
}

// ClearSession limpia una sesión
func (m *Manager) ClearSession(ctx context.Context, userID, sessionID string) error {
	if !m.enabled {
		return nil
	}

	// Limpiar de Redis
	if m.useRedis {
		key := fmt.Sprintf("session:%s", sessionID)
		_ = m.redis.Delete(ctx, key)
	}

	// Limpiar de DB
	if m.useDB {
		query := `DELETE FROM "Chat" WHERE id = $1 AND "userId" = $2`
		_, err := m.db.ExecContext(ctx, query, sessionID, userID)
		return err
	}

	return nil
}

// BuildPromptWithContext construye el prompt con el contexto del historial
func (m *Manager) BuildPromptWithContext(ctx context.Context, userID, sessionID, newPrompt string) (string, error) {
	if !m.enabled {
		return newPrompt, nil
	}

	messages, err := m.GetContext(ctx, userID, sessionID)
	if err != nil {
		return newPrompt, err
	}

	if len(messages) == 0 {
		return newPrompt, nil
	}

	// Construir contexto como string
	var contextStr string
	for _, msg := range messages {
		contextStr += fmt.Sprintf("[%s]: %s\n", msg.Role, msg.Content)
	}

	// Agregar nuevo prompt
	fullPrompt := fmt.Sprintf("Conversación anterior:\n%s\n[user]: %s", contextStr, newPrompt)

	return fullPrompt, nil
}
