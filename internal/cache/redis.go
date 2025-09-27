package cache

import (
	"context"
	"crypto/tls"
	"fmt"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

// RedisClient cliente de Redis para caché
type RedisClient struct {
	client *redis.Client
}

// NewRedisClient crea un nuevo cliente de Redis con soporte para SSL
func NewRedisClient(redisURL string) (*RedisClient, error) {
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Redis URL: %w", err)
	}

	// Configurar TLS para conexiones seguras (rediss://)
	if strings.HasPrefix(redisURL, "rediss://") {
		opts.TLSConfig = &tls.Config{
			MinVersion: tls.VersionTLS12,
			// Azure Redis Cache requiere TLS
			InsecureSkipVerify: false,
		}

		// Configuración adicional para Azure Redis Cache
		opts.MaxRetries = 3
		opts.MinRetryBackoff = 8 * time.Millisecond
		opts.MaxRetryBackoff = 512 * time.Millisecond
		opts.DialTimeout = 10 * time.Second
		opts.ReadTimeout = 5 * time.Second
		opts.WriteTimeout = 5 * time.Second
		opts.PoolSize = 10
		opts.MinIdleConns = 5
		opts.MaxIdleConns = 10
		opts.ConnMaxIdleTime = 5 * time.Minute
		opts.ConnMaxLifetime = 30 * time.Minute
	}

	client := redis.NewClient(opts)

	// Probar conexión con timeout más largo para Azure
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return &RedisClient{
		client: client,
	}, nil
}

// Get obtiene un valor del caché
func (rc *RedisClient) Get(ctx context.Context, key string) (string, error) {
	val, err := rc.client.Get(ctx, key).Result()
	if err == redis.Nil {
		return "", nil // Key no existe
	}
	if err != nil {
		return "", err
	}
	return val, nil
}

// Set establece un valor en el caché con TTL
func (rc *RedisClient) Set(ctx context.Context, key string, value string, ttl time.Duration) error {
	return rc.client.Set(ctx, key, value, ttl).Err()
}

// Delete elimina una clave del caché
func (rc *RedisClient) Delete(ctx context.Context, key string) error {
	return rc.client.Del(ctx, key).Err()
}

// Exists verifica si una clave existe
func (rc *RedisClient) Exists(ctx context.Context, key string) (bool, error) {
	n, err := rc.client.Exists(ctx, key).Result()
	if err != nil {
		return false, err
	}
	return n > 0, nil
}

// Close cierra la conexión con Redis
func (rc *RedisClient) Close() error {
	return rc.client.Close()
}

// Flush limpia todo el caché (usar con cuidado)
func (rc *RedisClient) Flush(ctx context.Context) error {
	return rc.client.FlushDB(ctx).Err()
}

// SetWithExpiry establece un valor con tiempo de expiración específico
func (rc *RedisClient) SetWithExpiry(ctx context.Context, key string, value string, expiry time.Time) error {
	ttl := time.Until(expiry)
	if ttl <= 0 {
		return fmt.Errorf("expiry time is in the past")
	}
	return rc.Set(ctx, key, value, ttl)
}

// GetTTL obtiene el tiempo de vida restante de una clave
func (rc *RedisClient) GetTTL(ctx context.Context, key string) (time.Duration, error) {
	return rc.client.TTL(ctx, key).Result()
}

// Increment incrementa un contador
func (rc *RedisClient) Increment(ctx context.Context, key string) (int64, error) {
	return rc.client.Incr(ctx, key).Result()
}

// IncrementBy incrementa un contador por un valor específico
func (rc *RedisClient) IncrementBy(ctx context.Context, key string, value int64) (int64, error) {
	return rc.client.IncrBy(ctx, key, value).Result()
}

// SetJSON establece un valor JSON en el caché
func (rc *RedisClient) SetJSON(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	return rc.client.Set(ctx, key, value, ttl).Err()
}

// GetJSON obtiene un valor JSON del caché
func (rc *RedisClient) GetJSON(ctx context.Context, key string, dest interface{}) error {
	return rc.client.Get(ctx, key).Scan(dest)
}
