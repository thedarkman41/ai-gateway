package gateway

import "context"

// Problem is an error a caller can act on: an HTTP status, a short code, and one sentence.
type Problem struct {
	Status     int
	Code       string
	Message    string
	RetryAfter int // seconds, for a quota problem
}

func (p *Problem) Error() string { return p.Message }

type scopeKey struct{}

// WithLocalOnly marks a request as local-only: whatever its content, only models on the user's own
// machines may see any part of it, helper calls included.
func WithLocalOnly(ctx context.Context) context.Context {
	return context.WithValue(ctx, scopeKey{}, true)
}

func localOnly(ctx context.Context) bool { v, _ := ctx.Value(scopeKey{}).(bool); return v }
