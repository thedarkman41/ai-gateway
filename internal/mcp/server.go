// Package mcp is a small MCP server: JSON-RPC 2.0 over newline-delimited stdio or over HTTP POST,
// serving tools only. It implements what an MCP client needs to list and call tools: initialize,
// ping, tools/list and tools/call.
package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"sync"
)

// Tool is one callable tool. Handler returns the text shown to the model; an error becomes a tool
// result with isError set, so the model can read it and react.
type Tool struct {
	Name        string
	Description string
	InputSchema map[string]any
	Annotations map[string]any
	Handler     func(ctx context.Context, args map[string]any) (string, error)
}

type Server struct {
	Name    string
	Version string
	Tools   []Tool
}

var supported = map[string]bool{"2025-06-18": true, "2025-03-26": true, "2024-11-05": true}

const latest = "2025-06-18"

type request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

// Handle answers one JSON-RPC message. It returns nil when no reply is due (a notification, or a
// response sent by the client).
func (s *Server) Handle(ctx context.Context, raw []byte) []byte {
	var req request
	if err := json.Unmarshal(raw, &req); err != nil {
		return encode(response{ID: json.RawMessage("null"), Error: &rpcError{-32700, "parse error"}})
	}
	if req.Method == "" || len(req.ID) == 0 {
		return nil
	}
	res := response{ID: req.ID}
	result, rerr := s.dispatch(ctx, req)
	if rerr != nil {
		res.Error = rerr
	} else {
		res.Result = result
	}
	return encode(res)
}

func (s *Server) dispatch(ctx context.Context, req request) (any, *rpcError) {
	switch req.Method {
	case "initialize":
		var p struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		json.Unmarshal(req.Params, &p)
		v := latest
		if supported[p.ProtocolVersion] {
			v = p.ProtocolVersion
		}
		return map[string]any{
			"protocolVersion": v,
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      map[string]any{"name": s.Name, "version": s.Version},
		}, nil
	case "ping":
		return map[string]any{}, nil
	case "tools/list":
		tools := make([]map[string]any, 0, len(s.Tools))
		for _, t := range s.Tools {
			m := map[string]any{"name": t.Name, "description": t.Description, "inputSchema": t.InputSchema}
			if t.Annotations != nil {
				m["annotations"] = t.Annotations
			}
			tools = append(tools, m)
		}
		return map[string]any{"tools": tools}, nil
	case "tools/call":
		var p struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return nil, &rpcError{-32602, "invalid params"}
		}
		for _, t := range s.Tools {
			if t.Name != p.Name {
				continue
			}
			text, err := t.Handler(ctx, p.Arguments)
			if err != nil {
				text = err.Error()
			}
			return map[string]any{
				"content": []map[string]any{{"type": "text", "text": text}},
				"isError": err != nil,
			}, nil
		}
		return nil, &rpcError{-32602, "unknown tool: " + p.Name}
	}
	return nil, &rpcError{-32601, "method not found: " + req.Method}
}

func encode(r response) []byte {
	r.JSONRPC = "2.0"
	b, _ := json.Marshal(r)
	return b
}

// ServeStdio reads newline-delimited messages from r and writes replies to w until r ends. Each
// request runs in its own goroutine, so a ping is answered while a slow tool call is in flight.
func (s *Server) ServeStdio(ctx context.Context, r io.Reader, w io.Writer) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wmu sync.Mutex
	var wg sync.WaitGroup
	defer wg.Wait()
	br := bufio.NewReaderSize(r, 1<<16)
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 1 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if out := s.Handle(ctx, line); out != nil {
					wmu.Lock()
					w.Write(append(out, '\n'))
					wmu.Unlock()
				}
			}()
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
	}
}

// HTTPHandler serves the same messages over POST, replying with plain JSON. It is meant for
// 127.0.0.1 only: a request whose Host or Origin is not this machine is refused, which stops a web
// page from reaching the gateway through the browser.
func (s *Server) HTTPHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !local(r.Host) || (r.Header.Get("Origin") != "" && !localOrigin(r.Header.Get("Origin"))) {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", "POST")
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		body, err := io.ReadAll(io.LimitReader(r.Body, 8<<20))
		if err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		out := s.Handle(r.Context(), body)
		if out == nil {
			w.WriteHeader(http.StatusAccepted)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(out)
	})
}

// LocalOnly wraps a handler so that it serves this machine only: a request whose Host or Origin is
// not a loopback address is refused, which stops a web page from reaching it through the browser.
func LocalOnly(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !local(r.Host) || (r.Header.Get("Origin") != "" && !localOrigin(r.Header.Get("Origin"))) {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		h.ServeHTTP(w, r)
	})
}

func local(hostport string) bool {
	h, _, err := net.SplitHostPort(hostport)
	if err != nil {
		h = hostport
	}
	return h == "127.0.0.1" || h == "localhost" || h == "::1"
}

func localOrigin(o string) bool {
	u, err := url.Parse(o)
	return err == nil && local(u.Host)
}
