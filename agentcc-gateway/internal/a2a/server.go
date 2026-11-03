package a2a

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/futureagi/agentcc-gateway/internal/mcp"
	"github.com/futureagi/agentcc-gateway/internal/models"
	"github.com/futureagi/agentcc-gateway/internal/pipeline"
	"github.com/futureagi/agentcc-gateway/internal/providers"
)

// Server handles incoming A2A requests (other agents sending tasks to Agentcc).
type Server struct {
	cardGen          *CardGenerator
	registry         *Registry
	tasks            *syncTaskStore
	taskCancels      sync.Map
	chatExecutor     ChatCompletionExecutor
	engine           *pipeline.Engine
	providerRegistry *providers.Registry
	done             chan struct{}
}

type ChatCompletionExecutor func(ctx context.Context, authHeader string, req *models.ChatCompletionRequest) (*models.ChatCompletionResponse, error)

// syncTaskStore is a thread-safe task store.
type syncTaskStore struct {
	mu    sync.RWMutex
	store *TaskStore
}

func newSyncTaskStore() *syncTaskStore {
	return &syncTaskStore{store: NewTaskStore()}
}

func (s *syncTaskStore) Store(task *Task) {
	s.mu.Lock()
	s.store.Store(task)
	s.mu.Unlock()
}

func (s *syncTaskStore) Get(id string) (*Task, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.store.Get(id)
}

// Update applies fn to the stored task under the write lock, so task-field
// mutations from the background execution goroutine and HTTP handlers
// cannot race. Returns false when the task is gone (cleaned up).
func (s *syncTaskStore) Update(id string, fn func(t *Task)) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.store.Get(id)
	if !ok {
		return false
	}
	fn(t)
	return true
}

// Snapshot returns a shallow copy of the stored task under the read lock.
// Safe because every writer replaces Status.Message / Artifacts slices
// wholesale and History is set once before the first Store.
func (s *syncTaskStore) Snapshot(id string) (Task, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	t, ok := s.store.Get(id)
	if !ok {
		return Task{}, false
	}
	return *t, true
}

func (s *syncTaskStore) Cleanup(maxAge time.Duration) {
	s.mu.Lock()
	s.store.Cleanup(maxAge)
	s.mu.Unlock()
}

// NewServer creates an A2A server.
// If engine and providerRegistry are provided, message/send routes through the
// full pipeline (pre-plugins → provider call → post-plugins). Otherwise it
// falls back to a simple echo for testing.
func NewServer(cardCfg CardConfig, registry *Registry, opts ...ServerOption) *Server {
	s := &Server{
		cardGen:  NewCardGenerator(cardCfg),
		registry: registry,
		tasks:    newSyncTaskStore(),
		done:     make(chan struct{}),
	}
	for _, opt := range opts {
		opt(s)
	}

	// Background cleanup every 10 minutes.
	go func() {
		ticker := time.NewTicker(10 * time.Minute)
		defer ticker.Stop()
		for {
			select {
			case <-s.done:
				return
			case <-ticker.C:
				s.tasks.Cleanup(1 * time.Hour)
			}
		}
	}()

	return s
}

// HandleAgentCard serves GET /.well-known/agent.json.
func (s *Server) HandleAgentCard(w http.ResponseWriter, r *http.Request) {
	s.cardGen.HandleAgentCard(w, r)
}

// HandleMessage handles POST /a2a — the main A2A JSON-RPC endpoint.
func (s *Server) HandleMessage(w http.ResponseWriter, r *http.Request) {
	var msg mcp.Message
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		writeA2AError(w, nil, mcp.ErrCodeParse, "invalid JSON: "+err.Error())
		return
	}

	if msg.JSONRPC != "2.0" {
		writeA2AError(w, msg.ID, mcp.ErrCodeInvalidRequest, "expected jsonrpc 2.0")
		return
	}

	switch msg.Method {
	case MethodMessageSend, MethodTasksSend:
		s.handleMessageSend(w, r, &msg)
	case MethodTasksGet:
		s.handleTasksGet(w, &msg)
	case MethodTasksCancel:
		s.handleTasksCancel(w, &msg)
	default:
		writeA2AError(w, msg.ID, mcp.ErrCodeMethodNotFound, "unknown method: "+msg.Method)
	}
}

// ListAgents serves GET /v1/agents — returns configured external agents.
func (s *Server) ListAgents(w http.ResponseWriter, r *http.Request) {
	agents := s.registry.List()

	type agentInfo struct {
		Name        string     `json:"name"`
		URL         string     `json:"url"`
		Description string     `json:"description,omitempty"`
		Skills      []Skill    `json:"skills,omitempty"`
		Healthy     bool       `json:"healthy"`
		Card        *AgentCard `json:"card,omitempty"`
	}

	result := make([]agentInfo, 0, len(agents))
	for _, a := range agents {
		result = append(result, agentInfo{
			Name:        a.Name,
			URL:         a.URL,
			Description: a.Description,
			Skills:      a.Skills,
			Healthy:     a.Healthy(),
			Card:        a.Card.Load(),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// ServerOption configures an A2A server.
type ServerOption func(*Server)

// WithPipeline enables pipeline integration for message/send.
func WithPipeline(engine *pipeline.Engine, providerRegistry *providers.Registry) ServerOption {
	return func(s *Server) {
		s.engine = engine
		s.providerRegistry = providerRegistry
	}
}

// WithChatCompletionExecutor routes A2A execution through an existing chat-completion path.
func WithChatCompletionExecutor(executor ChatCompletionExecutor) ServerOption {
	return func(s *Server) {
		s.chatExecutor = executor
	}
}

func (s *Server) handleMessageSend(w http.ResponseWriter, r *http.Request, msg *mcp.Message) {
	var params MessageSendParams
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		writeA2AError(w, msg.ID, mcp.ErrCodeInvalidParams, "invalid message/send params")
		return
	}

	// Create task.
	taskID := generateTaskID()
	task := &Task{
		ID: taskID,
		Status: TaskStatus{
			State: TaskStatusWorking,
		},
		History: []Message{params.Message},
	}

	// Extract text from the A2A message.
	var inputText string
	for _, part := range params.Message.Parts {
		if part.Type == "text" {
			if inputText != "" {
				inputText += "\n"
			}
			inputText += part.Text
		}
	}

	if inputText == "" {
		task.Status.State = TaskStatusFailed
		task.Status.Message = []MessagePart{
			{Type: "text", Text: "No text content in message"},
		}
		s.tasks.Store(task)
		writeA2AResult(w, msg.ID, task)
		return
	}

	// If pipeline is wired, route through the full plugin pipeline.
	if s.engine != nil && s.providerRegistry != nil {
		if params.Configuration != nil && params.Configuration.ReturnImmediately {
			s.tasks.Store(task)
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			s.taskCancels.Store(task.ID, cancel)
			go s.runMessageSendWithPipeline(ctx, r.Header.Get("Authorization"), task.ID, inputText, params)
			// The goroutine owns task mutation now; respond from a copy.
			snap, _ := s.tasks.Snapshot(task.ID)
			writeA2AResult(w, msg.ID, &snap)
			return
		}

		s.handleMessageSendWithPipeline(w, msg, r.Header.Get("Authorization"), task, inputText, params)
		return
	}

	// Fallback: echo back (used in tests without full pipeline).
	task.Status.State = TaskStatusCompleted
	task.Status.Message = []MessagePart{
		{Type: "text", Text: "Task received and processed by Agentcc gateway"},
	}
	task.Artifacts = []Artifact{
		{
			Name:  "response",
			Index: 0,
			Parts: []MessagePart{
				{Type: "text", Text: fmt.Sprintf("Received: %s", inputText)},
			},
		},
	}
	s.tasks.Store(task)
	slog.Info("a2a task completed (echo)", "task_id", task.ID)
	writeA2AResult(w, msg.ID, task)
}

// handleMessageSendWithPipeline converts the A2A message to a ChatCompletionRequest,
// processes it through the full pipeline (pre-plugins → provider → post-plugins),
// and converts the response back to an A2A task.
func (s *Server) handleMessageSendWithPipeline(w http.ResponseWriter, msg *mcp.Message, authHeader string, task *Task, inputText string, params MessageSendParams) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	s.runMessageSendWithPipeline(ctx, authHeader, task.ID, inputText, params)
	// A concurrent tasks/get could read while we executed; respond from a copy.
	snap, _ := s.tasks.Snapshot(task.ID)
	writeA2AResult(w, msg.ID, &snap)
}

func (s *Server) runMessageSendWithPipeline(ctx context.Context, authHeader string, taskID string, inputText string, params MessageSendParams) {
	defer s.taskCancels.Delete(taskID)

	fail := func(state string, text string) {
		s.tasks.Update(taskID, func(t *Task) {
			t.Status.State = state
			t.Status.Message = []MessagePart{{Type: "text", Text: text}}
		})
	}

	model := s.resolveA2AModel(params)
	if model == "" {
		fail(TaskStatusFailed, "A2A request must include metadata.model")
		slog.Warn("a2a: missing metadata.model", "task_id", taskID)
		return
	}

	// Build OpenAI ChatCompletionRequest.
	contentJSON, _ := json.Marshal(inputText)
	req := &models.ChatCompletionRequest{
		Model: model,
		Messages: []models.Message{
			{Role: "user", Content: contentJSON},
		},
	}

	if s.chatExecutor != nil {
		resp, err := s.chatExecutor(ctx, authHeader, req)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
				fail(TaskStatusCanceled, "Task canceled")
				return
			}
			fail(TaskStatusFailed, fmt.Sprintf("Pipeline error: %s", err.Error()))
			return
		}
		s.finishTaskFromResponse(taskID, resp)
		return
	}

	// Acquire RequestContext from pool.
	rc := models.AcquireRequestContext()
	defer rc.Release()

	rc.Model = model
	rc.Request = req
	rc.IsStream = false
	rc.SetMetadata("a2a_task_id", taskID)
	rc.SetMetadata("a2a_context_id", params.ContextID)
	rc.SetMetadata("endpoint_type", "a2a")

	// Resolve provider.
	provider, err := s.providerRegistry.Resolve(model)
	if err != nil {
		fail(TaskStatusFailed, fmt.Sprintf("No provider for model %q: %s", model, err.Error()))
		slog.Warn("a2a: provider resolution failed", "model", model, "error", err)
		return
	}
	rc.Provider = provider.ID()

	// Define provider call function.
	providerCall := func(callCtx context.Context, callRC *models.RequestContext) error {
		resp, err := provider.ChatCompletion(callCtx, callRC.Request)
		if err != nil {
			return err
		}
		callRC.Response = resp
		callRC.ResolvedModel = resp.Model
		return nil
	}

	// Execute pipeline.
	if err := s.engine.Process(ctx, rc, providerCall); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			fail(TaskStatusCanceled, "Task canceled")
			slog.Info("a2a task canceled", "task_id", taskID)
			return
		}

		fail(TaskStatusFailed, fmt.Sprintf("Pipeline error: %s", err.Error()))
		slog.Warn("a2a: pipeline error", "task_id", taskID, "error", err)
		return
	}

	s.finishTaskFromResponse(taskID, rc.Response)
	slog.Info("a2a task completed (pipeline)", "task_id", taskID, "model", model, "provider", rc.Provider)
}

// finishTaskFromResponse applies the provider response to the stored task
// under the store lock.
func (s *Server) finishTaskFromResponse(taskID string, resp *models.ChatCompletionResponse) {
	s.tasks.Update(taskID, func(task *Task) {
		if resp != nil && len(resp.Choices) > 0 {
			var responseText string
			if err := json.Unmarshal(resp.Choices[0].Message.Content, &responseText); err != nil {
				responseText = string(resp.Choices[0].Message.Content)
			}

			task.Status.State = TaskStatusCompleted
			task.Status.Message = []MessagePart{{Type: "text", Text: "Completed via pipeline"}}
			task.Artifacts = []Artifact{{
				Name:  "response",
				Index: 0,
				Parts: []MessagePart{{Type: "text", Text: responseText}},
			}}
			return
		}

		task.Status.State = TaskStatusFailed
		task.Status.Message = []MessagePart{{Type: "text", Text: "No response from provider"}}
	})
}

func (s *Server) handleTasksGet(w http.ResponseWriter, msg *mcp.Message) {
	var params TaskGetParams
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		writeA2AError(w, msg.ID, mcp.ErrCodeInvalidParams, "invalid tasks/get params")
		return
	}

	snap, ok := s.tasks.Snapshot(params.TaskID)
	if !ok && params.TaskIDSnake != "" {
		snap, ok = s.tasks.Snapshot(params.TaskIDSnake)
	}
	if !ok {
		taskID := params.TaskID
		if taskID == "" {
			taskID = params.TaskIDSnake
		}
		writeA2AError(w, msg.ID, mcp.ErrCodeInvalidParams, "task not found: "+taskID)
		return
	}

	writeA2AResult(w, msg.ID, &snap)
}

func (s *Server) handleTasksCancel(w http.ResponseWriter, msg *mcp.Message) {
	var params TaskCancelParams
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		writeA2AError(w, msg.ID, mcp.ErrCodeInvalidParams, "invalid tasks/cancel params")
		return
	}

	snap, ok := s.tasks.Snapshot(params.TaskID)
	if !ok && params.TaskIDSnake != "" {
		snap, ok = s.tasks.Snapshot(params.TaskIDSnake)
	}
	if !ok {
		taskID := params.TaskID
		if taskID == "" {
			taskID = params.TaskIDSnake
		}
		writeA2AError(w, msg.ID, mcp.ErrCodeInvalidParams, "task not found: "+taskID)
		return
	}

	if cancelFn, ok := s.taskCancels.Load(snap.ID); ok {
		cancelFn.(context.CancelFunc)()
	}
	// Only a working task transitions to canceled; a terminal task keeps its
	// state (previously cancel flipped completed/failed tasks to canceled).
	s.tasks.Update(snap.ID, func(t *Task) {
		if t.Status.State == TaskStatusWorking {
			t.Status.State = TaskStatusCanceled
			t.Status.Message = []MessagePart{{Type: "text", Text: "Task canceled"}}
		}
	})
	snap, _ = s.tasks.Snapshot(snap.ID)
	writeA2AResult(w, msg.ID, &snap)
}

func (s *Server) resolveA2AModel(params MessageSendParams) string {
	return extractModelFromMetadata(params.Metadata)
}

func extractModelFromMetadata(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	var meta map[string]any
	if err := json.Unmarshal(raw, &meta); err != nil {
		return ""
	}

	model, _ := meta["model"].(string)
	return model
}

// Close stops the background cleanup goroutine.
func (s *Server) Close() {
	close(s.done)
}

// Registry returns the agent registry.
func (s *Server) Registry() *Registry {
	return s.registry
}

func writeA2AResult(w http.ResponseWriter, id json.RawMessage, result interface{}) {
	resp, err := mcp.NewResponse(id, result)
	if err != nil {
		writeA2AError(w, id, mcp.ErrCodeInternal, "failed to marshal result")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func writeA2AError(w http.ResponseWriter, id json.RawMessage, code int, message string) {
	resp := mcp.NewErrorResponse(id, code, message)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func generateTaskID() string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		panic("crypto/rand.Read failed: " + err.Error())
	}
	return "task-" + hex.EncodeToString(b)
}
