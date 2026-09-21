export type AgentStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'partial'
  | 'insufficient_data'

export type AgentId = 'ai' | 'data' | 'ml' | 'research' | 'orchestrator'

export interface AgentResult {
  agent: AgentId
  status: AgentStatus
  message: string
  data: Record<string, unknown>
  metadata: Record<string, unknown>
}

export interface RunAccepted {
  task_id: string
  agent: AgentId
  agent_label: string
  status: AgentStatus
  routing_reason: string
  poll_url: string
}

export interface HealthInfo {
  ok: boolean
  version: string
  config: {
    base_url?: string
    model?: string
    api_key_configured?: boolean
  }
}

export interface DataReport {
  shape: { rows: number; columns: number }
  columns: { name: string; dtype: string; missing: number; missing_pct: number; unique: number }[]
  total_missing_cells: number
  duplicate_rows: number
  numeric_summary: Record<
    string,
    { mean: number | null; median: number | null; min: number | null; max: number | null; std: number | null }
  >
  correlations: { a: string; b: string; r: number | null }[]
  insights: string[]
}

export interface MLModelResult {
  model: string
  error?: string
  r2?: number | null
  mae?: number | null
  rmse?: number | null
  accuracy?: number | null
  precision?: number | null
  recall?: number | null
  f1_weighted?: number | null
  confusion_matrix?: { labels: string[]; matrix: number[][] } | null
}

export interface MLReport {
  mode: 'regression' | 'classification'
  target: string
  dataset: {
    filename: string
    rows: number
    train_rows: number
    test_rows: number
    features: string[]
    numeric_features: string[]
    categorical_features: string[]
    class_balance: Record<string, number> | null
  }
  models: MLModelResult[]
  best_model: string
  confusion_matrix: { labels: string[]; matrix: number[][] } | null
}

export interface ResearchSource {
  title: string
  url: string
  publisher: string
  published: string | null
  snippet: string
  kind: 'search_result' | 'article'
  evidence_status: 'unverified' | 'verified' | 'rejected'
  evidence_note: string
  extracted_chars: number
  error: string | null
  assessment?: {
    relevance?: string
    relevance_hits?: number
    recency?: string
    age_days?: number | null
    depth?: string
    note?: string
  } | null
}

export interface RunSummary {
  task_id: string
  task: string
  agent: string
  agent_label: string
  status: AgentStatus
  routing_reason: string
  created_at: number
  duration_seconds: number | null
}

export interface ResearchReport {
  request: string
  queries: string[]
  synthesis: string | null
  verified_evidence_count: number
  sources: ResearchSource[]
  evidence_disclaimer: string
}
