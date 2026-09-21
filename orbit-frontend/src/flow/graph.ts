import type { AgentResult, RunAccepted, ResearchSource } from '../types/orbit'

export type NodeState =
  | 'idle'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'partial'
  | 'insufficient_data'

export type NodeId = 'usertask' | 'orchestrator' | 'ai' | 'data' | 'ml' | 'research'

export interface FlowStep {
  label: string
  state: NodeState
  detail?: string
}

export interface NodeInfo {
  id: NodeId
  state: NodeState
  title: string
  lines: string[]
  steps?: FlowStep[]
  result?: AgentResult | null
  executed: boolean
}

/** Everything the Flow view needs about the run being visualized. */
export interface CurrentRun {
  task: string
  accepted: RunAccepted | null
  result: AgentResult | null
  /** Registry-level state from GET /status: queued | running | terminal status. */
  taskState: string
}

function trunc(s: string | undefined | null, n: number): string {
  if (!s) return ''
  return s.length > n ? `${s.slice(0, n - 1)}…` : s
}

// -- top-level node states ---------------------------------------------------

export function orchestratorState(run: CurrentRun | null): NodeState {
  if (!run || !run.accepted) return 'idle'
  return 'completed' // routing is decided synchronously at accept time
}

export function userTaskState(run: CurrentRun | null): NodeState {
  if (!run || (!run.accepted && !run.result)) return 'idle'
  if (run.result) return run.result.status
  return run.taskState === 'queued' ? 'queued' : 'running'
}

export function agentNodeState(agent: string, run: CurrentRun | null): NodeState {
  if (!run || !run.accepted) return 'idle'
  if (run.accepted.agent !== agent) return 'idle' // never fake execution
  if (run.result) return run.result.status
  return run.taskState === 'queued' ? 'queued' : 'running'
}

// -- node detail lines (from real payloads only) -----------------------------

function dataReport(res: AgentResult | null): Record<string, unknown> | null {
  const report = (res?.data as { report?: Record<string, unknown> } | undefined)?.report
  return report ?? null
}

function mlReport(res: AgentResult | null): Record<string, unknown> | null {
  const report = (res?.data as { format?: string } | undefined)
  return report && report.format === 'ml_report' ? (res!.data as Record<string, unknown>) : null
}

function researchReport(res: AgentResult | null): Record<string, unknown> | null {
  const data = res?.data as { format?: string } | undefined
  return data && data.format === 'research_report' ? (res!.data as Record<string, unknown>) : null
}

function bestMetricLine(report: Record<string, unknown>): string | undefined {
  const models = (report.models as { model: string; r2?: number | null; f1_weighted?: number | null }[]) || []
  const bestName = report.best_model as string | undefined
  const best = models.find((m) => m.model === bestName) || null
  if (!best) return undefined
  if (typeof best.r2 === 'number') return `best: ${best.model} (R² ${best.r2.toFixed(3)})`
  if (typeof best.f1_weighted === 'number') return `best: ${best.model} (F1 ${best.f1_weighted.toFixed(3)})`
  return bestName ? `best: ${bestName}` : undefined
}

export function nodeLines(id: NodeId, run: CurrentRun | null): string[] {
  const res = run?.result && run.result.agent === id ? run.result : null
  switch (id) {
    case 'usertask': {
      return [trunc(run?.task, 64) || '(empty task)']
    }
    case 'orchestrator': {
      if (!run?.accepted) return ['Deterministic routing, no LLM']
      return [
        trunc(run.accepted.routing_reason, 48),
        `task ${run.accepted.task_id.slice(0, 8)}…`,
      ]
    }
    case 'ai': {
      if (!res) return ['General AI']
      const model = (res.metadata as { model?: string }).model
      const lines = [trunc(run?.task, 40)]
      if (model) lines.push(`model: ${model}`)
      return lines
    }
    case 'data': {
      const report = dataReport(res)
      if (!report) return ['Local CSV profiling']
      const shape = report.shape as { rows: number; columns: number } | undefined
      const lines: string[] = []
      if (shape) lines.push(`${shape.rows} rows × ${shape.columns} cols`)
      if (typeof report.total_missing_cells === 'number') lines.push(`missing: ${report.total_missing_cells}`)
      if (typeof report.duplicate_rows === 'number') lines.push(`duplicates: ${report.duplicate_rows}`)
      return lines.length ? lines : ['Local CSV profiling']
    }
    case 'ml': {
      const report = mlReport(res)
      if (!report) return ['Train & evaluate models']
      const dataset = report.dataset as { features?: string[] } | undefined
      const models = (report.models as unknown[]) || []
      const lines = [`target: ${report.target}`]
      if (dataset?.features) lines.push(`features: ${dataset.features.length}`)
      lines.push(`models: ${models.length}`)
      const best = bestMetricLine(report)
      if (best) lines.push(trunc(best, 34))
      return lines
    }
    case 'research': {
      const report = researchReport(res)
      if (!report) return ['Grounded web research']
      const sources = (report.sources as ResearchSource[]) || []
      const verified = sources.filter((s) => s.evidence_status === 'verified').length
      const rejected = sources.filter((s) => s.evidence_status === 'rejected').length
      const lines = [
        `sources: ${sources.length}`,
        `verified: ${verified} · rejected: ${rejected}`,
      ]
      const scope = report.evidence_scope as string | undefined
      if (scope) lines.push(trunc(scope, 52))
      return lines
    }
  }
}

// -- internal flows (only steps that actually exist in the implementation) ----

export function dataSteps(res: AgentResult | null): FlowStep[] {
  const report = dataReport(res)
  const ok = res?.status === 'completed' && !!report
  const failDetail = res && res.status !== 'completed' ? trunc(res.message, 90) : undefined
  const shape = report?.shape as { rows: number; columns: number } | undefined
  const corr = (report?.correlations as unknown[]) || []
  const insights = (report?.insights as unknown[]) || []
  const narrative = (res?.data as { narrative?: string } | undefined)?.narrative
  return [
    { label: 'CSV Input', state: report ? 'completed' : res ? 'failed' : 'idle',
      detail: shape ? `${shape.rows} × ${shape.columns}` : failDetail },
    { label: 'Validation', state: report ? 'completed' : res ? 'failed' : 'idle', detail: failDetail },
    { label: 'Statistics', state: ok ? 'completed' : 'idle',
      detail: ok && report ? `${Object.keys(report.numeric_summary as object ?? {}).length} numeric columns` : undefined },
    { label: 'Correlation Analysis', state: ok ? 'completed' : 'idle',
      detail: ok ? `${corr.length} pair(s)` : undefined },
    { label: 'Insights', state: ok ? 'completed' : 'idle', detail: ok ? `${insights.length} insight(s)` : undefined },
    { label: 'Narrative', state: narrative ? 'completed' : ok ? 'completed' : 'idle',
      detail: narrative ? 'LLM narration generated' : undefined },
  ]
}

export function mlSteps(res: AgentResult | null): FlowStep[] {
  const report = mlReport(res)
  const ok = res?.status === 'completed' && !!report
  const insufficient = res?.status === 'insufficient_data'
  const failDetail = res && (insufficient || res.status === 'failed') ? trunc(res.message, 90) : undefined
  const dataset = report?.dataset as
    | { train_rows?: number; test_rows?: number; features?: string[] }
    | undefined
  const models = (report?.models as unknown[]) || []
  return [
    { label: 'CSV Input', state: res ? 'completed' : 'idle' },
    { label: 'Target Validation', state: ok ? 'completed' : res ? 'failed' : 'idle', detail: failDetail },
    { label: 'Feature Selection', state: ok ? 'completed' : 'idle',
      detail: dataset?.features ? dataset.features.join(', ') : undefined },
    { label: 'Train/Test Split', state: ok ? 'completed' : 'idle',
      detail: dataset ? `${dataset.train_rows} train / ${dataset.test_rows} test` : undefined },
    { label: 'Model Training', state: ok ? 'completed' : 'idle', detail: ok ? `${models.length} model(s)` : undefined },
    { label: 'Model Evaluation', state: ok ? 'completed' : 'idle',
      detail: ok ? bestMetricLine(report!) ?? 'metrics computed' : undefined },
    { label: 'Comparison', state: ok ? 'completed' : 'idle',
      detail: ok ? (report!.best_model as string) : undefined },
  ]
}

export function researchSteps(res: AgentResult | null): FlowStep[] {
  const report = researchReport(res)
  const sources = (report?.sources as ResearchSource[]) || []
  const queries = (report?.queries as string[]) || []
  const synthesis = (report?.synthesis as string | null | undefined) ?? null
  const has = (report ?? null) !== null
  const fetched = sources.filter((s) => s.extracted_chars > 0 || s.kind === 'article').length
  const verified = sources.filter((s) => s.evidence_status === 'verified').length
  const rejected = sources.filter((s) => s.evidence_status === 'rejected').length
  const assessed = sources.some((s) => s.assessment)
  const offTopic = sources.filter((s) => s.assessment?.relevance === 'off_topic').length
  return [
    { label: 'Research Task', state: has ? 'completed' : res ? 'failed' : 'idle',
      detail: report ? trunc(report.request as string, 56) : undefined },
    { label: 'Query Generation', state: queries.length ? 'completed' : res ? 'failed' : 'idle',
      detail: queries.length ? `${queries.length} query(ies)` : undefined },
    { label: 'Source Discovery', state: sources.length ? 'completed' : res ? 'failed' : 'idle',
      detail: sources.length ? `${sources.length} candidate(s)` : undefined },
    { label: 'Relevance Filtering', state: sources.length ? 'completed' : res ? 'failed' : 'idle',
      detail: offTopic ? `${offTopic} off-topic flagged` : sources.length ? 'all candidates on-topic' : undefined },
    { label: 'Evidence Retrieval', state: fetched ? 'completed' : sources.length ? 'failed' : res ? 'failed' : 'idle',
      detail: sources.length ? `${fetched} fetched / ${sources.length} attempted` : undefined },
    { label: 'Evidence Validation', state: sources.length ? 'completed' : res ? 'failed' : 'idle',
      detail: sources.length ? `${verified} verified · ${rejected} rejected` : undefined },
    { label: 'Quality / Recency Assessment', state: assessed ? 'completed' : sources.length ? 'idle' : 'idle',
      detail: assessed ? 'verified sources assessed' : undefined },
    { label: 'Grounded Synthesis', state: synthesis ? 'completed' : res ? 'failed' : 'idle',
      detail: synthesis ? 'findings grounded in verified evidence' : undefined },
  ]
}

export function stepsForAgent(agent: string, res: AgentResult | null): FlowStep[] | undefined {
  if (agent === 'data') return dataSteps(res)
  if (agent === 'ml') return mlSteps(res)
  if (agent === 'research') return researchSteps(res)
  return undefined
}
