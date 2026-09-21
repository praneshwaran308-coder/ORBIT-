import type { AgentResult } from '../types/orbit'
import type { CurrentRun, FlowStep, NodeId, NodeState } from './graph'

const STATE_LABEL: Record<NodeState, string> = {
  idle: 'Idle',
  queued: 'Queued',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  partial: 'Partial',
  insufficient_data: 'Insufficient data',
}

const NODE_TITLES: Record<NodeId, string> = {
  usertask: 'User Task',
  orchestrator: 'Orchestrator',
  ai: 'AI Agent',
  data: 'Data Agent',
  ml: 'ML Agent',
  research: 'Research Agent',
}

function detailRows(nodeId: NodeId, result: AgentResult | null, run: CurrentRun | null): [string, string][] {
  if (!result) {
    if (nodeId === 'orchestrator' && run?.accepted) {
      return [
        ['Status', 'Routing decided'],
        ['Decision', run.accepted.agent_label],
        ['Reason', run.accepted.routing_reason],
        ['Task ID', `${run.accepted.task_id.slice(0, 8)}…`],
      ]
    }
    return []
  }
  const d = result.data as Record<string, unknown>
  const rows: [string, string][] = [['Status', STATE_LABEL[result.status]]]

  if (nodeId === 'ai') {
    const model = (result.metadata as { model?: string }).model
    if (model) rows.push(['Model', model])
    rows.push(['Task', run?.task ?? ''])
  }
  if (nodeId === 'data' && d.report) {
    const report = d.report as Record<string, unknown>
    const shape = report.shape as { rows: number; columns: number } | undefined
    if (shape) {
      rows.push(['Rows × Cols', `${shape.rows} × ${shape.columns}`])
    }
    rows.push(['Missing values', String(report.total_missing_cells ?? '—')])
    rows.push(['Duplicates', String(report.duplicate_rows ?? '—')])
  }
  if (nodeId === 'ml') {
    const dataset = d.dataset as
      | { rows?: number; train_rows?: number; test_rows?: number; features?: string[] }
      | undefined
    const models = (d.models as unknown[]) || []
    if (d.target) rows.push(['Target', String(d.target)])
    if (dataset?.features) rows.push(['Features', dataset.features.join(', ')])
    if (dataset?.rows != null) rows.push(['Rows', String(dataset.rows)])
    if (dataset?.train_rows != null) rows.push(['Train', String(dataset.train_rows)])
    if (dataset?.test_rows != null) rows.push(['Test', String(dataset.test_rows)])
    rows.push(['Models', String(models.length)])
    if (d.mode) rows.push(['Mode', String(d.mode)])
    if (d.best_model) rows.push(['Best', String(d.best_model)])
  }
  if (nodeId === 'research') {
    const sources = (d.sources as { evidence_status: string }[]) || []
    const verified = sources.filter((s) => s.evidence_status === 'verified').length
    const rejected = sources.filter((s) => s.evidence_status === 'rejected').length
    rows.push(['Sources found', String(sources.length)])
    rows.push(['Verified', String(verified)])
    rows.push(['Rejected', String(rejected)])
    if (d.evidence_scope) rows.push(['Evidence scope', String(d.evidence_scope)])
    const queries = (d.queries as string[]) || []
    if (queries.length) rows.push(['Queries', queries.join(' | ')])
  }
  return rows
}

export default function NodeInspector({
  nodeId,
  state,
  result,
  steps,
  run,
  onClose,
}: {
  nodeId: NodeId | null
  state: NodeState
  result: AgentResult | null
  steps?: FlowStep[]
  run: CurrentRun | null
  onClose: () => void
}) {
  if (!nodeId) return null
  const title = NODE_TITLES[nodeId]
  const rows = detailRows(nodeId, result, run)
  return (
    <aside className="node-inspector" aria-label={`${title} details`}>
      <div className="inspector-head">
        <h3>{title}</h3>
        <button className="ghost" onClick={onClose} aria-label="Close inspector">✕</button>
      </div>
      <div className={`inspector-state state-${state}`}>
        <span className="state-label">{STATE_LABEL[state]}</span>
      </div>

      {nodeId === 'usertask' && (
        <p className="inspector-empty">{run ? run.task : 'No task submitted yet.'}</p>
      )}

      {rows.length > 0 && (
        <table className="inspector-table">
          <tbody>
            {rows.map(([k, v]) => (
              <tr key={k}>
                <th scope="row">{k}</th>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {steps && steps.length > 0 && (
        <>
          <h4>Execution steps</h4>
          <ol className="inspector-steps">
            {steps.map((s) => (
              <li key={s.label} className={`step step-${s.state}`}>
                <span className="step-icon" aria-hidden="true">
                  {s.state === 'completed' ? '✓' : s.state === 'failed' ? '✕' : '○'}
                </span>
                <span className="step-label">{s.label}</span>
                {s.detail && <span className="step-detail">{s.detail}</span>}
              </li>
            ))}
          </ol>
        </>
      )}

      {result && nodeId !== 'usertask' && (
        <p className="inspector-note">{result.message}</p>
      )}
    </aside>
  )
}
