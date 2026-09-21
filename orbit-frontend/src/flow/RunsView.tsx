import { useCallback, useEffect, useState } from 'react'
import { getRuns, pollStatusFull } from '../api/orbit'
import type { AgentResult, RunSummary } from '../types/orbit'

const STATUS_ICON: Record<string, string> = {
  completed: '✓',
  failed: '✕',
  partial: '◑',
  insufficient_data: '△',
  running: '●',
  queued: '◔',
}

function timeAgo(ts: number): string {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts))
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}

function clockTime(ts: number): string {
  const d = new Date(ts * 1000)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ', ' + d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export default function RunsView({ onOpenRun }: { onOpenRun: (run: OpenedRun) => void }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setRuns(await getRuns())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const open = useCallback(
    async (summary: RunSummary) => {
      try {
        const full = await pollStatusFull(summary.task_id)
        onOpenRun({ summary, result: full.result, taskState: full.status })
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      }
    },
    [onOpenRun],
  )

  return (
    <div className="runs-view">
      <div className="runs-head">
        <h2>Previous executions</h2>
        <div>
          <button className="ghost" onClick={refresh} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>
      <p className="runs-note">
        Session scope: ORBIT keeps recent tasks in memory (bounded registry) — this list
        covers the current backend session and clears on restart.
      </p>
      {error && <div className="error-banner">{error}</div>}
      {runs && runs.length === 0 && (
        <div className="card">
          <p className="runs-note">No runs yet in this session. Run a task from the Workspace.</p>
        </div>
      )}
      {runs && runs.length > 0 && (
        <ul className="runs-list">
          {runs.map((r) => (
            <li key={r.task_id}>
              <button className="run-row" onClick={() => open(r)}>
                <span className={`run-icon status-${r.status}`}>
                  {STATUS_ICON[r.status] ?? '○'}
                </span>
                <span className="run-main">
                  <span className="run-title">{r.task || r.agent_label}</span>
                  <span className="run-meta">
                    <span className={`run-status run-status-${r.status}`}>{r.status.replace('_', ' ')}</span>
                    {' · '}{r.agent_label} · {timeAgo(r.created_at)} ({clockTime(r.created_at)})
                    {r.duration_seconds != null ? ` · ${r.duration_seconds.toFixed(1)}s` : ''}
                  </span>
                </span>
                <span className="run-open" aria-hidden="true">View flow →</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export interface OpenedRun {
  summary: RunSummary
  result: AgentResult | null
  taskState: string
}
