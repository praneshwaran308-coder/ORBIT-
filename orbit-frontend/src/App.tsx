import { useCallback, useEffect, useRef, useState } from 'react'
import { getHealth, pollStatusFull, runTask } from './api/orbit'
import ResultView, { isTerminal } from './components/ResultView'
import FlowCanvas from './flow/FlowCanvas'
import RunsView, { type OpenedRun } from './flow/RunsView'
import type { AgentId, AgentResult, HealthInfo, RunAccepted } from './types/orbit'

type View = 'workspace' | 'flow' | 'runs'

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const AGENT_OPTIONS = [
  { value: '', label: 'Auto (deterministic routing)' },
  { value: 'ai', label: 'AI Agent — general, explanations, summarization' },
  { value: 'data', label: 'Data Agent — CSV statistics & insights' },
  { value: 'ml', label: 'ML Agent — train & evaluate models' },
  { value: 'research', label: 'Research Agent — grounded web research' },
]

const EXAMPLES = [
  'Explain how transformers work in simple terms',
  'Summarize the key ideas of entropy in information theory',
  'What is the latest news about the James Webb telescope?',
  'Research the current state of fusion energy with sources',
  'Analyze this CSV: statistics, correlations and insights',
  'Train a model to predict price from this CSV',
]

export default function App() {
  const [task, setTask] = useState('')
  const [agentOverride, setAgentOverride] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [accepted, setAccepted] = useState<RunAccepted | null>(null)
  const [result, setResult] = useState<AgentResult | null>(null)
  const [running, setRunning] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [view, setView] = useState<View>('workspace')
  const [taskState, setTaskState] = useState<string>('unknown')
  const [submittedTask, setSubmittedTask] = useState('')
  const pollRef = useRef<number | null>(null)
  const startedRef = useRef<number>(0)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const acceptFile = useCallback((f: File | null | undefined) => {
    if (!f) return
    if (!f.name.toLowerCase().endsWith('.csv')) {
      setFileError(`"${f.name}" is not a CSV file — please choose a .csv`)
      setFile(null)
      return
    }
    setFileError(null)
    setFile(f)
  }, [])

  const handleSubmit = useCallback(async () => {
    if (!task.trim() || running) return
    setError(null)
    setResult(null)
    setAccepted(null)
    setRunning(true)
    setSubmittedTask(task.trim())
    setTaskState('queued')
    startedRef.current = Date.now()
    setElapsed(0)
    stopPolling()

    try {
      const acc = await runTask(task.trim(), agentOverride || null, file)
      setAccepted(acc)
      pollRef.current = window.setInterval(async () => {
        setElapsed(Math.round((Date.now() - startedRef.current) / 100) / 10)
        try {
          const r = await pollStatusFull(acc.task_id)
          setTaskState(r.status)
          if (r.result) {
            setResult(r.result)
            if (isTerminal(r.result.status)) {
              stopPolling()
              setRunning(false)
            }
          }
        } catch (e) {
          setError(e instanceof Error ? e.message : String(e))
          stopPolling()
          setRunning(false)
        }
      }, 600)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setRunning(false)
    }
  }, [task, running, agentOverride, file, stopPolling])

  useEffect(() => stopPolling, [stopPolling])

  const currentRun = {
    task: submittedTask || task,
    accepted,
    result,
    taskState,
  }

  const openRun = useCallback((opened: OpenedRun) => {
    stopPolling()
    setRunning(false)
    setSubmittedTask(opened.summary.task)
    setAccepted({
      task_id: opened.summary.task_id,
      agent: opened.summary.agent as AgentId,
      agent_label: opened.summary.agent_label,
      status: opened.summary.status,
      routing_reason: opened.summary.routing_reason,
      poll_url: `/status/${opened.summary.task_id}`,
    })
    setResult(opened.result)
    setTaskState(opened.taskState)
    setView('flow')
  }, [stopPolling])

  return (
    <div className={view === 'flow' ? 'workspace flow-mode' : 'workspace'}>
      <header>
        <h1>
          <span className="logo">◍</span> ORBIT <span className="subtitle">Multi-Agent Workspace</span>
        </h1>
        <nav className="view-tabs" aria-label="ORBIT views">
          {(['workspace', 'flow', 'runs'] as View[]).map((v) => (
            <button
              key={v}
              className={`tab ${view === v ? 'active' : ''}`}
              onClick={() => setView(v)}
              aria-current={view === v ? 'page' : undefined}
            >
              {v === 'workspace' ? 'Workspace' : v === 'flow' ? 'Flow' : 'Runs'}
            </button>
          ))}
        </nav>
      </header>

      {view === 'workspace' && (
      <section className="input-panel card">
        <textarea
          value={task}
          onChange={(e) => setTask(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit()
          }}
          placeholder="Describe your task… e.g. 'Analyze this CSV' or 'Research the latest on fusion energy'"
          rows={4}
        />
        <div className="input-row">
          <select value={agentOverride} onChange={(e) => setAgentOverride(e.target.value)}>
            {AGENT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>

          <label
            className={`file-drop ${dragOver ? 'drag' : ''} ${file ? 'has-file' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragOver(false)
              acceptFile(e.dataTransfer.files?.[0])
            }}
          >
            <input
              type="file"
              accept=".csv"
              onChange={(e) => acceptFile(e.target.files?.[0])}
            />
            {file ? `📎 ${file.name} · ${fmtSize(file.size)}` : ' Drop a CSV here or click to browse'}
          </label>

          {file && (
            <button className="ghost" onClick={() => { setFile(null); setFileError(null) }} title="Remove file">
              ✕
            </button>
          )}

          <button className="primary" onClick={handleSubmit} disabled={running || !task.trim()}>
            {running ? 'Running…' : 'Run task'}
          </button>
        </div>

        {fileError && <div className="file-error">⚠ {fileError}</div>}

        {running && (
          <div className="loading">
            <span className="spinner" /> {accepted ? `Routed to ${accepted.agent_label}` : 'Submitting…'}
            {' · '}{elapsed.toFixed(1)}s
          </div>
        )}
        {error && <div className="error-banner">{error}</div>}
        {!health && (
          <div className="error-banner">
            Backend unreachable — start it with uvicorn (see README).
          </div>
        )}

        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="example" onClick={() => setTask(ex)}>
              {ex}
            </button>
          ))}
        </div>
      </section>

      )}

      {view === 'workspace' && accepted && (
        <div className={`routing-info card routing-${accepted.agent}`}>
          <span className="routing-arrow" aria-hidden="true">→</span>
          <strong>Routed to {accepted.agent_label}</strong>
          <span className="muted"> — {accepted.routing_reason}</span>
          <span className="muted task-id"> (task {accepted.task_id.slice(0, 8)}…)</span>
        </div>
      )}

      {view === 'workspace' && <ResultView result={result} />}

      {view === 'flow' && <FlowCanvas run={currentRun} />}

      {view === 'runs' && <RunsView onOpenRun={openRun} />}
    </div>
  )
}
