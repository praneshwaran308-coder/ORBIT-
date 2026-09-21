import { Component, type ReactNode } from 'react'
import type { AgentResult, DataReport, MLReport, ResearchReport } from '../types/orbit'
import AgentBadge from './AgentBadge'

class ErrorBoundary extends Component<{ label: string; children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="card error-card">
          <h3>Could not render {this.props.label}</h3>
          <p className="error-message">{this.state.error.message}</p>
        </div>
      )
    }
    return this.props.children
  }
}

const TERMINAL: AgentResult['status'][] = ['completed', 'failed', 'partial', 'insufficient_data']

export function isTerminal(status: string): boolean {
  return TERMINAL.includes(status as AgentResult['status'])
}

export default function ResultView({ result }: { result: AgentResult | null }) {
  if (!result) return null

  if (result.status === 'failed') {
    return (
      <div className="card error-card">
        <AgentBadge agent={result.agent} status={result.status} />
        <h3>Task failed</h3>
        <p className="error-message">{result.message}</p>
      </div>
    )
  }

  if (result.status === 'insufficient_data') {
    return (
      <div className="card insufficient-card">
        <AgentBadge agent={result.agent} status={result.status} />
        <h3>Insufficient data</h3>
        <p>{result.message}</p>
      </div>
    )
  }

  const format = result.data?.['format'] as string | undefined

  return (
    <div className="result-stack">
      <div className="result-header">
        <AgentBadge agent={result.agent} status={result.status} />
        <span className="result-message">{result.message}</span>
      </div>

      {result.status === 'partial' && (
        <div className="card partial-card">
          <strong>Partial result:</strong> {result.message}
        </div>
      )}

      {format === 'data_report' && (
        <ErrorBoundary label="Data report">
          <DataReportView
            report={
              ((result.data as Record<string, unknown>)['report'] as unknown as DataReport) ?? {
                shape: { rows: 0, columns: 0 },
                columns: [],
                total_missing_cells: 0,
                duplicate_rows: 0,
                numeric_summary: {},
                correlations: [],
                insights: [],
              }
            }
          />
          {typeof (result.data as Record<string, unknown>)['narrative'] === 'string' && (
            <div className="card markdown-card">
              <h3>Narrative</h3>
              <Markdown text={String((result.data as Record<string, unknown>)['narrative'])} />
            </div>
          )}
        </ErrorBoundary>
      )}
      {format === 'ml_report' && (
        <ErrorBoundary label="ML report">
          <MLReportView report={result.data as unknown as MLReport} />
        </ErrorBoundary>
      )}
      {format === 'research_report' && (
        <ErrorBoundary label="Research report">
          <ResearchView report={result.data as unknown as ResearchReport} />
        </ErrorBoundary>
      )}
      {(format === 'markdown' || format === undefined) &&
        'response' in (result.data ?? {}) && (
          <AIMarkdown text={String(result.data['response'])} />
        )}
    </div>
  )
}

function AIMarkdown({ text }: { text: string }) {
  return (
    <div className="card markdown-card">
      <Markdown text={text} />
    </div>
  )
}

/** Minimal, dependency-free markdown-ish renderer (headings, bold, lists, code). */
export function Markdown({ text }: { text: string }) {
  const lines = text.split('\n')
  const out: React.ReactNode[] = []
  let listBuffer: string[] = []
  let olBuffer: string[] = []
  let codeBuffer: string[] = []

  const flushList = (key: string) => {
    if (listBuffer.length) {
      out.push(
        <ul key={key}>
          {listBuffer.map((li, i) => (
            <li key={i} dangerouslySetInnerHTML={{ __html: inline(li) }} />
          ))}
        </ul>,
      )
      listBuffer = []
    }
  }

  const flushOl = (key: string) => {
    if (olBuffer.length) {
      out.push(
        <ol key={key}>
          {olBuffer.map((li, i) => (
            <li key={i} dangerouslySetInnerHTML={{ __html: inline(li) }} />
          ))}
        </ol>,
      )
      olBuffer = []
    }
  }

  lines.forEach((raw, idx) => {
    const line = raw.trimEnd()

    // fenced code blocks — collect lines verbatim until the closing fence
    if (/^```/.test(line.trim())) {
      if (codeBuffer.length) {
        out.push(<pre key={idx} className="code-block"><code>{codeBuffer.join('\n')}</code></pre>)
        codeBuffer = []
      } else {
        flushList(`ul-${idx}`)
        flushOl(`ol-${idx}`)
        codeBuffer = []
      }
      return
    }
    if (codeBuffer.length) {
      codeBuffer.push(raw)
      return
    }

    if (/^\d+\.\s+/.test(line)) {
      flushList(`ul-${idx}`)
      olBuffer.push(line.replace(/^\d+\.\s+/, ''))
      return
    }
    flushOl(`ol-${idx}`)
    if (/^[-*]\s+/.test(line)) {
      listBuffer.push(line.replace(/^[-*]\s+/, ''))
      return
    }
    flushList(`ul-${idx}`)
    if (/^###\s/.test(line)) {
      out.push(<h4 key={idx} dangerouslySetInnerHTML={{ __html: inline(line.slice(4)) }} />)
    } else if (/^##\s/.test(line)) {
      out.push(<h3 key={idx} dangerouslySetInnerHTML={{ __html: inline(line.slice(3)) }} />)
    } else if (/^#\s/.test(line)) {
      out.push(<h2 key={idx} dangerouslySetInnerHTML={{ __html: inline(line.slice(2)) }} />)
    } else if (line.trim() === '') {
      // paragraph spacing
    } else {
      out.push(<p key={idx} dangerouslySetInnerHTML={{ __html: inline(line) }} />)
    }
  })
  if (codeBuffer.length) {
    out.push(<pre key="code-final" className="code-block"><code>{codeBuffer.join('\n')}</code></pre>)
  }
  flushList('ul-final')
  flushOl('ol-final')
  return <div className="markdown">{out}</div>
}

function inline(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
}

function DataReportView({ report }: { report: DataReport }) {
  const numericCols = Object.keys(report.numeric_summary ?? {})
  return (
    <div className="stack">
      <div className="card">
        <h3>Dataset overview</h3>
        <p className="muted">
          {report.shape.rows.toLocaleString()} rows × {report.shape.columns} columns ·{' '}
          {report.total_missing_cells} missing cells · {report.duplicate_rows} duplicate rows
        </p>
        <table className="table">
          <thead>
            <tr>
              <th>Column</th><th>Type</th><th>Missing</th><th>Unique</th>
            </tr>
          </thead>
          <tbody>
            {report.columns.map((c) => (
              <tr key={c.name}>
                <td>{c.name}</td>
                <td><code>{c.dtype}</code></td>
                <td className={c.missing > 0 ? 'warn' : ''}>{c.missing} ({c.missing_pct}%)</td>
                <td>{c.unique.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {numericCols.length > 0 && (
        <div className="card">
          <h3>Numeric summary</h3>
          <table className="table">
            <thead>
              <tr><th>Column</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th><th>Std</th></tr>
            </thead>
            <tbody>
              {numericCols.map((col) => {
                const s = report.numeric_summary[col]
                return (
                  <tr key={col}>
                    <td>{col}</td>
                    <td>{fmt(s.mean)}</td>
                    <td>{fmt(s.median)}</td>
                    <td>{fmt(s.min)}</td>
                    <td>{fmt(s.max)}</td>
                    <td>{fmt(s.std)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {report.correlations?.length > 0 && (
        <div className="card">
          <h3>Top correlations</h3>
          <ul>
            {report.correlations.map((p, i) => (
              <li key={i}>
                <strong>{p.a}</strong> ↔ <strong>{p.b}</strong>: r = {fmt(p.r, 3)}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="card">
        <h3>Insights</h3>
        <ul>
          {report.insights.map((ins, i) => (
            <li key={i}>{ins}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}

function MLReportView({ report }: { report: MLReport }) {
  return (
    <div className="stack">
      <div className="card">
        <h3>
          {report.mode === 'regression' ? 'Regression' : 'Classification'} · target{' '}
          <code>{report.target}</code>
        </h3>
        <p className="muted">
          {report.dataset.filename} · {report.dataset.rows} rows ({report.dataset.train_rows}{' '}
          train / {report.dataset.test_rows} test) · {report.dataset.features.length} features
        </p>
        {report.dataset.class_balance && (
          <p className="muted">
            Class balance:{' '}
            {Object.entries(report.dataset.class_balance)
              .map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`)
              .join(' · ')}
          </p>
        )}
      </div>

      <div className="card">
        <h3>Model comparison</h3>
        <table className="table">
          <thead>
            <tr>
              <th>Model</th>
              {report.mode === 'regression' ? (
                <>
                  <th>R²</th><th>MAE</th><th>RMSE</th>
                </>
              ) : (
                <>
                  <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {report.models.map((m) => (
              <tr key={m.model} className={m.model === report.best_model ? 'best-row' : ''}>
                <td>
                  {m.model} {m.model === report.best_model && <span className="chip">best</span>}
                </td>
                {m.error ? (
                  <td colSpan={report.mode === 'regression' ? 3 : 4} className="warn">{m.error}</td>
                ) : report.mode === 'regression' ? (
                  <>
                    <td>{fmt(m.r2, 3)}</td>
                    <td>{fmt(m.mae, 2)}</td>
                    <td>{fmt(m.rmse, 2)}</td>
                  </>
                ) : (
                  <>
                    <td>{fmt(m.accuracy, 3)}</td>
                    <td>{fmt(m.precision, 3)}</td>
                    <td>{fmt(m.recall, 3)}</td>
                    <td>{fmt(m.f1_weighted, 3)}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {report.confusion_matrix && (
        <div className="card">
          <h3>Confusion matrix ({report.best_model})</h3>
          <table className="table matrix">
            <thead>
              <tr>
                <th></th>
                {report.confusion_matrix.labels.map((l) => (
                  <th key={l}>{l}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {report.confusion_matrix.matrix.map((row, i) => (
                <tr key={i}>
                  <th>{report.confusion_matrix!.labels[i]}</th>
                  {row.map((v, j) => (
                    <td key={j} className={i === j ? 'diag' : ''}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function ResearchView({ report }: { report: ResearchReport }) {
  return (
    <div className="stack">
      {report.synthesis && (
        <div className="card markdown-card">
          <h3>Grounded synthesis</h3>
          <Markdown text={report.synthesis} />
        </div>
      )}
      <div className="card">
        <h3>Sources ({report.verified_evidence_count} verified / {report.sources.length} found)</h3>
        <p className="muted disclaimer">{report.evidence_disclaimer}</p>
        <ul className="sources">
          {report.sources.map((s, i) => (
            <li key={i} className={`source source-${s.evidence_status}`}>
              <div className="source-head">
                <span className={`chip chip-${s.evidence_status}`}>
                  {s.evidence_status}
                </span>
                <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a>
              </div>
              <div className="muted">
                {s.publisher}
                {s.published ? ` · ${s.published.slice(0, 10)}` : ' · date unknown'}
                {s.kind === 'article' && s.extracted_chars > 0 && ` · ${s.extracted_chars} chars extracted`}
              </div>
              {s.snippet && s.evidence_status === 'unverified' && (
                <div className="snippet">Search snippet (not evidence): {s.snippet}</div>
              )}
              {s.evidence_note && <div className="muted small">{s.evidence_note}</div>}
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

function fmt(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined) return '—'
  if (Number.isInteger(v)) return v.toLocaleString()
  return v.toLocaleString(undefined, { maximumFractionDigits: digits })
}
