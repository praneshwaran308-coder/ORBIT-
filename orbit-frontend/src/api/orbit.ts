import type { AgentResult, HealthInfo, RunAccepted, RunSummary } from '../types/orbit'

/**
 * API base URL: '/api' in development (Vite dev proxy strips the prefix;
 * see vite.config.ts). In hosted builds, VITE_API_BASE supplies the backend
 * origin directly (backend routes live at root, e.g. https://<backend>/health).
 */
const BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? JSON.stringify(body)
    } catch {
      /* keep statusText */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export async function getHealth(): Promise<HealthInfo> {
  const res = await fetch(`${BASE}/health`)
  return handle<HealthInfo>(res)
}

export async function runTask(
  task: string,
  agentOverride: string | null,
  file: File | null,
): Promise<RunAccepted> {
  if (file) {
    const form = new FormData()
    form.append('task', task)
    if (agentOverride) form.append('agent', agentOverride)
    form.append('file', file)
    const res = await fetch(`${BASE}/run/file`, { method: 'POST', body: form })
    return handle<RunAccepted>(res)
  }
  const res = await fetch(`${BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task, agent: agentOverride ?? undefined }),
  })
  return handle<RunAccepted>(res)
}

export async function pollStatus(taskId: string): Promise<AgentResult | null> {
  const res = await fetch(`${BASE}/status/${taskId}`)
  return handle<StatusResponse>(res).then((r) => r.result)
}

export async function pollStatusFull(taskId: string): Promise<StatusResponse> {
  const res = await fetch(`${BASE}/status/${taskId}`)
  return handle<StatusResponse>(res)
}

export async function getRuns(): Promise<RunSummary[]> {
  const res = await fetch(`${BASE}/runs`)
  return handle<RunSummary[]>(res)
}

interface StatusResponse {
  task_id: string
  result: AgentResult | null
  status: string
}
