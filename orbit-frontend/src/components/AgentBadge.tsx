import type { AgentId, AgentStatus } from '../types/orbit'

const AGENT_META: Record<AgentId, { label: string; className: string }> = {
  ai: { label: 'AI Agent', className: 'badge-ai' },
  data: { label: 'Data Agent', className: 'badge-data' },
  ml: { label: 'ML Agent', className: 'badge-ml' },
  research: { label: 'Research Agent', className: 'badge-research' },
  orchestrator: { label: 'Orchestrator', className: 'badge-orchestrator' },
}

const STATUS_CLASS: Partial<Record<AgentStatus, string>> = {
  completed: 'status-completed',
  failed: 'status-failed',
  partial: 'status-partial',
  insufficient_data: 'status-insufficient',
}

export default function AgentBadge({ agent, status }: { agent: AgentId; status: AgentStatus }) {
  const meta = AGENT_META[agent] ?? AGENT_META.orchestrator
  return (
    <span className={`badge ${meta.className} ${STATUS_CLASS[status] ?? ''}`}>
      {meta.label}
      <span className="badge-status">{status.replace('_', ' ')}</span>
    </span>
  )
}
