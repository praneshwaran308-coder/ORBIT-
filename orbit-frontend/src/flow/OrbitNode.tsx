import { Handle, Position } from '@xyflow/react'
import type { NodeProps } from '@xyflow/react'
import type { NodeId, NodeState } from './graph'

const STATE_META: Record<NodeState, { icon: string; cls: string; label: string }> = {
  idle: { icon: '○', cls: 'idle', label: 'idle' },
  queued: { icon: '◔', cls: 'queued', label: 'queued' },
  running: { icon: '●', cls: 'running', label: 'running' },
  completed: { icon: '✓', cls: 'completed', label: 'completed' },
  failed: { icon: '✕', cls: 'failed', label: 'failed' },
  partial: { icon: '◑', cls: 'partial', label: 'partial' },
  insufficient_data: { icon: '△', cls: 'insufficient', label: 'insufficient data' },
}

type OrbitNodeData = {
  nodeId: NodeId
  state: NodeState
  title: string
  lines: string[]
  selected: boolean
  onSelect: (id: NodeId) => void
}

export default function OrbitNode({ data }: NodeProps) {
  const { nodeId, state, title, lines, selected, onSelect } = data as unknown as OrbitNodeData
  const meta = STATE_META[state] ?? STATE_META.idle
  const active = state !== 'idle'
  return (
    <div
      className={`orbit-node state-${meta.cls} ${active ? 'active' : ''} ${selected ? 'selected' : ''}`}
      onClick={() => onSelect(nodeId)}
      role="button"
      aria-label={`${title}: ${meta.label}`}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onSelect(nodeId)
      }}
    >
      <div className="orbit-node-head">
        <span className={`dot dot-${meta.cls}`} aria-hidden="true" />
        <span className="orbit-node-title">{title}</span>
        <span className={`chip chip-${meta.cls}`}>{meta.label}</span>
      </div>
      {lines.map((l) => (
        <div key={l} className="orbit-node-line">{l}</div>
      ))}
      <Handle type="target" position={Position.Left} isConnectable={false} />
      <Handle type="source" position={Position.Right} isConnectable={false} />
    </div>
  )
}
