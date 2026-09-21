import { useMemo, useState } from 'react'
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import OrbitNode from './OrbitNode'
import {
  agentNodeState,
  nodeLines,
  orchestratorState,
  stepsForAgent,
  userTaskState,
  type CurrentRun,
  type FlowStep,
  type NodeId,
  type NodeState,
} from './graph'
import NodeInspector from './NodeInspector'

const nodeTypes = { orbit: OrbitNode }

const BASE_NODES: { id: NodeId; title: string; x: number; y: number }[] = [
  { id: 'usertask', title: 'User Task', x: 0, y: 150 },
  { id: 'orchestrator', title: 'Orchestrator', x: 250, y: 150 },
  { id: 'ai', title: 'AI Agent', x: 540, y: 0 },
  { id: 'data', title: 'Data Agent', x: 540, y: 100 },
  { id: 'ml', title: 'ML Agent', x: 540, y: 200 },
  { id: 'research', title: 'Research Agent', x: 540, y: 300 },
]

const BASE_EDGES: { id: string; source: NodeId; target: NodeId }[] = [
  { id: 'e-task-orch', source: 'usertask', target: 'orchestrator' },
  { id: 'e-orch-ai', source: 'orchestrator', target: 'ai' },
  { id: 'e-orch-data', source: 'orchestrator', target: 'data' },
  { id: 'e-orch-ml', source: 'orchestrator', target: 'ml' },
  { id: 'e-orch-research', source: 'orchestrator', target: 'research' },
]

export default function FlowCanvas({ run }: { run: CurrentRun | null }) {
  const [selected, setSelected] = useState<NodeId | null>(null)

  const nodes = useMemo<Node[]>(
    () =>
      BASE_NODES.map((n) => {
        const state =
          n.id === 'usertask'
            ? userTaskState(run)
            : n.id === 'orchestrator'
              ? orchestratorState(run)
              : agentNodeState(n.id, run)
        return {
          id: n.id,
          type: 'orbit',
          position: { x: n.x, y: n.y },
          data: {
            nodeId: n.id,
            state,
            title: n.title,
            lines: nodeLines(n.id, run),
            selected: selected === n.id,
            onSelect: setSelected,
          },
        } satisfies Node
      }),
    [run, selected],
  )

  const edges = useMemo<Edge[]>(
    () =>
      BASE_EDGES.filter((e) => typeof e.target === 'string').map((e) => {
        const stateOf = (id: string) =>
          (nodes.find((n) => n.id === id)?.data as { state?: string } | undefined)?.state ?? 'idle'
        const src = stateOf(e.source)
        const dst = stateOf(e.target as string)
        // Trace the executed path: an edge is highlighted when both of its
        // endpoints participated (running now, or reached a terminal state).
        const traced = src !== 'idle' && dst !== 'idle'
        // Motion only while data is actually flowing through the target.
        const flowing = dst === 'running' || dst === 'queued'
        return {
          id: e.id,
          source: e.source,
          target: e.target as string,
          animated: flowing,
          className: traced ? 'orbit-edge active' : 'orbit-edge',
        }
      }),
    [nodes],
  )

  const selectedResult =
    run?.result && run.result.agent === selected ? run.result : null
  const selectedState = nodes.find((n) => n.id === selected)?.data as
    | { state?: string }
    | undefined
  const steps: FlowStep[] | undefined =
    selected && selected !== 'usertask' && selected !== 'orchestrator'
      ? stepsForAgent(selected, selectedResult)
      : undefined

  return (
    <div className="flow-wrap">
      <div className="flow-canvas">
        <ReactFlow
          colorMode="dark"
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.4}
          maxZoom={1.6}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          nodesFocusable
          edgesFocusable={false}
        >
          <Background variant={BackgroundVariant.Dots} gap={18} size={1} />
          <Controls showInteractive={false} position="bottom-right" />
          <MiniMap pannable zoomable position="bottom-left" nodeColor="#4a5a80" nodeStrokeColor="#6ea8fe" />
        </ReactFlow>
      </div>
      <NodeInspector
        nodeId={selected}
        state={(selectedState?.state ?? 'idle') as NodeState}
        result={selectedResult}
        steps={steps}
        run={run}
        onClose={() => setSelected(null)}
      />
    </div>
  )
}
