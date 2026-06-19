/**
 * web/src/pages/CodeAtlas/ArchExplorer.tsx
 *
 * CodeAtlas — Phase 10: Architecture Explorer UI
 * ------------------------------------------------
 * Interactive graph visualization panel that fetches the architecture graph
 * from the Phase 9 API and renders it using @antv/g6 (already in the project).
 *
 * Scope (Phase 10 only)
 * ----------------------
 * - Fetch GET /api/codeatlas/graph
 * - Render nodes + edges as a force-directed graph using @antv/g6
 * - Click a node → show symbol name, file, language, dependency count
 * - Handle: loading / error / empty graph states gracefully
 * - No backend changes; no graph mutations
 *
 * Dependencies
 * ------------
 * All are already in web/package.json — NO new packages needed:
 *   @antv/g6         ^5.1.0   — graph rendering (same lib used by KnowledgeGraph page)
 *   @tanstack/react-query ^5  — data fetching (same as all other hooks)
 *   axios / next-request     — HTTP client (same pattern as all services)
 *   lucide-react             — icons
 *   tailwindcss              — styling
 */

import { Graph, IElementEvent } from '@antv/g6';
import { useQuery } from '@tanstack/react-query';
import {
  AlertCircle,
  Code2,
  FileCode2,
  GitBranch,
  Globe,
  Loader2,
  Network,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import request from '@/utils/next-request';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface GraphNode {
  id: string;
  kb_id: string;
  symbol: string;
  kind: 'function' | 'class';
  file: string;
  start_line: number;
  end_line: number;
  language: string;
}

interface GraphEdge {
  id: string;
  source_id: string;
  target_id: string;
  edge_type: string;
  kb_id: string;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface ApiResponse {
  code: number;
  message: string;
  data: GraphData;
}

// ---------------------------------------------------------------------------
// API hook
// ---------------------------------------------------------------------------

function useFetchArchGraph(kbId?: string) {
  return useQuery<GraphData>({
    queryKey: ['codeatlas-arch-graph', kbId],
    queryFn: async () => {
      const params = kbId ? `?kb_id=${encodeURIComponent(kbId)}` : '';
      const response = await request.get<ApiResponse>(
        `/api/codeatlas/graph${params}`,
      );
      // next-request wraps in response.data
      const body = (response as any).data ?? response;
      if (body?.code !== 0) {
        throw new Error(body?.message ?? 'Failed to fetch graph');
      }
      return body.data as GraphData;
    },
    initialData: { nodes: [], edges: [] },
    retry: 1,
    staleTime: 30_000,
  });
}

// ---------------------------------------------------------------------------
// Graph renderer — wraps @antv/g6 identically to the existing ForceGraph
// component in web/src/pages/dataset/knowledge-graph/force-graph.tsx
// ---------------------------------------------------------------------------

interface GraphViewProps {
  data: GraphData;
  onNodeClick: (node: GraphNode) => void;
}

function GraphView({ data, onNodeClick }: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<Graph | null>(null);

  const buildG6Data = useCallback(() => {
    const nodes = data.nodes.map((n) => ({
      id: n.id,
      data: {
        label: n.symbol,
        kind: n.kind,
        file: n.file,
        language: n.language,
        start_line: n.start_line,
        end_line: n.end_line,
        kb_id: n.kb_id,
        // Pass original node so click handler can recover it
        _raw: n,
      },
    }));

    const edges = data.edges.map((e) => ({
      id: e.id,
      source: e.source_id,
      target: e.target_id,
      data: { edge_type: e.edge_type },
    }));

    return { nodes, edges };
  }, [data]);

  const initGraph = useCallback(() => {
    if (!containerRef.current) return;

    // Destroy previous instance
    if (graphRef.current) {
      graphRef.current.destroy();
      graphRef.current = null;
    }

    const g6Data = buildG6Data();
    if (g6Data.nodes.length === 0) return;

    const graph = new Graph({
      container: containerRef.current,
      autoFit: 'view',
      autoResize: true,
      behaviors: [
        'drag-element',
        'drag-canvas',
        'zoom-canvas',
        {
          type: 'hover-activate',
          degree: 1,
        },
        {
          type: 'click-select',
        },
      ],
      layout: {
        type: 'force',
        preventOverlap: true,
        gravity: 2,
        factor: 3,
        linkDistance: 180,
      },
      node: {
        style: {
          size: (d: any) => (d.data?.kind === 'class' ? 48 : 36),
          fill: (d: any) =>
            d.data?.kind === 'class' ? '#6366f1' : '#0ea5e9',
          stroke: (d: any) =>
            d.data?.kind === 'class' ? '#4f46e5' : '#0284c7',
          lineWidth: 2,
          labelText: (d: any) => d.data?.label ?? d.id,
          labelFontSize: 12,
          labelPlacement: 'bottom',
          labelOffsetY: 4,
        },
      },
      edge: {
        style: {
          stroke: 'rgba(100,116,139,0.5)',
          lineWidth: 1.5,
          endArrow: true,
          endArrowSize: 8,
        },
      },
    });

    graphRef.current = graph;
    graph.setData(g6Data);
    graph.render();

    // Node click → expose raw node to parent
    graph.on('node:click', (evt: IElementEvent) => {
      const nodeId = (evt.target as any)?.id ?? (evt as any).itemId;
      if (!nodeId) return;
      const raw = g6Data.nodes.find((n) => n.id === nodeId);
      if (raw?.data?._raw) {
        onNodeClick(raw.data._raw as GraphNode);
      }
    });
  }, [buildG6Data, onNodeClick]);

  useEffect(() => {
    initGraph();
    return () => {
      graphRef.current?.destroy();
      graphRef.current = null;
    };
  }, [initGraph]);

  return (
    <div
      ref={containerRef}
      className="size-full"
      aria-label="Architecture dependency graph"
    />
  );
}

// ---------------------------------------------------------------------------
// Node detail panel
// ---------------------------------------------------------------------------

interface DetailPanelProps {
  node: GraphNode | null;
  edgeCount: number;
}

function DetailPanel({ node, edgeCount }: DetailPanelProps) {
  if (!node) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-400 gap-3 p-6 text-center">
        <Network className="w-10 h-10 opacity-40" />
        <p className="text-sm font-medium">Click a node to see details</p>
        <p className="text-xs opacity-70">
          Purple = class&nbsp;&nbsp;·&nbsp;&nbsp;Blue = function
        </p>
      </div>
    );
  }

  const rows: Array<{ icon: React.ReactNode; label: string; value: string }> = [
    {
      icon: <Code2 className="w-3.5 h-3.5 shrink-0" />,
      label: 'Symbol',
      value: node.symbol,
    },
    {
      icon: <FileCode2 className="w-3.5 h-3.5 shrink-0" />,
      label: 'File',
      value: node.file,
    },
    {
      icon: <Globe className="w-3.5 h-3.5 shrink-0" />,
      label: 'Language',
      value: node.language,
    },
    {
      icon: <GitBranch className="w-3.5 h-3.5 shrink-0" />,
      label: 'Lines',
      value: `${node.start_line}–${node.end_line}`,
    },
    {
      icon: <Network className="w-3.5 h-3.5 shrink-0" />,
      label: 'Dependencies',
      value: String(edgeCount),
    },
  ];

  return (
    <div className="p-4 flex flex-col gap-3">
      {/* Kind badge */}
      <div className="flex items-center gap-2">
        <span
          className={`
            inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold
            ${
              node.kind === 'class'
                ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300'
                : 'bg-sky-100 text-sky-700 dark:bg-sky-900/40 dark:text-sky-300'
            }
          `}
        >
          {node.kind}
        </span>
        <span className="font-semibold text-sm text-slate-800 dark:text-slate-100 truncate">
          {node.symbol}
        </span>
      </div>

      {/* Detail rows */}
      <dl className="flex flex-col gap-2">
        {rows.map(({ icon, label, value }) => (
          <div key={label} className="flex items-start gap-2 text-xs">
            <dt className="flex items-center gap-1 text-slate-500 shrink-0 w-24">
              {icon}
              {label}
            </dt>
            <dd
              className="text-slate-800 dark:text-slate-100 break-all font-mono"
              title={value}
            >
              {value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-slate-500 p-8 text-center">
      <div className="w-16 h-16 rounded-2xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center">
        <Network className="w-8 h-8 opacity-50" />
      </div>
      <div>
        <p className="font-semibold text-slate-700 dark:text-slate-200 mb-1">
          No architecture graph yet
        </p>
        <p className="text-sm text-slate-400 max-w-xs">
          Add a code repository as a knowledge-base source, then ingest it to
          build the architecture graph.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page component
// ---------------------------------------------------------------------------

export default function ArchExplorer() {
  // Optional kb_id from query string — e.g. /codeatlas/arch-explorer?kb_id=xxx
  const kbId =
    typeof window !== 'undefined'
      ? new URLSearchParams(window.location.search).get('kb_id') ?? undefined
      : undefined;

  const { data, isLoading, isError, error } = useFetchArchGraph(kbId);

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  // Count edges connecting to the selected node
  const dependencyCount = selectedNode
    ? (data?.edges ?? []).filter(
        (e) =>
          e.source_id === selectedNode.id || e.target_id === selectedNode.id,
      ).length
    : 0;

  const isEmpty = !isLoading && !isError && (data?.nodes ?? []).length === 0;

  return (
    <div className="flex flex-col h-full bg-white dark:bg-slate-900">
      {/* ── Header ──────────────────────────────────────────────────── */}
      <header className="flex items-center gap-3 px-5 py-3 border-b border-slate-200 dark:border-slate-700 shrink-0">
        <Network className="w-5 h-5 text-indigo-500" />
        <h1 className="text-base font-semibold text-slate-800 dark:text-slate-100">
          Architecture Explorer
        </h1>
        {data && data.nodes.length > 0 && (
          <span className="ml-auto text-xs text-slate-400">
            {data.nodes.length} symbols · {data.edges.length} edges
          </span>
        )}
      </header>

      {/* ── Body ────────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Graph canvas — left / main area */}
        <main className="flex-1 overflow-hidden relative">
          {/* Loading */}
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-white/60 dark:bg-slate-900/60 z-10">
              <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
            </div>
          )}

          {/* Error */}
          {isError && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-red-500 p-8">
              <AlertCircle className="w-10 h-10" />
              <p className="font-semibold">Failed to load graph</p>
              <p className="text-sm text-center text-slate-500 max-w-sm">
                {(error as Error)?.message ??
                  'An unexpected error occurred while fetching the architecture graph.'}
              </p>
            </div>
          )}

          {/* Empty state */}
          {isEmpty && <EmptyState />}

          {/* Graph */}
          {!isLoading && !isError && !isEmpty && data && (
            <GraphView data={data} onNodeClick={setSelectedNode} />
          )}
        </main>

        {/* Detail panel — right sidebar */}
        <aside
          className="w-64 shrink-0 border-l border-slate-200 dark:border-slate-700 overflow-y-auto"
          aria-label="Node details"
        >
          <div className="px-4 py-2 border-b border-slate-100 dark:border-slate-800">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
              Node Details
            </p>
          </div>
          <DetailPanel node={selectedNode} edgeCount={dependencyCount} />
        </aside>
      </div>
    </div>
  );
}
