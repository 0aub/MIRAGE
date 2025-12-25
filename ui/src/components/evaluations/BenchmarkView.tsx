import { useState, useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Play,
  Loader2,
  Clock,
  Layers,
  BarChart3,
  CheckCircle2,
  XCircle,
  Zap,
  Trophy,
  Info,
  FileText,
  Database,
  Network,
  Target,
  Users,
  GitBranch,
  Minimize2,
  Download,
  Plus,
  Trash2,
  FileJson,
  TrendingUp,
  Activity,
  Copy,
  Check,
} from "lucide-react";
import { chatApi, RetrievalMode, ChunkInfo } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

// ==================== TYPES ====================

interface SingleRunResult {
  mode: RetrievalMode;
  query: string;
  queryIndex: number;
  answer: string;
  chunks: ChunkInfo[];
  chunksCount: number;
  retrievalTimeMs: number;
  generationTimeMs: number;
  totalTimeMs: number;
  error?: string;
  status: "pending" | "running" | "completed" | "error";
  retrievalStats?: {
    total_chunks: number;
    vector_chunks: number;
    graph_1hop_chunks: number;
    graph_2hop_chunks: number;
    graph_total: number;
    entities_used: string[];
  };
  entitiesFound?: string[];
}

interface BenchmarkReport {
  timestamp: string;
  config: {
    modes: RetrievalMode[];
    topK: number;
    queries: string[];
  };
  results: SingleRunResult[];
  statistics: {
    byMode: Record<RetrievalMode, ModeStats>;
    overall: AggregateStats;
  };
}

interface ModeStats {
  mode: RetrievalMode;
  avgTotalTime: number;
  avgRetrievalTime: number;
  avgGenerationTime: number;
  avgChunks: number;
  successRate: number;
}

interface AggregateStats {
  avgTotalTime: number;
  avgRetrievalTime: number;
  avgGenerationTime: number;
  avgChunks: number;
  minTime: number;
  maxTime: number;
  count: number;
}

// ==================== CONSTANTS ====================

const ALL_MODES: RetrievalMode[] = [
  "vector", "local", "global", "hybrid", "semantic", "mix", "global_search", "drift",
];

const MODE_COLORS: Record<RetrievalMode, string> = {
  vector: "bg-blue-500",
  local: "bg-green-500",
  global: "bg-purple-500",
  hybrid: "bg-orange-500",
  semantic: "bg-pink-500",
  mix: "bg-red-500",
  global_search: "bg-teal-500",
  drift: "bg-cyan-500",
};

const MODE_HEX_COLORS: Record<RetrievalMode, string> = {
  vector: "#3b82f6",
  local: "#22c55e",
  global: "#a855f7",
  hybrid: "#f97316",
  semantic: "#ec4899",
  mix: "#ef4444",
  global_search: "#14b8a6",
  drift: "#06b6d4",
};

const MODE_DESCRIPTIONS: Record<RetrievalMode, string> = {
  vector: "Vector similarity search",
  local: "Entity-based graph traversal",
  global: "Relationship-focused traversal",
  hybrid: "Fusion of multiple modes",
  semantic: "Cross-encoder re-ranking",
  mix: "All modes combined",
  global_search: "Community-based summaries",
  drift: "Dynamic global-to-local with claims",
};

// Predefined query sets for testing
const QUERY_SETS: Record<string, { name: string; queries: string[] }> = {
  arabic_factual: {
    name: "Arabic Factual",
    queries: [
      "ما هي جائزة الحكومة الرقمية؟",
      "من هم المشاركون في منتدى الحكومة الرقمية؟",
      "ما هو مؤشر قياس التحول الرقمي؟",
      "ما هي أهداف رؤية 2030 للحكومة الرقمية؟",
    ],
  },
  english_factual: {
    name: "English Factual",
    queries: [
      "What is the Digital Government Forum 2025?",
      "Who are the participants in the Digital Government Forum?",
      "What are the digital transformation metrics?",
      "What services does the digital government provide?",
    ],
  },
  mixed_complex: {
    name: "Mixed Complex",
    queries: [
      "ما العلاقة بين الحكومة الرقمية والتحول الوطني؟",
      "How does the digital government initiative support Vision 2030?",
      "ما هي أبرز الإنجازات في مجال الحكومة الرقمية؟",
      "What partnerships exist for digital government development?",
    ],
  },
  overview_questions: {
    name: "Overview Questions",
    queries: [
      "What are the main themes discussed?",
      "ما هي أهم المواضيع المطروحة؟",
      "Summarize the key initiatives",
      "ما هي الرؤية العامة للتحول الرقمي؟",
    ],
  },
};

// ==================== COMPONENT ====================

export function BenchmarkView() {
  // State
  const [activeTab, setActiveTab] = useState<"setup" | "results" | "report">("setup");
  const [queries, setQueries] = useState<string[]>([""]);
  const [selectedModes, setSelectedModes] = useState<RetrievalMode[]>(ALL_MODES);
  const [topK, setTopK] = useState(5);
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<SingleRunResult[]>([]);
  const [progress, setProgress] = useState({ current: 0, total: 0, label: "" });
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();

  // Modal states
  const [chunksModal, setChunksModal] = useState<{
    open: boolean;
    result: SingleRunResult | null;
  }>({ open: false, result: null });

  // Computed values
  const totalRuns = useMemo(() => {
    const validQueries = queries.filter(q => q.trim()).length;
    return validQueries * selectedModes.length;
  }, [queries, selectedModes]);

  const completedResults = useMemo(() =>
    results.filter(r => r.status === "completed"), [results]);

  // Calculate statistics
  const statistics = useMemo(() => {
    if (completedResults.length === 0) return null;

    const byMode: Record<RetrievalMode, ModeStats> = {} as any;

    // Group by mode
    for (const mode of selectedModes) {
      const modeResults = completedResults.filter(r => r.mode === mode);

      if (modeResults.length > 0) {
        byMode[mode] = {
          mode,
          avgTotalTime: avg(modeResults.map(r => r.totalTimeMs)),
          avgRetrievalTime: avg(modeResults.map(r => r.retrievalTimeMs)),
          avgGenerationTime: avg(modeResults.map(r => r.generationTimeMs)),
          avgChunks: avg(modeResults.map(r => r.chunksCount)),
          successRate: modeResults.length / queries.filter(q => q.trim()).length,
        };
      }
    }

    return {
      byMode,
      overall: calcAggregateStats(completedResults),
    };
  }, [completedResults, selectedModes, queries]);

  // Helper functions
  function avg(arr: number[]): number {
    return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  }

  function calcAggregateStats(results: SingleRunResult[]): AggregateStats {
    if (results.length === 0) {
      return { avgTotalTime: 0, avgRetrievalTime: 0, avgGenerationTime: 0, avgChunks: 0, minTime: 0, maxTime: 0, count: 0 };
    }
    const times = results.map(r => r.totalTimeMs);
    return {
      avgTotalTime: avg(times),
      avgRetrievalTime: avg(results.map(r => r.retrievalTimeMs)),
      avgGenerationTime: avg(results.map(r => r.generationTimeMs)),
      avgChunks: avg(results.map(r => r.chunksCount)),
      minTime: Math.min(...times),
      maxTime: Math.max(...times),
      count: results.length,
    };
  }

  // Run benchmark
  const runBenchmark = async () => {
    const validQueries = queries.filter(q => q.trim());
    if (validQueries.length === 0) {
      toast({ title: "Error", description: "Add at least one query", variant: "destructive" });
      return;
    }
    if (selectedModes.length === 0) {
      toast({ title: "Error", description: "Select at least one mode", variant: "destructive" });
      return;
    }

    setIsRunning(true);
    setActiveTab("results");

    // Build run queue
    const runQueue: { query: string; queryIndex: number; mode: RetrievalMode }[] = [];

    for (let qi = 0; qi < validQueries.length; qi++) {
      for (const mode of selectedModes) {
        runQueue.push({ query: validQueries[qi], queryIndex: qi, mode });
      }
    }

    // Initialize results
    const initialResults: SingleRunResult[] = runQueue.map(run => ({
      ...run,
      answer: "",
      chunks: [],
      chunksCount: 0,
      retrievalTimeMs: 0,
      generationTimeMs: 0,
      totalTimeMs: 0,
      status: "pending",
    }));
    setResults(initialResults);
    setProgress({ current: 0, total: runQueue.length, label: "Starting..." });

    // Run sequentially
    for (let i = 0; i < runQueue.length; i++) {
      const run = runQueue[i];
      const resultKey = `${run.queryIndex}-${run.mode}`;

      setProgress({
        current: i + 1,
        total: runQueue.length,
        label: `${run.mode} Q${run.queryIndex + 1}`,
      });

      // Update status to running
      setResults(prev => prev.map((r, idx) =>
        idx === i ? { ...r, status: "running" } : r
      ));

      try {
        const response = await chatApi.ask({
          message: run.query,
          retrieval_mode: run.mode,
          top_k: topK,
        });

        setResults(prev => prev.map((r, idx) =>
          idx === i ? {
            ...r,
            answer: response.answer || "",
            chunks: response.chunks || [],
            chunksCount: response.chunks?.length || 0,
            retrievalTimeMs: response.retrieval_time_ms,
            generationTimeMs: response.generation_time_ms,
            totalTimeMs: response.total_time_ms,
            status: "completed",
            retrievalStats: response.retrieval_stats,
            entitiesFound: response.entities_found || response.retrieval_stats?.entities_used,
            compressionStats: response.compression_stats,
          } : r
        ));
      } catch (error: any) {
        setResults(prev => prev.map((r, idx) =>
          idx === i ? { ...r, error: error.message || "Failed", status: "error" } : r
        ));
      }
    }

    setIsRunning(false);
    setProgress({ current: runQueue.length, total: runQueue.length, label: "Complete!" });
  };

  // Export report
  const exportReport = () => {
    const report: BenchmarkReport = {
      timestamp: new Date().toISOString(),
      config: {
        modes: selectedModes,
        topK,
        queries: queries.filter(q => q.trim()),
      },
      results: completedResults,
      statistics: statistics!,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `benchmark-report-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);

    toast({ title: "Exported!", description: "Report saved as JSON" });
  };

  // Copy report to clipboard
  const copyReport = async () => {
    const report: BenchmarkReport = {
      timestamp: new Date().toISOString(),
      config: {
        modes: selectedModes,
        topK,
        queries: queries.filter(q => q.trim()),
      },
      results: completedResults,
      statistics: statistics!,
    };

    await navigator.clipboard.writeText(JSON.stringify(report, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Load query set
  const loadQuerySet = (setKey: string) => {
    const qs = QUERY_SETS[setKey];
    if (qs) {
      setQueries(qs.queries);
    }
  };

  // ==================== RENDER ====================

  return (
    <TooltipProvider>
      <div className="space-y-6 animate-fade-in pb-20 md:pb-0">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Advanced Benchmark</h1>
            <p className="text-muted-foreground">
              Multi-query, multi-mode comparison across retrieval strategies
            </p>
          </div>
          <BarChart3 className="w-8 h-8 text-muted-foreground" />
        </div>

        {/* Main Tabs */}
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="setup" disabled={isRunning}>
              <FileText className="w-4 h-4 mr-2" />
              Setup
            </TabsTrigger>
            <TabsTrigger value="results">
              <Activity className="w-4 h-4 mr-2" />
              Results
            </TabsTrigger>
            <TabsTrigger value="report" disabled={completedResults.length === 0}>
              <TrendingUp className="w-4 h-4 mr-2" />
              Report
            </TabsTrigger>
          </TabsList>

          {/* ==================== SETUP TAB ==================== */}
          <TabsContent value="setup" className="space-y-6">
            {/* Query Input */}
            <Card className="p-6">
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <FileText className="w-4 h-4" />
                Test Queries
              </h3>

              {/* Query Set Buttons */}
              <div className="flex flex-wrap gap-2 mb-4">
                <span className="text-sm text-muted-foreground py-1">Load set:</span>
                {Object.entries(QUERY_SETS).map(([key, set]) => (
                  <Button
                    key={key}
                    variant="outline"
                    size="sm"
                    onClick={() => loadQuerySet(key)}
                    className="text-xs"
                  >
                    {set.name}
                  </Button>
                ))}
              </div>

              {/* Query List */}
              <div className="space-y-2">
                {queries.map((query, idx) => (
                  <div key={idx} className="flex gap-2">
                    <span className="text-sm text-muted-foreground w-6 pt-2">
                      {idx + 1}.
                    </span>
                    <Input
                      value={query}
                      onChange={(e) => {
                        const newQueries = [...queries];
                        newQueries[idx] = e.target.value;
                        setQueries(newQueries);
                      }}
                      placeholder="Enter query..."
                      className="flex-1"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => setQueries(queries.filter((_, i) => i !== idx))}
                      disabled={queries.length === 1}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                ))}
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={() => setQueries([...queries, ""])}
                className="mt-2"
              >
                <Plus className="w-4 h-4 mr-1" />
                Add Query
              </Button>
            </Card>

            {/* Mode Selection */}
            <Card className="p-6">
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Layers className="w-4 h-4" />
                Retrieval Modes
              </h3>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {ALL_MODES.map((mode) => (
                  <label
                    key={mode}
                    className={`flex items-center gap-2 p-3 rounded-lg border cursor-pointer transition-colors ${
                      selectedModes.includes(mode) ? "bg-secondary border-primary" : "hover:bg-secondary/50"
                    }`}
                  >
                    <Checkbox
                      checked={selectedModes.includes(mode)}
                      onCheckedChange={(checked) => {
                        if (checked) {
                          setSelectedModes([...selectedModes, mode]);
                        } else {
                          setSelectedModes(selectedModes.filter(m => m !== mode));
                        }
                      }}
                    />
                    <div className={`w-3 h-3 rounded-full ${MODE_COLORS[mode]}`} />
                    <span className="text-sm capitalize">{mode.replace('_', ' ')}</span>
                  </label>
                ))}
              </div>

              <div className="flex gap-2 mt-3">
                <Button variant="outline" size="sm" onClick={() => setSelectedModes(ALL_MODES)}>
                  Select All
                </Button>
                <Button variant="outline" size="sm" onClick={() => setSelectedModes([])}>
                  Clear All
                </Button>
              </div>
            </Card>

            {/* Options */}
            <Card className="p-6">
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Options
              </h3>

              {/* Top-K */}
              <div className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg">
                <div>
                  <div className="font-medium flex items-center gap-2">
                    <Layers className="w-4 h-4" />
                    Top-K Chunks
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Number of chunks to retrieve per query
                  </p>
                </div>
                <Select value={topK.toString()} onValueChange={(v) => setTopK(parseInt(v))}>
                  <SelectTrigger className="w-24">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {[3, 5, 10, 15, 20].map((k) => (
                      <SelectItem key={k} value={k.toString()}>K = {k}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Summary */}
              <div className="mt-6 p-4 bg-primary/5 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Total Runs</div>
                    <p className="text-sm text-muted-foreground">
                      {queries.filter(q => q.trim()).length} queries × {selectedModes.length} modes
                    </p>
                  </div>
                  <div className="text-3xl font-bold text-primary">{totalRuns}</div>
                </div>
              </div>

              {/* Run Button */}
              <Button
                onClick={runBenchmark}
                disabled={isRunning || totalRuns === 0}
                className="w-full mt-4 h-12"
                size="lg"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Run Benchmark ({totalRuns} runs)
                  </>
                )}
              </Button>
            </Card>
          </TabsContent>

          {/* ==================== RESULTS TAB ==================== */}
          <TabsContent value="results" className="space-y-4">
            {/* Progress */}
            {isRunning && (
              <Card className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">{progress.label}</span>
                  <span className="text-sm text-muted-foreground">
                    {progress.current}/{progress.total}
                  </span>
                </div>
                <Progress value={(progress.current / progress.total) * 100} className="h-2" />
              </Card>
            )}

            {/* Quick Stats */}
            {completedResults.length > 0 && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Completed</div>
                  <div className="text-2xl font-bold text-green-500">
                    {completedResults.length}/{totalRuns}
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Avg Time</div>
                  <div className="text-2xl font-bold">
                    {statistics ? (statistics.overall.avgTotalTime / 1000).toFixed(2) : '0'}s
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Fastest</div>
                  <div className="text-2xl font-bold text-blue-500">
                    {statistics ? (statistics.overall.minTime / 1000).toFixed(2) : '0'}s
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Avg Chunks</div>
                  <div className="text-2xl font-bold">
                    {statistics ? statistics.overall.avgChunks.toFixed(1) : '0'}
                  </div>
                </Card>
              </div>
            )}

            {/* Results Grid */}
            <ScrollArea className="h-[500px]">
              <div className="space-y-2">
                {results.map((result, idx) => (
                  <Card
                    key={idx}
                    className={`p-3 cursor-pointer transition-colors hover:bg-secondary/30 ${
                      result.status === "running" ? "ring-2 ring-primary" : ""
                    }`}
                    onClick={() => result.status === "completed" && setChunksModal({ open: true, result })}
                  >
                    <div className="flex items-center gap-3">
                      {/* Status */}
                      <div className="w-8">
                        {result.status === "pending" && <Clock className="w-4 h-4 text-muted-foreground" />}
                        {result.status === "running" && <Loader2 className="w-4 h-4 animate-spin text-primary" />}
                        {result.status === "completed" && <CheckCircle2 className="w-4 h-4 text-green-500" />}
                        {result.status === "error" && <XCircle className="w-4 h-4 text-red-500" />}
                      </div>

                      {/* Mode Badge */}
                      <Badge className={`${MODE_COLORS[result.mode]} text-white min-w-20 justify-center`}>
                        {result.mode}
                      </Badge>

                      {/* Query */}
                      <span className="flex-1 text-sm truncate text-muted-foreground">
                        Q{result.queryIndex + 1}: {result.query.substring(0, 40)}...
                      </span>

                      {/* Metrics */}
                      {result.status === "completed" && (
                        <div className="flex items-center gap-4 text-xs">
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {(result.totalTimeMs / 1000).toFixed(2)}s
                          </span>
                          <span className="flex items-center gap-1">
                            <Layers className="w-3 h-3" />
                            {result.chunksCount}
                          </span>
                          {result.compressionStats?.enabled && (
                            <span className="flex items-center gap-1 text-green-500">
                              <Minimize2 className="w-3 h-3" />
                              {(result.compressionStats.compression_ratio * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </Card>
                ))}
              </div>
            </ScrollArea>

            {results.length === 0 && (
              <Card className="p-12 text-center">
                <Activity className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-xl font-semibold mb-2">No results yet</h3>
                <p className="text-muted-foreground">
                  Configure your benchmark in the Setup tab and run it
                </p>
              </Card>
            )}
          </TabsContent>

          {/* ==================== REPORT TAB ==================== */}
          <TabsContent value="report" className="space-y-6">
            {statistics && (
              <>
                {/* Export Buttons */}
                <div className="flex justify-end gap-2">
                  <Button variant="outline" onClick={copyReport}>
                    {copied ? <Check className="w-4 h-4 mr-2" /> : <Copy className="w-4 h-4 mr-2" />}
                    {copied ? "Copied!" : "Copy JSON"}
                  </Button>
                  <Button onClick={exportReport}>
                    <Download className="w-4 h-4 mr-2" />
                    Export JSON
                  </Button>
                </div>

                {/* Mode Comparison Chart */}
                <Card className="p-6">
                  <h3 className="font-semibold mb-4 flex items-center gap-2">
                    <BarChart3 className="w-4 h-4" />
                    Mode Performance Comparison
                  </h3>

                  {/* Simple bar chart */}
                  <div className="space-y-3">
                    {selectedModes.map((mode) => {
                      const modeStats = statistics.byMode[mode];
                      if (!modeStats) return null;
                      const maxTime = Math.max(...Object.values(statistics.byMode).map(s => s.avgTotalTime));
                      const barWidth = (modeStats.avgTotalTime / maxTime) * 100;

                      return (
                        <div key={mode} className="space-y-1">
                          <div className="flex justify-between text-sm">
                            <div className="flex items-center gap-2">
                              <div className={`w-3 h-3 rounded-full ${MODE_COLORS[mode]}`} />
                              <span className="capitalize">{mode.replace('_', ' ')}</span>
                            </div>
                            <span className="font-mono">
                              {(modeStats.avgTotalTime / 1000).toFixed(2)}s avg | {modeStats.avgChunks.toFixed(1)} chunks
                            </span>
                          </div>
                          <div className="h-6 bg-secondary rounded-full overflow-hidden">
                            <div
                              className={`h-full ${MODE_COLORS[mode]} transition-all duration-500`}
                              style={{ width: `${barWidth}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </Card>

                {/* Detailed Stats Table */}
                <Card className="p-6">
                  <h3 className="font-semibold mb-4 flex items-center gap-2">
                    <FileJson className="w-4 h-4" />
                    Detailed Statistics
                  </h3>

                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left p-2">Mode</th>
                          <th className="text-right p-2">Avg Total</th>
                          <th className="text-right p-2">Avg Retrieval</th>
                          <th className="text-right p-2">Avg Generation</th>
                          <th className="text-right p-2">Avg Chunks</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedModes.map((mode) => {
                          const stats = statistics.byMode[mode];
                          if (!stats) return null;
                          return (
                            <tr key={mode} className="border-b hover:bg-secondary/30">
                              <td className="p-2">
                                <div className="flex items-center gap-2">
                                  <div className={`w-2 h-2 rounded-full ${MODE_COLORS[mode]}`} />
                                  <span className="capitalize">{mode}</span>
                                </div>
                              </td>
                              <td className="text-right p-2 font-mono">
                                {(stats.avgTotalTime / 1000).toFixed(2)}s
                              </td>
                              <td className="text-right p-2 font-mono">
                                {stats.avgRetrievalTime.toFixed(0)}ms
                              </td>
                              <td className="text-right p-2 font-mono">
                                {stats.avgGenerationTime.toFixed(0)}ms
                              </td>
                              <td className="text-right p-2 font-mono">
                                {stats.avgChunks.toFixed(1)}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>

                {/* Time Distribution */}
                <Card className="p-6">
                  <h3 className="font-semibold mb-4 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4" />
                    Time Distribution
                  </h3>

                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 bg-secondary/30 rounded-lg text-center">
                      <div className="text-sm text-muted-foreground">Min</div>
                      <div className="text-2xl font-bold text-blue-500">
                        {(statistics.overall.minTime / 1000).toFixed(2)}s
                      </div>
                    </div>
                    <div className="p-4 bg-secondary/30 rounded-lg text-center">
                      <div className="text-sm text-muted-foreground">Average</div>
                      <div className="text-2xl font-bold">
                        {(statistics.overall.avgTotalTime / 1000).toFixed(2)}s
                      </div>
                    </div>
                    <div className="p-4 bg-secondary/30 rounded-lg text-center">
                      <div className="text-sm text-muted-foreground">Max</div>
                      <div className="text-2xl font-bold text-red-500">
                        {(statistics.overall.maxTime / 1000).toFixed(2)}s
                      </div>
                    </div>
                  </div>
                </Card>
              </>
            )}
          </TabsContent>
        </Tabs>

        {/* Result Detail Modal */}
        <Dialog open={chunksModal.open} onOpenChange={(open) => setChunksModal({ open, result: null })}>
          <DialogContent className="max-w-4xl max-h-[80vh]">
            {chunksModal.result && (
              <>
                <DialogHeader>
                  <DialogTitle className="flex items-center gap-2">
                    <Badge className={`${MODE_COLORS[chunksModal.result.mode]} text-white`}>
                      {chunksModal.result.mode}
                    </Badge>
                    <span className="text-sm font-normal text-muted-foreground">
                      Q{chunksModal.result.queryIndex + 1}
                    </span>
                  </DialogTitle>
                  <DialogDescription>
                    {chunksModal.result.query}
                  </DialogDescription>
                </DialogHeader>

                {/* Metrics */}
                <div className="grid grid-cols-4 gap-2 text-xs">
                  <div className="bg-secondary/30 rounded p-2">
                    <div className="text-muted-foreground">Total</div>
                    <div className="font-bold">{(chunksModal.result.totalTimeMs / 1000).toFixed(2)}s</div>
                  </div>
                  <div className="bg-secondary/30 rounded p-2">
                    <div className="text-muted-foreground">Retrieval</div>
                    <div className="font-bold">{chunksModal.result.retrievalTimeMs}ms</div>
                  </div>
                  <div className="bg-secondary/30 rounded p-2">
                    <div className="text-muted-foreground">Generation</div>
                    <div className="font-bold">{chunksModal.result.generationTimeMs}ms</div>
                  </div>
                  <div className="bg-secondary/30 rounded p-2">
                    <div className="text-muted-foreground">Chunks</div>
                    <div className="font-bold">{chunksModal.result.chunksCount}</div>
                  </div>
                </div>

                {/* Answer */}
                <div className="mt-4">
                  <h4 className="font-medium mb-2">Answer</h4>
                  <ScrollArea className="h-32 bg-secondary/20 rounded-lg p-3">
                    <p className="text-sm whitespace-pre-wrap">{chunksModal.result.answer}</p>
                  </ScrollArea>
                </div>

                {/* Chunks */}
                <div className="mt-4">
                  <h4 className="font-medium mb-2">Retrieved Chunks ({chunksModal.result.chunks.length})</h4>
                  <ScrollArea className="h-64">
                    <div className="space-y-2">
                      {chunksModal.result.chunks.map((chunk, idx) => (
                        <Card key={idx} className="p-3">
                          <div className="flex items-center gap-2 mb-2">
                            <Badge variant="outline" className="text-xs">#{idx + 1}</Badge>
                            <span className="text-xs text-muted-foreground">
                              Score: {chunk.score?.toFixed(3)}
                            </span>
                            {chunk.via_entity && (
                              <Badge variant="secondary" className="text-xs bg-green-500/10">
                                <Target className="w-3 h-3 mr-1" />
                                {chunk.via_entity}
                              </Badge>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground line-clamp-3">
                            {chunk.text}
                          </p>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              </>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </TooltipProvider>
  );
}
