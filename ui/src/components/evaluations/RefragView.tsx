import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Play,
  Loader2,
  Clock,
  Zap,
  Minimize2,
  Maximize2,
  BarChart3,
  CheckCircle2,
  ArrowRight,
  Sparkles,
  FileText,
  TrendingDown,
  TrendingUp,
  Percent,
  Timer,
  Activity,
} from "lucide-react";
import { chatApi, RetrievalMode, refragApi } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

interface RefragTestResult {
  withRefrag: {
    answer: string;
    totalTimeMs: number;
    retrievalTimeMs: number;
    generationTimeMs: number;
    chunksCount: number;
    chunks: any[];
    compressionStats?: {
      enabled: boolean;
      original_length: number;
      compressed_length: number;
      compression_ratio: number;
      chunks_compressed: number;
      strategy: string;
      compression_time_ms: number;
    };
    originalContext?: string;
    compressedContext?: string;
  };
  withoutRefrag: {
    answer: string;
    totalTimeMs: number;
    retrievalTimeMs: number;
    generationTimeMs: number;
    chunksCount: number;
    chunks: any[];
    context?: string;
  };
  speedup: number;
  tokenSavings: number;
}

const TEST_QUERIES = [
  "ما هي جائزة الحكومة الرقمية؟",
  "What is the Digital Government Forum?",
  "من هم المشاركون في منتدى الحكومة الرقمية؟",
  "Explain the relationship between SDAIA and digital transformation",
];

export function RefragView() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [mode, setMode] = useState<RetrievalMode>("local");
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<RefragTestResult | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentPhase, setCurrentPhase] = useState("");
  const [history, setHistory] = useState<Array<{
    query: string;
    speedup: number;
    tokenSavings: number;
    timestamp: Date;
  }>>([]);
  const { toast } = useToast();

  const runBenchmark = async () => {
    if (!query.trim()) {
      toast({
        title: "Error",
        description: "Please enter a query to benchmark",
        variant: "destructive",
      });
      return;
    }

    setIsRunning(true);
    setProgress(0);
    setResults(null);

    try {
      // Phase 1: Run without REFRAG
      setCurrentPhase("Running without REFRAG...");
      setProgress(25);

      const responseWithout = await chatApi.ask({
        message: query,
        retrieval_mode: mode,
        top_k: topK,
        use_refrag: false,
      });

      setProgress(50);

      // Phase 2: Run with REFRAG
      setCurrentPhase("Running with REFRAG compression...");

      const responseWith = await chatApi.ask({
        message: query,
        retrieval_mode: mode,
        top_k: topK,
        use_refrag: true,
      });

      setProgress(100);

      // Calculate metrics
      const compressionStats = responseWith.compression_stats || responseWith.metadata?.compression_stats;
      const speedup = responseWithout.generation_time_ms > 0 && responseWith.generation_time_ms > 0
        ? responseWithout.generation_time_ms / responseWith.generation_time_ms
        : responseWithout.total_time_ms / responseWith.total_time_ms;
      const originalLength = compressionStats?.original_length || 0;
      const compressedLength = compressionStats?.compressed_length || 0;
      const tokenSavings = originalLength > 0
        ? ((originalLength - compressedLength) / originalLength) * 100
        : 0;

      const result: RefragTestResult = {
        withRefrag: {
          answer: responseWith.answer,
          totalTimeMs: responseWith.total_time_ms,
          retrievalTimeMs: responseWith.retrieval_time_ms,
          generationTimeMs: responseWith.generation_time_ms,
          chunksCount: responseWith.chunks?.length || 0,
          chunks: responseWith.chunks || [],
          compressionStats: compressionStats,
        },
        withoutRefrag: {
          answer: responseWithout.answer,
          totalTimeMs: responseWithout.total_time_ms,
          retrievalTimeMs: responseWithout.retrieval_time_ms,
          generationTimeMs: responseWithout.generation_time_ms,
          chunksCount: responseWithout.chunks?.length || 0,
          chunks: responseWithout.chunks || [],
        },
        speedup,
        tokenSavings,
      };

      setResults(result);

      // Add to history
      setHistory(prev => [
        { query, speedup, tokenSavings, timestamp: new Date() },
        ...prev.slice(0, 9), // Keep last 10
      ]);

    } catch (error: any) {
      toast({
        title: "Benchmark Failed",
        description: error.message || "Failed to run benchmark",
        variant: "destructive",
      });
    } finally {
      setIsRunning(false);
      setCurrentPhase("");
    }
  };

  // Calculate average stats from history
  const avgSpeedup = history.length > 0
    ? history.reduce((sum, h) => sum + h.speedup, 0) / history.length
    : 0;
  const avgTokenSavings = history.length > 0
    ? history.reduce((sum, h) => sum + h.tokenSavings, 0) / history.length
    : 0;

  return (
    <div className="space-y-6 animate-fade-in pb-20 md:pb-0">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">REFRAG Benchmark</h1>
          <p className="text-muted-foreground">
            Compare performance with and without REFRAG compression
          </p>
        </div>
        <Minimize2 className="w-8 h-8 text-muted-foreground" />
      </div>

      {/* Query Input */}
      <Card className="p-6">
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <Input
                placeholder="Enter a query to test REFRAG compression..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="h-12 text-lg"
                disabled={isRunning}
              />
            </div>
            <div className="flex gap-2">
              <Select
                value={mode}
                onValueChange={(v) => setMode(v as RetrievalMode)}
                disabled={isRunning}
              >
                <SelectTrigger className="w-32 h-12">
                  <SelectValue placeholder="Mode" />
                </SelectTrigger>
                <SelectContent>
                  {["naive", "local", "global", "hybrid", "mix"].map((m) => (
                    <SelectItem key={m} value={m}>
                      {m.charAt(0).toUpperCase() + m.slice(1)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select
                value={topK.toString()}
                onValueChange={(v) => setTopK(parseInt(v))}
                disabled={isRunning}
              >
                <SelectTrigger className="w-24 h-12">
                  <SelectValue placeholder="Top-K" />
                </SelectTrigger>
                <SelectContent>
                  {[3, 5, 10, 15, 20].map((k) => (
                    <SelectItem key={k} value={k.toString()}>
                      Top {k}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                onClick={runBenchmark}
                disabled={isRunning || !query.trim()}
                className="h-12 px-6"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Testing...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Run Test
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Quick Query Buttons */}
          <div className="flex flex-wrap gap-2">
            <span className="text-sm text-muted-foreground py-1">Quick tests:</span>
            {TEST_QUERIES.map((q, idx) => (
              <Button
                key={idx}
                variant="outline"
                size="sm"
                onClick={() => setQuery(q)}
                disabled={isRunning}
                className="text-xs"
              >
                {q.substring(0, 30)}...
              </Button>
            ))}
          </div>

          {/* Progress Bar */}
          {isRunning && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">{currentPhase}</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          )}
        </div>
      </Card>

      {/* Summary Stats */}
      {history.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-500/10 rounded-lg">
                <Activity className="w-5 h-5 text-green-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Tests Run</p>
                <p className="text-xl font-bold">{history.length}</p>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <Zap className="w-5 h-5 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Avg Speedup</p>
                <p className="text-xl font-bold">
                  {avgSpeedup.toFixed(2)}x
                  {avgSpeedup > 1 && <TrendingUp className="w-4 h-4 inline ml-1 text-green-500" />}
                </p>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-500/10 rounded-lg">
                <Percent className="w-5 h-5 text-purple-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Avg Token Savings</p>
                <p className="text-xl font-bold">
                  {avgTokenSavings.toFixed(1)}%
                  <TrendingDown className="w-4 h-4 inline ml-1 text-green-500" />
                </p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Results Comparison */}
      {results && (
        <div className="space-y-4">
          {/* Key Metrics Banner */}
          <Card className="p-6 bg-gradient-to-r from-green-500/10 to-blue-500/10">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600">
                  {results.speedup.toFixed(2)}x
                </div>
                <div className="text-sm text-muted-foreground">Speedup</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">
                  {results.tokenSavings.toFixed(1)}%
                </div>
                <div className="text-sm text-muted-foreground">Token Savings</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600">
                  {results.withRefrag.compressionStats?.compression_ratio?.toFixed(2) || "N/A"}
                </div>
                <div className="text-sm text-muted-foreground">Compression Ratio</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-orange-600">
                  {results.withRefrag.compressionStats?.compression_time_ms?.toFixed(0) || 0}ms
                </div>
                <div className="text-sm text-muted-foreground">Compression Time</div>
              </div>
            </div>
          </Card>

          {/* Context Compression Visualization */}
          <Card className="p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Context Compression (What Gets Sent to LLM)
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Original Context */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-red-600">Original Context</span>
                  <Badge variant="outline" className="text-xs">
                    {results.withRefrag.compressionStats?.original_length?.toLocaleString() ||
                     results.withoutRefrag.chunks.reduce((acc, c) => acc + (c.text?.length || 0), 0).toLocaleString()} chars
                  </Badge>
                </div>
                <ScrollArea className="h-64 border rounded-lg p-3 bg-red-500/5">
                  <div className="text-xs font-mono whitespace-pre-wrap">
                    {results.withoutRefrag.chunks.map((chunk, idx) => (
                      <div key={idx} className="mb-3 pb-3 border-b border-dashed last:border-0">
                        <div className="text-red-600 font-bold mb-1">[Chunk {idx + 1}] Score: {chunk.score?.toFixed(3)}</div>
                        <p>{chunk.text}</p>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>

              {/* Compressed Context */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-green-600">Compressed Context</span>
                  <Badge variant="outline" className="text-xs bg-green-500/10">
                    {results.withRefrag.compressionStats?.compressed_length?.toLocaleString() || '?'} chars
                    <span className="ml-1 text-green-600">
                      ({Math.round((1 - (results.withRefrag.compressionStats?.compression_ratio || 1)) * 100)}% saved)
                    </span>
                  </Badge>
                </div>
                <ScrollArea className="h-64 border rounded-lg p-3 bg-green-500/5">
                  <div className="text-xs font-mono whitespace-pre-wrap">
                    {results.withRefrag.chunks.map((chunk, idx) => (
                      <div key={idx} className="mb-3 pb-3 border-b border-dashed last:border-0">
                        <div className="text-green-600 font-bold mb-1">[Chunk {idx + 1}] Score: {chunk.score?.toFixed(3)}</div>
                        <p>{chunk.text}</p>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            </div>

            {/* Explanation */}
            <div className="mt-4 p-3 bg-blue-500/10 rounded-lg text-xs">
              <div className="font-medium text-blue-700 dark:text-blue-400 mb-1">How MIRAGE REFRAG Works:</div>
              <p className="text-muted-foreground">
                MIRAGE uses <strong>context compression</strong> - after retrieving chunks, it compresses the combined context
                before sending to the LLM. This reduces LLM input tokens and speeds up generation.
                The chunks retrieved are the same, but the text sent to the LLM is shorter.
              </p>
            </div>
          </Card>

          {/* Side-by-Side Timing Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Without REFRAG */}
            <Card className="p-4">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-2 bg-red-500/10 rounded-lg">
                  <Maximize2 className="w-5 h-5 text-red-500" />
                </div>
                <div>
                  <h3 className="font-semibold">Without REFRAG</h3>
                  <p className="text-xs text-muted-foreground">Full context sent to LLM</p>
                </div>
              </div>

              <div className="space-y-3">
                {/* Timing */}
                <div className="grid grid-cols-3 gap-2">
                  <div className="bg-secondary/30 rounded-lg p-2 text-center">
                    <div className="text-lg font-bold">{results.withoutRefrag.retrievalTimeMs.toFixed(0)}ms</div>
                    <div className="text-xs text-muted-foreground">Retrieval</div>
                  </div>
                  <div className="bg-red-500/10 rounded-lg p-2 text-center">
                    <div className="text-lg font-bold text-red-600">{results.withoutRefrag.generationTimeMs.toFixed(0)}ms</div>
                    <div className="text-xs text-muted-foreground">Generation</div>
                  </div>
                  <div className="bg-secondary/30 rounded-lg p-2 text-center">
                    <div className="text-lg font-bold">{(results.withoutRefrag.totalTimeMs / 1000).toFixed(2)}s</div>
                    <div className="text-xs text-muted-foreground">Total</div>
                  </div>
                </div>

                {/* Answer */}
                <div>
                  <div className="text-xs text-muted-foreground mb-1">Answer:</div>
                  <ScrollArea className="h-40">
                    <div className="bg-secondary/20 rounded-lg p-3">
                      <p className="text-sm whitespace-pre-wrap">{results.withoutRefrag.answer}</p>
                    </div>
                  </ScrollArea>
                </div>
              </div>
            </Card>

            {/* With REFRAG */}
            <Card className="p-4 ring-2 ring-green-500/30">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-2 bg-green-500/10 rounded-lg">
                  <Minimize2 className="w-5 h-5 text-green-500" />
                </div>
                <div>
                  <h3 className="font-semibold">With REFRAG</h3>
                  <p className="text-xs text-muted-foreground">Compressed context sent to LLM</p>
                </div>
                <Badge className="ml-auto bg-green-500">
                  {results.speedup > 1 ? `${results.speedup.toFixed(1)}x faster` : 'Similar speed'}
                </Badge>
              </div>

              <div className="space-y-3">
                {/* Timing with compression */}
                <div className="grid grid-cols-4 gap-2">
                  <div className="bg-secondary/30 rounded-lg p-2 text-center">
                    <div className="text-lg font-bold">{results.withRefrag.retrievalTimeMs.toFixed(0)}ms</div>
                    <div className="text-xs text-muted-foreground">Retrieval</div>
                  </div>
                  <div className="bg-yellow-500/10 rounded-lg p-2 text-center">
                    <div className="text-lg font-bold text-yellow-600">{results.withRefrag.compressionStats?.compression_time_ms?.toFixed(0) || 0}ms</div>
                    <div className="text-xs text-muted-foreground">Compress</div>
                  </div>
                  <div className="bg-green-500/10 rounded-lg p-2 text-center">
                    <div className="text-lg font-bold text-green-600">{results.withRefrag.generationTimeMs.toFixed(0)}ms</div>
                    <div className="text-xs text-muted-foreground">Generate</div>
                  </div>
                  <div className="bg-green-500/10 rounded-lg p-2 text-center">
                    <div className="text-lg font-bold text-green-600">{(results.withRefrag.totalTimeMs / 1000).toFixed(2)}s</div>
                    <div className="text-xs text-muted-foreground">Total</div>
                  </div>
                </div>

                {/* Compression Stats */}
                {results.withRefrag.compressionStats && (
                  <div className="bg-green-500/10 rounded-lg p-2 flex items-center justify-between text-xs">
                    <span className="text-green-700 dark:text-green-400">
                      <Sparkles className="w-3 h-3 inline mr-1" />
                      {results.withRefrag.compressionStats.strategy} compression
                    </span>
                    <span className="font-bold text-green-600">
                      {results.withRefrag.compressionStats.original_length.toLocaleString()} → {results.withRefrag.compressionStats.compressed_length.toLocaleString()} chars
                    </span>
                  </div>
                )}

                {/* Answer */}
                <div>
                  <div className="text-xs text-muted-foreground mb-1">Answer:</div>
                  <ScrollArea className="h-40">
                    <div className="bg-secondary/20 rounded-lg p-3">
                      <p className="text-sm whitespace-pre-wrap">{results.withRefrag.answer}</p>
                    </div>
                  </ScrollArea>
                </div>
              </div>
            </Card>
          </div>

          {/* Timing Comparison Visual */}
          <Card className="p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Timer className="w-4 h-4" />
              Timing Comparison
            </h3>
            <div className="space-y-4">
              {/* Without REFRAG bar */}
              <div className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span>Without REFRAG</span>
                  <span>{(results.withoutRefrag.totalTimeMs / 1000).toFixed(2)}s</span>
                </div>
                <div className="h-8 bg-secondary rounded-lg overflow-hidden flex">
                  <div
                    className="bg-blue-500 h-full flex items-center justify-center text-xs text-white"
                    style={{ width: `${(results.withoutRefrag.retrievalTimeMs / results.withoutRefrag.totalTimeMs) * 100}%` }}
                  >
                    Retrieval
                  </div>
                  <div
                    className="bg-purple-500 h-full flex items-center justify-center text-xs text-white"
                    style={{ width: `${(results.withoutRefrag.generationTimeMs / results.withoutRefrag.totalTimeMs) * 100}%` }}
                  >
                    Generation
                  </div>
                </div>
              </div>

              {/* With REFRAG bar */}
              <div className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-green-600 font-medium">With REFRAG</span>
                  <span className="text-green-600 font-medium">{(results.withRefrag.totalTimeMs / 1000).toFixed(2)}s</span>
                </div>
                <div className="h-8 bg-secondary rounded-lg overflow-hidden flex">
                  <div
                    className="bg-blue-500 h-full flex items-center justify-center text-xs text-white"
                    style={{ width: `${(results.withRefrag.retrievalTimeMs / results.withoutRefrag.totalTimeMs) * 100}%` }}
                  >
                    Retrieval
                  </div>
                  <div
                    className="bg-yellow-500 h-full flex items-center justify-center text-xs text-white"
                    style={{ width: `${((results.withRefrag.compressionStats?.compression_time_ms || 0) / results.withoutRefrag.totalTimeMs) * 100}%` }}
                  >
                    Compress
                  </div>
                  <div
                    className="bg-green-500 h-full flex items-center justify-center text-xs text-white"
                    style={{ width: `${(results.withRefrag.generationTimeMs / results.withoutRefrag.totalTimeMs) * 100}%` }}
                  >
                    Generation
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground pt-2">
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-blue-500" />
                  Retrieval
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-yellow-500" />
                  Compression
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-purple-500" />
                  Generation (No REFRAG)
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded bg-green-500" />
                  Generation (REFRAG)
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <Card className="p-4">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Test History
          </h3>
          <ScrollArea className="h-48">
            <div className="space-y-2">
              {history.map((h, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 bg-secondary/20 rounded-lg text-sm"
                >
                  <div className="flex-1 truncate mr-4">
                    {h.query}
                  </div>
                  <div className="flex items-center gap-4 text-xs">
                    <Badge variant="outline" className="gap-1">
                      <Zap className="w-3 h-3" />
                      {h.speedup.toFixed(2)}x
                    </Badge>
                    <Badge variant="outline" className="gap-1">
                      <TrendingDown className="w-3 h-3" />
                      {h.tokenSavings.toFixed(1)}%
                    </Badge>
                    <span className="text-muted-foreground">
                      {h.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </Card>
      )}

      {/* Empty State */}
      {!results && history.length === 0 && (
        <Card className="p-12 text-center">
          <Minimize2 className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-xl font-semibold mb-2">No benchmark results yet</h3>
          <p className="text-muted-foreground max-w-md mx-auto mb-4">
            REFRAG compresses retrieved context before sending to the LLM,
            reducing generation time and token usage while maintaining answer quality.
          </p>
          <div className="flex flex-wrap justify-center gap-4 text-sm">
            <div className="flex items-center gap-2 bg-green-500/10 rounded-lg px-3 py-2">
              <Zap className="w-4 h-4 text-green-500" />
              <span>Faster responses</span>
            </div>
            <div className="flex items-center gap-2 bg-blue-500/10 rounded-lg px-3 py-2">
              <TrendingDown className="w-4 h-4 text-blue-500" />
              <span>Lower token cost</span>
            </div>
            <div className="flex items-center gap-2 bg-purple-500/10 rounded-lg px-3 py-2">
              <CheckCircle2 className="w-4 h-4 text-purple-500" />
              <span>Same quality</span>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
