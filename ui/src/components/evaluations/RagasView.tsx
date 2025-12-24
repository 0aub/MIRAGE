import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
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
  Target,
  CheckCircle2,
  XCircle,
  TrendingUp,
  TrendingDown,
  Zap,
  BarChart3,
  Award,
  FileText,
  Percent,
  Clock,
  AlertTriangle,
  ChevronRight,
  Sparkles,
  Minus,
} from "lucide-react";
import {
  ragasApi,
  RagasEvaluationResponse,
  RagasTestCase,
  RagasModeResult,
} from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

const ALL_MODES = ["naive", "local", "global", "hybrid", "mix"];

const DIFFICULTY_COLORS: Record<string, string> = {
  easy: "bg-green-500/20 text-green-700 dark:text-green-400",
  medium: "bg-yellow-500/20 text-yellow-700 dark:text-yellow-400",
  hard: "bg-red-500/20 text-red-700 dark:text-red-400",
};

const SCORE_COLORS = {
  excellent: "text-green-600 dark:text-green-400",
  good: "text-blue-600 dark:text-blue-400",
  fair: "text-yellow-600 dark:text-yellow-400",
  poor: "text-red-600 dark:text-red-400",
};

function getScoreColor(score: number): string {
  if (score >= 0.8) return SCORE_COLORS.excellent;
  if (score >= 0.6) return SCORE_COLORS.good;
  if (score >= 0.4) return SCORE_COLORS.fair;
  return SCORE_COLORS.poor;
}

function getScoreBg(score: number): string {
  if (score >= 0.8) return "bg-green-500";
  if (score >= 0.6) return "bg-blue-500";
  if (score >= 0.4) return "bg-yellow-500";
  return "bg-red-500";
}

export function RagasView() {
  const [testCases, setTestCases] = useState<RagasTestCase[]>([]);
  const [selectedTestCases, setSelectedTestCases] = useState<string[]>([]);
  const [selectedModes, setSelectedModes] = useState<string[]>(ALL_MODES);
  const [compareRefrag, setCompareRefrag] = useState(true);
  const [topK, setTopK] = useState(5);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentPhase, setCurrentPhase] = useState("");
  const [results, setResults] = useState<RagasEvaluationResponse | null>(null);
  const { toast } = useToast();

  // Load test cases on mount
  useEffect(() => {
    loadTestCases();
  }, []);

  const loadTestCases = async () => {
    try {
      const response = await ragasApi.getTestCases();
      setTestCases(response.test_cases);
    } catch (error: any) {
      toast({
        title: "Failed to load test cases",
        description: error.message || "Could not load RAGAS test cases",
        variant: "destructive",
      });
    }
  };

  const runEvaluation = async () => {
    if (selectedModes.length === 0) {
      toast({
        title: "No modes selected",
        description: "Please select at least one retrieval mode to test",
        variant: "destructive",
      });
      return;
    }

    setIsRunning(true);
    setProgress(0);
    setResults(null);

    try {
      setCurrentPhase("Starting RAGAS evaluation...");
      setProgress(10);

      // Simulate progress during long-running evaluation
      const progressInterval = setInterval(() => {
        setProgress((p) => Math.min(p + 5, 90));
      }, 2000);

      const response = await ragasApi.runEvaluation({
        test_case_ids: selectedTestCases.length > 0 ? selectedTestCases : undefined,
        modes: selectedModes,
        compare_refrag: compareRefrag,
        top_k: topK,
      });

      clearInterval(progressInterval);
      setProgress(100);
      setResults(response);

      toast({
        title: "Evaluation Complete",
        description: `Tested ${response.summary.total_tests} cases across ${response.summary.modes_tested.length} modes`,
      });
    } catch (error: any) {
      toast({
        title: "Evaluation Failed",
        description: error.message || "Failed to run RAGAS evaluation",
        variant: "destructive",
      });
    } finally {
      setIsRunning(false);
      setCurrentPhase("");
    }
  };

  const toggleMode = (mode: string) => {
    setSelectedModes((prev) =>
      prev.includes(mode) ? prev.filter((m) => m !== mode) : [...prev, mode]
    );
  };

  return (
    <div className="space-y-6 animate-fade-in pb-20 md:pb-0">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">RAGAS Evaluation</h1>
          <p className="text-muted-foreground">
            Test retrieval quality using ground truth answers with semantic similarity scoring
          </p>
        </div>
        <Target className="w-8 h-8 text-muted-foreground" />
      </div>

      {/* Configuration Panel */}
      <Card className="p-6">
        <div className="space-y-6">
          {/* Mode Selection */}
          <div>
            <Label className="text-sm font-medium mb-3 block">Retrieval Modes</Label>
            <div className="flex flex-wrap gap-2">
              {ALL_MODES.map((mode) => (
                <Button
                  key={mode}
                  variant={selectedModes.includes(mode) ? "default" : "outline"}
                  size="sm"
                  onClick={() => toggleMode(mode)}
                  disabled={isRunning}
                >
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </Button>
              ))}
            </div>
          </div>

          {/* Options Row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* REFRAG Toggle */}
            <div className="flex items-center space-x-3">
              <Switch
                id="compare-refrag"
                checked={compareRefrag}
                onCheckedChange={setCompareRefrag}
                disabled={isRunning}
              />
              <Label htmlFor="compare-refrag" className="cursor-pointer">
                Compare with REFRAG
              </Label>
            </div>

            {/* Top-K Slider */}
            <div className="space-y-2">
              <Label className="text-sm">Top-K: {topK}</Label>
              <Slider
                value={[topK]}
                onValueChange={([v]) => setTopK(v)}
                min={3}
                max={20}
                step={1}
                disabled={isRunning}
              />
            </div>

            {/* Test Case Filter */}
            <div>
              <Select
                value={selectedTestCases.length > 0 ? "selected" : "all"}
                onValueChange={(v) => {
                  if (v === "all") setSelectedTestCases([]);
                }}
                disabled={isRunning}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Test cases" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Test Cases ({testCases.length})</SelectItem>
                  <SelectItem value="selected">Selected Only</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Run Button */}
          <div className="flex items-center gap-4">
            <Button
              onClick={runEvaluation}
              disabled={isRunning || selectedModes.length === 0}
              size="lg"
              className="px-8"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Run RAGAS Evaluation
                </>
              )}
            </Button>

            {isRunning && (
              <div className="flex-1">
                <div className="flex items-center justify-between text-sm mb-1">
                  <span className="text-muted-foreground">{currentPhase}</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <Progress value={progress} className="h-2" />
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Results */}
      {results && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-500/10 rounded-lg">
                  <BarChart3 className="w-5 h-5 text-blue-500" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Tests Run</p>
                  <p className="text-2xl font-bold">{results.summary.total_tests}</p>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-green-500/10 rounded-lg">
                  <Award className="w-5 h-5 text-green-500" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Best Mode</p>
                  <p className="text-2xl font-bold">{results.summary.best_overall_mode}</p>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-500/10 rounded-lg">
                  <Target className="w-5 h-5 text-purple-500" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Best Score</p>
                  <p className={`text-2xl font-bold ${getScoreColor(results.summary.best_overall_score)}`}>
                    {(results.summary.best_overall_score * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-orange-500/10 rounded-lg">
                  <Clock className="w-5 h-5 text-orange-500" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Duration</p>
                  <p className="text-2xl font-bold">{(results.duration_ms / 1000).toFixed(1)}s</p>
                </div>
              </div>
            </Card>
          </div>

          {/* Mode Comparison */}
          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Mode Performance Comparison
            </h3>
            <div className="space-y-3">
              {Object.entries(results.summary.mode_scores)
                .sort(([, a], [, b]) => b.avg_score - a.avg_score)
                .map(([mode, scores]) => (
                  <div key={mode} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <span className="font-medium w-20">{mode}</span>
                        <Badge variant="outline" className="text-xs">
                          {scores.passed}/{scores.passed + scores.failed} passed
                        </Badge>
                      </div>
                      <span className={`font-bold ${getScoreColor(scores.avg_score)}`}>
                        {(scores.avg_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-3 bg-secondary rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getScoreBg(scores.avg_score)} transition-all`}
                        style={{ width: `${scores.avg_score * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
            </div>
          </Card>

          {/* REFRAG Impact Analysis */}
          {results.refrag_impact && (
            <Card className="p-6 bg-gradient-to-r from-purple-500/5 to-blue-500/5">
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                REFRAG Compression Impact Analysis
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                <div className="text-center p-3 bg-background rounded-lg">
                  <div className="flex items-center justify-center gap-1 text-green-600">
                    <TrendingUp className="w-4 h-4" />
                    <span className="text-2xl font-bold">{results.refrag_impact.modes_improved}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Quality Improved</p>
                </div>
                <div className="text-center p-3 bg-background rounded-lg">
                  <div className="flex items-center justify-center gap-1 text-red-600">
                    <TrendingDown className="w-4 h-4" />
                    <span className="text-2xl font-bold">{results.refrag_impact.modes_hurt}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Quality Hurt</p>
                </div>
                <div className="text-center p-3 bg-background rounded-lg">
                  <div className="flex items-center justify-center gap-1 text-gray-500">
                    <Minus className="w-4 h-4" />
                    <span className="text-2xl font-bold">{results.refrag_impact.modes_unchanged}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">No Change</p>
                </div>
                <div className="text-center p-3 bg-background rounded-lg">
                  <div className={`text-2xl font-bold ${results.refrag_impact.avg_ground_truth_delta >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {results.refrag_impact.avg_ground_truth_delta >= 0 ? '+' : ''}{(results.refrag_impact.avg_ground_truth_delta * 100).toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Avg Quality Delta</p>
                </div>
                <div className="text-center p-3 bg-background rounded-lg">
                  <div className="flex items-center justify-center gap-1 text-blue-600">
                    <Zap className="w-4 h-4" />
                    <span className="text-2xl font-bold">{results.refrag_impact.avg_speed_improvement.toFixed(2)}x</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Speed Improvement</p>
                </div>
                <div className="text-center p-3 bg-background rounded-lg">
                  <div className="flex items-center justify-center gap-1 text-purple-600">
                    <Percent className="w-4 h-4" />
                    <span className="text-2xl font-bold">{results.refrag_impact.avg_token_savings.toFixed(1)}%</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Token Savings</p>
                </div>
              </div>

              {/* Insight */}
              <div className="mt-4 p-3 bg-blue-500/10 rounded-lg text-sm">
                <span className="font-medium text-blue-700 dark:text-blue-400">Insight: </span>
                {results.refrag_impact.avg_ground_truth_delta >= 0 ? (
                  <span className="text-muted-foreground">
                    REFRAG compression maintains or improves answer quality while providing {results.refrag_impact.avg_speed_improvement.toFixed(1)}x speedup and {results.refrag_impact.avg_token_savings.toFixed(0)}% token savings.
                  </span>
                ) : (
                  <span className="text-muted-foreground">
                    REFRAG compression reduces answer quality by {Math.abs(results.refrag_impact.avg_ground_truth_delta * 100).toFixed(1)}%. Consider using lower compression for these query types.
                  </span>
                )}
              </div>
            </Card>
          )}

          {/* Detailed Test Results */}
          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Detailed Test Results
            </h3>
            <Accordion type="single" collapsible className="space-y-2">
              {results.test_results.map((test) => (
                <AccordionItem
                  key={test.test_id}
                  value={test.test_id}
                  className="border rounded-lg px-4"
                >
                  <AccordionTrigger className="hover:no-underline py-3">
                    <div className="flex items-center gap-4 w-full pr-4">
                      <div className="flex items-center gap-2">
                        {test.best_score >= 0.6 ? (
                          <CheckCircle2 className="w-5 h-5 text-green-500" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-500" />
                        )}
                        <span className="font-medium">{test.test_id}</span>
                      </div>
                      <Badge className={DIFFICULTY_COLORS[test.difficulty]}>
                        {test.difficulty}
                      </Badge>
                      <div className="flex-1 text-left text-sm text-muted-foreground truncate">
                        {test.query}
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          Best: {test.best_mode}
                        </Badge>
                        <span className={`font-bold ${getScoreColor(test.best_score)}`}>
                          {(test.best_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="pb-4">
                    <div className="space-y-4 pt-2">
                      {/* Query & Expected Answer */}
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        <div className="p-3 bg-secondary/30 rounded-lg">
                          <p className="text-xs text-muted-foreground mb-1">Query</p>
                          <p className="text-sm">{test.query}</p>
                        </div>
                        <div className="p-3 bg-green-500/10 rounded-lg">
                          <p className="text-xs text-muted-foreground mb-1">Expected Answer (Ground Truth)</p>
                          <p className="text-sm text-green-700 dark:text-green-400">{test.expected_answer}</p>
                        </div>
                      </div>

                      {/* Mode Results */}
                      <div className="space-y-3">
                        {test.mode_results.map((modeResult) => (
                          <ModeResultCard
                            key={modeResult.mode}
                            result={modeResult}
                            isBest={modeResult.mode === test.best_mode}
                          />
                        ))}
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </Card>
        </>
      )}

      {/* Empty State */}
      {!results && !isRunning && (
        <Card className="p-12 text-center">
          <Target className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-xl font-semibold mb-2">No evaluation results yet</h3>
          <p className="text-muted-foreground max-w-md mx-auto mb-6">
            RAGAS evaluation compares generated answers against ground truth using semantic similarity scoring.
            Select modes and run the evaluation to see results.
          </p>
          <div className="flex flex-wrap justify-center gap-4 text-sm">
            <div className="flex items-center gap-2 bg-blue-500/10 rounded-lg px-3 py-2">
              <Target className="w-4 h-4 text-blue-500" />
              <span>Ground Truth Comparison</span>
            </div>
            <div className="flex items-center gap-2 bg-green-500/10 rounded-lg px-3 py-2">
              <CheckCircle2 className="w-4 h-4 text-green-500" />
              <span>Semantic Similarity</span>
            </div>
            <div className="flex items-center gap-2 bg-purple-500/10 rounded-lg px-3 py-2">
              <Sparkles className="w-4 h-4 text-purple-500" />
              <span>REFRAG Impact Analysis</span>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}

// Mode Result Card Component
function ModeResultCard({
  result,
  isBest,
}: {
  result: RagasModeResult;
  isBest: boolean;
}) {
  return (
    <div className={`p-4 rounded-lg border ${isBest ? 'border-green-500/50 bg-green-500/5' : 'border-border'}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="font-medium">{result.mode.toUpperCase()}</span>
          {isBest && (
            <Badge className="bg-green-500 text-white text-xs">
              <Award className="w-3 h-3 mr-1" />
              Best
            </Badge>
          )}
          {result.scores.passed ? (
            <CheckCircle2 className="w-4 h-4 text-green-500" />
          ) : (
            <XCircle className="w-4 h-4 text-red-500" />
          )}
        </div>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-muted-foreground">{result.timing_ms.toFixed(0)}ms</span>
          <span className="text-muted-foreground">{result.chunks_used} chunks</span>
          <span className={`font-bold ${getScoreColor(result.scores.weighted_score)}`}>
            {(result.scores.weighted_score * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Score Breakdown */}
      <div className="grid grid-cols-5 gap-2 mb-3">
        <div className="text-center p-2 bg-secondary/30 rounded">
          <p className="text-lg font-bold">{(result.scores.ground_truth_similarity * 100).toFixed(0)}%</p>
          <p className="text-xs text-muted-foreground">Ground Truth</p>
        </div>
        <div className="text-center p-2 bg-secondary/30 rounded">
          <p className="text-lg font-bold">{(result.scores.answer_keywords * 100).toFixed(0)}%</p>
          <p className="text-xs text-muted-foreground">Keywords</p>
        </div>
        <div className="text-center p-2 bg-secondary/30 rounded">
          <p className="text-lg font-bold">{(result.scores.chunk_content * 100).toFixed(0)}%</p>
          <p className="text-xs text-muted-foreground">Chunk Match</p>
        </div>
        <div className="text-center p-2 bg-secondary/30 rounded">
          <p className="text-lg font-bold">{(result.scores.entity_match * 100).toFixed(0)}%</p>
          <p className="text-xs text-muted-foreground">Entities</p>
        </div>
        <div className="text-center p-2 bg-secondary/30 rounded">
          <p className="text-lg font-bold">{(result.scores.chunk_count * 100).toFixed(0)}%</p>
          <p className="text-xs text-muted-foreground">Chunks</p>
        </div>
      </div>

      {/* Generated Answer */}
      <div className="p-3 bg-secondary/20 rounded text-sm">
        <p className="text-xs text-muted-foreground mb-1">Generated Answer</p>
        <p className="line-clamp-3">{result.answer}</p>
      </div>

      {/* REFRAG Comparison */}
      {result.refrag_scores && (
        <div className="mt-3 p-3 bg-purple-500/10 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-purple-700 dark:text-purple-400">
              <Sparkles className="w-3 h-3 inline mr-1" />
              With REFRAG
            </span>
            <div className="flex items-center gap-3 text-sm">
              {result.ground_truth_delta !== undefined && (
                <span className={result.ground_truth_delta >= 0 ? 'text-green-600' : 'text-red-600'}>
                  {result.ground_truth_delta >= 0 ? '+' : ''}{(result.ground_truth_delta * 100).toFixed(1)}% quality
                </span>
              )}
              {result.speed_improvement && (
                <span className="text-blue-600">
                  <Zap className="w-3 h-3 inline" /> {result.speed_improvement.toFixed(1)}x faster
                </span>
              )}
              {result.token_savings !== undefined && (
                <span className="text-purple-600">
                  {result.token_savings.toFixed(0)}% tokens saved
                </span>
              )}
            </div>
          </div>
          <div className="grid grid-cols-5 gap-2 text-xs">
            <div className="text-center">
              <span className={getScoreColor(result.refrag_scores.ground_truth_similarity)}>
                {(result.refrag_scores.ground_truth_similarity * 100).toFixed(0)}%
              </span>
            </div>
            <div className="text-center">
              <span className={getScoreColor(result.refrag_scores.answer_keywords)}>
                {(result.refrag_scores.answer_keywords * 100).toFixed(0)}%
              </span>
            </div>
            <div className="text-center">
              <span className={getScoreColor(result.refrag_scores.chunk_content)}>
                {(result.refrag_scores.chunk_content * 100).toFixed(0)}%
              </span>
            </div>
            <div className="text-center">
              <span className={getScoreColor(result.refrag_scores.entity_match)}>
                {(result.refrag_scores.entity_match * 100).toFixed(0)}%
              </span>
            </div>
            <div className="text-center">
              <span className={getScoreColor(result.refrag_scores.weighted_score)}>
                {(result.refrag_scores.weighted_score * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
