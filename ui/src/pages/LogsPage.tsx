import { useState, useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Terminal,
  Pause,
  Play,
  Trash2,
  Download,
  Filter,
  RefreshCw
} from "lucide-react";
import { cn } from "@/lib/utils";

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  raw: string;
}

interface Container {
  name: string;
  status: string;
  running: boolean;
}

export default function LogsPage() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isStreaming, setIsStreaming] = useState(true);
  const [selectedContainer, setSelectedContainer] = useState("mirage-api");
  const [containers, setContainers] = useState<Container[]>([]);
  const [levelFilter, setLevelFilter] = useState<string | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

  // Load available containers
  useEffect(() => {
    loadContainers();
  }, []);

  // Setup SSE stream
  useEffect(() => {
    if (!isStreaming) {
      closeEventSource();
      return;
    }

    // Build SSE URL
    const params = new URLSearchParams({
      container: selectedContainer,
      tail: "200"
    });

    if (levelFilter) {
      params.append("level", levelFilter);
    }

    const url = `${API_BASE}/logs/stream?${params.toString()}`;

    // Create EventSource for SSE
    const eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      try {
        const entry: LogEntry = JSON.parse(event.data);
        setLogs((prev) => [...prev, entry]);
      } catch (error) {
        console.error("Error parsing log entry:", error);
      }
    };

    eventSource.onerror = (error) => {
      console.error("SSE connection error:", error);
      // Auto-reconnect after 3 seconds
      setTimeout(() => {
        if (isStreaming) {
          eventSource.close();
          // Will reconnect via useEffect dependency
        }
      }, 3000);
    };

    eventSourceRef.current = eventSource;

    return () => {
      closeEventSource();
    };
  }, [isStreaming, selectedContainer, levelFilter]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const closeEventSource = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  };

  const loadContainers = async () => {
    try {
      const response = await fetch(`${API_BASE}/logs/containers`);
      const data = await response.json();
      setContainers(data.containers || []);
    } catch (error) {
      console.error("Error loading containers:", error);
    }
  };

  const handleClearLogs = () => {
    setLogs([]);
  };

  const handleToggleStreaming = () => {
    setIsStreaming(!isStreaming);
  };

  const handleDownloadLogs = () => {
    const logText = logs.map(log =>
      `[${log.timestamp}] ${log.level}: ${log.message}`
    ).join('\n');

    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mirage-logs-${selectedContainer}-${new Date().toISOString()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getLevelColor = (level: string): string => {
    switch (level.toUpperCase()) {
      case 'ERROR':
        return 'text-red-500 dark:text-red-400';
      case 'WARNING':
        return 'text-yellow-500 dark:text-yellow-400';
      case 'DEBUG':
        return 'text-gray-500 dark:text-gray-400';
      case 'INFO':
      default:
        return 'text-green-500 dark:text-green-400';
    }
  };

  const getLevelBadgeVariant = (level: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (level.toUpperCase()) {
      case 'ERROR':
        return 'destructive';
      case 'WARNING':
        return 'default';
      case 'DEBUG':
        return 'outline';
      case 'INFO':
      default:
        return 'secondary';
    }
  };

  const filteredLogs = logs;

  const logStats = {
    total: logs.length,
    errors: logs.filter(l => l.level === 'ERROR').length,
    warnings: logs.filter(l => l.level === 'WARNING').length,
    info: logs.filter(l => l.level === 'INFO').length,
  };

  return (
    <div className="space-y-6 animate-fade-in pb-20 md:pb-0">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-lg bg-primary/10">
            <Terminal className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">System Logs</h1>
            <p className="text-muted-foreground">Real-time monitoring and debugging</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoScroll(!autoScroll)}
          >
            {autoScroll ? "Auto-scroll: ON" : "Auto-scroll: OFF"}
          </Button>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Total Logs</p>
            <Badge variant="secondary">{logStats.total}</Badge>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Errors</p>
            <Badge variant="destructive">{logStats.errors}</Badge>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Warnings</p>
            <Badge variant="default">{logStats.warnings}</Badge>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Info</p>
            <Badge variant="secondary">{logStats.info}</Badge>
          </div>
        </Card>
      </div>

      {/* Controls */}
      <Card className="p-4">
        <div className="flex flex-wrap items-center gap-4">
          {/* Container Selection */}
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-muted-foreground" />
            <select
              value={selectedContainer}
              onChange={(e) => setSelectedContainer(e.target.value)}
              className="px-3 py-2 rounded-md border bg-background text-sm"
            >
              {containers.map((container) => (
                <option key={container.name} value={container.name}>
                  {container.name} {container.running ? '(Running)' : '(Stopped)'}
                </option>
              ))}
              {containers.length === 0 && (
                <option value="mirage-api">mirage-api</option>
              )}
            </select>
          </div>

          {/* Level Filter */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Level:</span>
            <div className="flex gap-1">
              {['INFO', 'WARNING', 'ERROR', 'DEBUG'].map((level) => (
                <Button
                  key={level}
                  variant={levelFilter === level ? "default" : "outline"}
                  size="sm"
                  onClick={() => setLevelFilter(levelFilter === level ? null : level)}
                >
                  {level}
                </Button>
              ))}
            </div>
          </div>

          <div className="ml-auto flex items-center gap-2">
            {/* Refresh Containers */}
            <Button
              variant="outline"
              size="icon"
              onClick={loadContainers}
              title="Refresh containers"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>

            {/* Play/Pause */}
            <Button
              variant={isStreaming ? "default" : "outline"}
              size="icon"
              onClick={handleToggleStreaming}
              title={isStreaming ? "Pause streaming" : "Resume streaming"}
            >
              {isStreaming ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </Button>

            {/* Clear */}
            <Button
              variant="outline"
              size="icon"
              onClick={handleClearLogs}
              title="Clear logs"
            >
              <Trash2 className="w-4 h-4" />
            </Button>

            {/* Download */}
            <Button
              variant="outline"
              size="icon"
              onClick={handleDownloadLogs}
              title="Download logs"
              disabled={logs.length === 0}
            >
              <Download className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </Card>

      {/* Log Display */}
      <Card className="p-0">
        <div className="border-b p-4 bg-muted/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-medium">
                {selectedContainer}
              </span>
              {isStreaming && (
                <Badge variant="default" className="animate-pulse">
                  LIVE
                </Badge>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              {filteredLogs.length} log entries
            </p>
          </div>
        </div>

        <ScrollArea className="h-[600px]" ref={scrollRef}>
          <div className="p-4 font-mono text-sm space-y-1">
            {filteredLogs.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <Terminal className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No logs yet. {isStreaming ? 'Waiting for logs...' : 'Start streaming to see logs.'}</p>
              </div>
            ) : (
              filteredLogs.map((log, index) => (
                <div
                  key={index}
                  className={cn(
                    "py-1 px-2 rounded hover:bg-muted/50 transition-colors",
                    "border-l-2",
                    log.level === 'ERROR' && "border-red-500",
                    log.level === 'WARNING' && "border-yellow-500",
                    log.level === 'DEBUG' && "border-gray-500",
                    log.level === 'INFO' && "border-green-500"
                  )}
                >
                  <div className="flex items-start gap-3">
                    <span className="text-xs text-muted-foreground whitespace-nowrap">
                      {log.timestamp}
                    </span>
                    <Badge
                      variant={getLevelBadgeVariant(log.level)}
                      className="text-xs min-w-[70px] justify-center"
                    >
                      {log.level}
                    </Badge>
                    <span className={cn("flex-1", getLevelColor(log.level))}>
                      {log.message}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </Card>

      {/* Connection Status */}
      {!isStreaming && (
        <Card className="p-4 bg-yellow-500/10 border-yellow-500/20">
          <div className="flex items-center gap-2 text-yellow-600 dark:text-yellow-400">
            <Pause className="w-4 h-4" />
            <p className="text-sm">Log streaming is paused. Click Play to resume.</p>
          </div>
        </Card>
      )}
    </div>
  );
}
