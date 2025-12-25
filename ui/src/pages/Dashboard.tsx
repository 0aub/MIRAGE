import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Database, FileText, Network, Activity, TrendingUp, Clock,
  Loader2, Cpu, Users, GitBranch, Zap, BarChart3, CheckCircle2, XCircle, RefreshCw
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { dbApi, documentsApi, graphragApi, chatApi } from "@/lib/api";

interface DashboardStats {
  documents: number;
  entities: number;
  relationships: number;
  vectors: number;
  communities: number;
}

interface PerformanceMetrics {
  totalQueries: number;
  avgLatencyMs: number;
  p95LatencyMs: number;
  cacheHitRate: number;
  errorRate: number;
}

interface ServiceHealth {
  name: string;
  status: 'healthy' | 'unhealthy' | 'unknown';
  details?: string;
}

interface ModelConfig {
  llm: {
    provider: string;
    model: string;
    description: string;
    endpoint?: string;
  };
  embedding: {
    model: string;
    dimensions: number;
  };
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    documents: 0,
    entities: 0,
    relationships: 0,
    vectors: 0,
    communities: 0,
  });
  const [performance, setPerformance] = useState<PerformanceMetrics | null>(null);
  const [services, setServices] = useState<ServiceHealth[]>([]);
  const [modeDistribution, setModeDistribution] = useState<Record<string, number>>({});
  const [recentDocs, setRecentDocs] = useState<any[]>([]);
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    llm: { provider: 'Unknown', model: 'Unknown', description: 'Loading...' },
    embedding: { model: 'Unknown', dimensions: 0 },
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setIsLoading(true);

      // Load all data in parallel
      const [dbStats, healthData, docsData, graphragHealth, v5Metrics, settingsData] = await Promise.allSettled([
        dbApi.stats(),
        dbApi.health(),
        documentsApi.list({ limit: 5 }),
        graphragApi.health(),
        chatApi.v5Metrics(),
        dbApi.getSettings(),
      ]);

      // Process database stats
      if (dbStats.status === 'fulfilled') {
        const data = dbStats.value;
        setStats(prev => ({
          ...prev,
          entities: data.graph.total_nodes || 0,
          relationships: data.graph.total_edges || 0,
          vectors: data.vector.vectors_count || 0,
        }));
      }

      // Process health data
      if (healthData.status === 'fulfilled') {
        const health = healthData.value;
        setServices(health.databases?.map((db: any) => ({
          name: db.database,
          status: db.status === 'healthy' ? 'healthy' : 'unhealthy',
          details: db.nodes ? `${db.nodes} nodes, ${db.edges} edges` :
                   db.vectors ? `${db.vectors} vectors` : undefined,
        })) || []);
      }

      // Process documents
      if (docsData.status === 'fulfilled') {
        setStats(prev => ({
          ...prev,
          documents: docsData.value.total,
        }));
        setRecentDocs(docsData.value.documents);
      }

      // Process GraphRAG health (for community count)
      if (graphragHealth.status === 'fulfilled') {
        setStats(prev => ({
          ...prev,
          communities: graphragHealth.value.community_count || 0,
        }));
      }

      // Process V5 metrics
      if (v5Metrics.status === 'fulfilled') {
        const metrics = v5Metrics.value;
        setPerformance({
          totalQueries: metrics.metrics.total_queries || 0,
          avgLatencyMs: metrics.metrics.avg_latency_ms || 0,
          p95LatencyMs: metrics.metrics.p95_latency_ms || 0,
          cacheHitRate: metrics.metrics.cache_hit_rate || 0,
          errorRate: metrics.metrics.error_rate || 0,
        });
        setModeDistribution(metrics.mode_distribution || {});
      }

      // Process settings for model config
      if (settingsData.status === 'fulfilled') {
        const settings = settingsData.value.settings;
        const models = settings?.models || {};

        // Find the active LLM provider
        const llmProviders = models.llm_providers || {};
        let activeLLM = { provider: 'Unknown', model: 'Unknown', description: 'Not configured', endpoint: '' };

        for (const [provider, config] of Object.entries(llmProviders)) {
          const cfg = config as any;
          if (cfg.enabled) {
            activeLLM = {
              provider: provider.toUpperCase(),
              model: cfg.model || provider,
              description: cfg.description || `${provider} LLM`,
              endpoint: cfg.endpoint || '',
            };
            break;
          }
        }

        // Get embedding model config
        const embeddings = models.embeddings || {};
        const jinaConfig = embeddings.jina || {};
        const embeddingModel = jinaConfig.model || 'Unknown';
        const embeddingDim = jinaConfig.embedding_dim || 768;

        setModelConfig(prev => ({
          ...prev,
          llm: activeLLM,
          embedding: {
            model: embeddingModel,
            dimensions: embeddingDim,
          },
        }));
      }

    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await loadDashboardData();
    setIsRefreshing(false);
  };

  const statCards = [
    { label: "Documents", value: stats.documents, icon: FileText, color: "text-blue-500", bg: "bg-blue-500/10" },
    { label: "Entities", value: stats.entities, icon: Network, color: "text-green-500", bg: "bg-green-500/10" },
    { label: "Relationships", value: stats.relationships, icon: GitBranch, color: "text-purple-500", bg: "bg-purple-500/10" },
    { label: "Vectors", value: stats.vectors, icon: Database, color: "text-orange-500", bg: "bg-orange-500/10" },
    { label: "Communities", value: stats.communities, icon: Users, color: "text-pink-500", bg: "bg-pink-500/10" },
  ];

  return (
    <div className="space-y-6 animate-fade-in pb-20 md:pb-0">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-1">Dashboard</h1>
          <p className="text-muted-foreground">MIRAGE System Overview</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="gap-2"
        >
          <RefreshCw className={cn("w-4 h-4", isRefreshing && "animate-spin")} />
          Refresh
        </Button>
      </div>

      {/* Hero Section */}
      <Card className="bg-gradient-to-r from-primary/10 via-primary/5 to-transparent p-6 border-primary/20">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-1">Welcome to MIRAGE</h2>
            <p className="text-muted-foreground">
              Multilingual Information Retrieval with Accelerated Graph Embeddings
            </p>
            <div className="flex items-center gap-4 mt-3">
              <Badge variant="secondary" className="gap-1">
                <Cpu className="w-3 h-3" />
                Local TGI Model
              </Badge>
              <Badge variant="secondary" className="gap-1">
                <Zap className="w-3 h-3" />
                7 Retrieval Modes
              </Badge>
              <Badge variant="secondary" className="gap-1">
                <Activity className="w-3 h-3" />
                GraphRAG Enabled
              </Badge>
            </div>
          </div>
          <Database className="w-16 h-16 text-primary/20 hidden md:block" />
        </div>
      </Card>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {isLoading ? (
          <div className="col-span-full text-center py-8">
            <Loader2 className="w-8 h-8 mx-auto text-primary animate-spin" />
          </div>
        ) : (
          statCards.map((stat) => {
            const Icon = stat.icon;
            return (
              <Card key={stat.label} className="p-4 hover:shadow-md transition-shadow">
                <div className={cn("p-2 rounded-lg w-fit mb-3", stat.bg)}>
                  <Icon className={cn("w-5 h-5", stat.color)} />
                </div>
                <p className="text-2xl font-bold">{stat.value.toLocaleString()}</p>
                <p className="text-sm text-muted-foreground">{stat.label}</p>
              </Card>
            );
          })
        )}
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Performance Metrics */}
        <Card className="p-6 lg:col-span-2">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-5 h-5 text-primary" />
            <h3 className="font-semibold">Performance Metrics</h3>
          </div>

          {performance ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-secondary/30 rounded-lg p-4">
                <p className="text-xs text-muted-foreground mb-1">Total Queries</p>
                <p className="text-xl font-bold">{performance.totalQueries.toLocaleString()}</p>
              </div>
              <div className="bg-secondary/30 rounded-lg p-4">
                <p className="text-xs text-muted-foreground mb-1">Avg Latency</p>
                <p className="text-xl font-bold">{performance.avgLatencyMs.toFixed(0)}ms</p>
              </div>
              <div className="bg-secondary/30 rounded-lg p-4">
                <p className="text-xs text-muted-foreground mb-1">P95 Latency</p>
                <p className="text-xl font-bold">{performance.p95LatencyMs.toFixed(0)}ms</p>
              </div>
              <div className="bg-secondary/30 rounded-lg p-4">
                <p className="text-xs text-muted-foreground mb-1">Cache Hit Rate</p>
                <p className="text-xl font-bold">{(performance.cacheHitRate * 100).toFixed(1)}%</p>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <TrendingUp className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No performance data available yet</p>
              <p className="text-xs">Run some queries to see metrics</p>
            </div>
          )}

          {/* Mode Distribution */}
          {Object.keys(modeDistribution).length > 0 && (
            <div className="mt-6">
              <p className="text-sm font-medium mb-3">Mode Distribution</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(modeDistribution).map(([mode, count]) => (
                  <Badge key={mode} variant="outline" className="gap-1">
                    {mode}: {count}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </Card>

        {/* System Health */}
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="w-5 h-5 text-primary" />
            <h3 className="font-semibold">System Health</h3>
          </div>

          <div className="space-y-3">
            {isLoading ? (
              <div className="text-center py-4">
                <Loader2 className="w-6 h-6 mx-auto text-primary animate-spin" />
              </div>
            ) : services.length > 0 ? (
              services.map((service) => (
                <div key={service.name} className="flex items-center justify-between p-3 bg-secondary/30 rounded-lg">
                  <div className="flex items-center gap-3">
                    {service.status === 'healthy' ? (
                      <CheckCircle2 className="w-4 h-4 text-green-500" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-500" />
                    )}
                    <div>
                      <p className="font-medium capitalize">{service.name}</p>
                      {service.details && (
                        <p className="text-xs text-muted-foreground">{service.details}</p>
                      )}
                    </div>
                  </div>
                  <Badge
                    variant={service.status === 'healthy' ? 'default' : 'destructive'}
                    className="text-xs"
                  >
                    {service.status}
                  </Badge>
                </div>
              ))
            ) : (
              <p className="text-sm text-muted-foreground text-center py-4">
                Unable to fetch health status
              </p>
            )}
          </div>
        </Card>
      </div>

      {/* Recent Documents */}
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <FileText className="w-5 h-5 text-primary" />
          <h3 className="font-semibold">Recent Documents</h3>
        </div>

        {recentDocs.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {recentDocs.map((doc) => (
              <Card key={doc.document_id} className="p-4 bg-secondary/20">
                <div className="flex items-start justify-between gap-2 mb-2">
                  <p className="font-medium text-sm line-clamp-1">{doc.title}</p>
                  <Badge variant="outline" className="text-xs shrink-0">
                    {doc.content_type}
                  </Badge>
                </div>
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Network className="w-3 h-3" />
                    {doc.entity_count} entities
                  </span>
                  <span className="flex items-center gap-1">
                    <GitBranch className="w-3 h-3" />
                    {doc.relationship_count} rels
                  </span>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No documents processed yet</p>
            <p className="text-xs">Go to Data Sources to add documents</p>
          </div>
        )}
      </Card>

      {/* Quick Info - Dynamic from Backend */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4 bg-blue-500/5 border-blue-500/20">
          <div className="flex items-center gap-2 mb-2">
            <Cpu className="w-4 h-4 text-blue-500" />
            <p className="font-medium text-sm">LLM Engine</p>
          </div>
          <p className="text-xs text-muted-foreground">
            {modelConfig.llm.description}
          </p>
          {modelConfig.llm.provider !== 'Unknown' && (
            <Badge variant="outline" className="mt-2 text-xs">
              {modelConfig.llm.provider}
            </Badge>
          )}
        </Card>
        <Card className="p-4 bg-green-500/5 border-green-500/20">
          <div className="flex items-center gap-2 mb-2">
            <Database className="w-4 h-4 text-green-500" />
            <p className="font-medium text-sm">Embedding Model</p>
          </div>
          <p className="text-xs text-muted-foreground">
            {modelConfig.embedding.model} ({modelConfig.embedding.dimensions}-dim) for semantic search
          </p>
        </Card>
      </div>
    </div>
  );
}
