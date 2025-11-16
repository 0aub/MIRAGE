import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Database, FileText, Network, Activity, TrendingUp, Clock, Loader2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { dbApi, filesApi } from "@/lib/api";

export default function Dashboard() {
  const [stats, setStats] = useState([
    { label: "Total Files", value: "...", icon: FileText, color: "text-node-document" },
    { label: "Entities", value: "...", icon: Network, color: "text-node-entity" },
    { label: "Relationships", value: "...", icon: Activity, color: "text-node-concept" },
    { label: "Vector Count", value: "...", icon: TrendingUp, color: "text-accent" },
  ]);
  const [healthStatus, setHealthStatus] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setIsLoading(true);

      // Load database stats
      const [dbStats, fileStats] = await Promise.all([
        dbApi.stats(),
        filesApi.stats(),
      ]);

      // Update stats cards
      setStats([
        {
          label: "Total Files",
          value: fileStats.total_files.toString(),
          icon: FileText,
          color: "text-node-document"
        },
        {
          label: "Entities",
          value: dbStats.graph.total_nodes?.toString() || "0",
          icon: Network,
          color: "text-node-entity"
        },
        {
          label: "Relationships",
          value: dbStats.graph.total_edges?.toString() || "0",
          icon: Activity,
          color: "text-node-concept"
        },
        {
          label: "Vector Count",
          value: dbStats.vector.vectors_count?.toString() || "0",
          icon: TrendingUp,
          color: "text-accent"
        },
      ]);

      // Load health status
      const health = await dbApi.health();
      setHealthStatus(health.databases || []);

    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const recentChats: any[] = [];
  const processingQueue: any[] = [];

  return (
    <div className="space-y-8 animate-fade-in pb-20 md:pb-0">
      {/* Hero Section */}
      <div className="gradient-hero rounded-2xl p-8 text-white animate-slide-up">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold mb-2">Welcome to MIRAGE</h1>
            <p className="text-white/80 text-lg">Multilingual Information Retrieval with Accelerated Graph Embeddings</p>
          </div>
          <Database className="w-16 h-16 text-white/20 hidden md:block" />
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 animate-scale-in">
        {isLoading ? (
          <div className="col-span-4 text-center py-8">
            <Loader2 className="w-8 h-8 mx-auto text-primary animate-spin" />
          </div>
        ) : (
          stats.map((stat) => {
            const Icon = stat.icon;
            return (
              <Card key={stat.label} className="p-6 hover:shadow-lg transition-shadow">
                <div className="flex items-start justify-between mb-4">
                  <div className={cn("p-3 rounded-lg bg-secondary/40", stat.color)}>
                    <Icon className="w-6 h-6" />
                  </div>
                </div>
                <p className="text-2xl font-bold mb-1">{stat.value}</p>
                <p className="text-sm text-muted-foreground">{stat.label}</p>
              </Card>
            );
          })
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Processing Queue */}
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-6">
            <Activity className="w-5 h-5 text-primary" />
            <h2 className="text-xl font-semibold">Processing Queue</h2>
          </div>
          <div className="space-y-4">
            {processingQueue.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-8">No documents processing</p>
            ) : (
              processingQueue.map((item) => (
                <div key={item.id} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium truncate flex-1">{item.name}</p>
                    <span className="text-xs text-muted-foreground ml-2">{item.status}</span>
                  </div>
                  <Progress value={item.progress} className="h-2" />
                </div>
              ))
            )}
          </div>
        </Card>

        {/* Recent Chat Sessions */}
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-6">
            <Clock className="w-5 h-5 text-primary" />
            <h2 className="text-xl font-semibold">Recent Sessions</h2>
          </div>
          <div className="space-y-4">
            {recentChats.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-8">No recent chat sessions</p>
            ) : (
              recentChats.map((chat) => (
                <div key={chat.id} className="p-4 rounded-lg bg-secondary/50 hover:bg-secondary transition-colors cursor-pointer">
                  <p className="text-sm mb-2 line-clamp-1">{chat.query}</p>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>{chat.time}</span>
                    <span>{chat.nodes} nodes</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>

      {/* System Health */}
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-6">
          <Activity className="w-5 h-5 text-accent" />
          <h2 className="text-xl font-semibold">System Health</h2>
        </div>
        {isLoading ? (
          <div className="text-center py-4">
            <Loader2 className="w-6 h-6 mx-auto text-primary animate-spin" />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {healthStatus.map((db) => (
              <div key={db.database} className="space-y-2">
                <p className="text-sm text-muted-foreground capitalize">{db.database}</p>
                <div className="flex items-center gap-2">
                  <div className={cn(
                    "w-2 h-2 rounded-full",
                    db.status === "healthy" ? "bg-accent animate-pulse" : "bg-destructive"
                  )} />
                  <span className="text-sm font-medium capitalize">{db.status}</span>
                </div>
                {db.nodes !== undefined && (
                  <p className="text-xs text-muted-foreground">{db.nodes} nodes, {db.edges} edges</p>
                )}
                {db.vectors !== undefined && (
                  <p className="text-xs text-muted-foreground">{db.vectors} vectors</p>
                )}
              </div>
            ))}
            {healthStatus.length === 0 && (
              <div className="col-span-3 text-center text-sm text-muted-foreground">
                Unable to fetch health status
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
