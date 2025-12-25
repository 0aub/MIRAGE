import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card } from "@/components/ui/card";
import {
  BarChart3,
  Target,
  Database,
  LayoutDashboard
} from "lucide-react";

import { BenchmarkView } from "@/components/evaluations/BenchmarkView";
import { RagasView } from "@/components/evaluations/RagasView";
import { VectorExplorerView } from "@/components/evaluations/VectorExplorerView";

export default function EvaluationsPage() {
  const [activeTab, setActiveTab] = useState("benchmark");

  return (
    <div className="container mx-auto py-8 px-4 max-w-7xl animate-fade-in">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
          <LayoutDashboard className="w-10 h-10 text-primary" />
          System Evaluation & Inspection
        </h1>
        <p className="text-xl text-muted-foreground">
          Comprehensive suite for benchmarking, optimizing, and inspecting the RAG pipeline
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <div className="flex items-center overflow-x-auto pb-2">
            <TabsList className="h-12 p-1 bg-background border">
            <TabsTrigger value="benchmark" className="h-full px-6 text-sm font-medium data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                <BarChart3 className="w-4 h-4 mr-2" />
                Comparative Benchmark
            </TabsTrigger>
            <TabsTrigger value="ragas" className="h-full px-6 text-sm font-medium data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                <Target className="w-4 h-4 mr-2" />
                RAGAS Scoring
            </TabsTrigger>
            <TabsTrigger value="vectors" className="h-full px-6 text-sm font-medium data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                <Database className="w-4 h-4 mr-2" />
                Vector Explorer
            </TabsTrigger>
            </TabsList>
        </div>

        <TabsContent value="benchmark" className="space-y-6 focus-visible:outline-none">
          <BenchmarkView />
        </TabsContent>

        <TabsContent value="ragas" className="space-y-6 focus-visible:outline-none">
          <RagasView />
        </TabsContent>

        <TabsContent value="vectors" className="space-y-6 focus-visible:outline-none">
          <VectorExplorerView />
        </TabsContent>
      </Tabs>
    </div>
  );
}
