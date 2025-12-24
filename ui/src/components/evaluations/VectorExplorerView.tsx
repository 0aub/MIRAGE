import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Search,
  Database,
  FileText,
  Target,
  Sparkles,
  Loader2,
  AlertCircle
} from "lucide-react";
import { dbApi, chatApi } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

export function VectorExplorerView() {
  const [query, setQuery] = useState("");
  const [chunks, setChunks] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [dbStats, setDbStats] = useState<any>(null);
  const { toast } = useToast();

  const searchVectors = async () => {
    setIsLoading(true);
    setChunks([]);

    try {
      if (query.trim()) {
        // Semantic Search using Chat API with Naive mode
        const response = await chatApi.retrieve({
          message: query,
          retrieval_mode: 'naive',
          top_k: 20
        });
        
        // Map ChunkInfo to the expected display format
        setChunks(response.chunks.map(c => ({
            id: c.chunk_id,
            text: c.text,
            score: c.score,
            metadata: c.metadata || { score: c.score }
        })));
        
        if (response.chunks.length === 0) {
           toast({ title: "No results found", description: "Try a different query" });
        }
      } else {
        // Browse mode: List chunks from database
        const stats = await dbApi.vectorInfo();
        setDbStats(stats);

        const response = await dbApi.vector.getChunks(20, 0);
        setChunks(response.chunks);
      }

    } catch (error: any) {
      toast({
        title: "Search Failed",
        description: error.message,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
       <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-1">Vector Explorer</h2>
            <p className="text-muted-foreground">
              Inspect semantic vectors and chunks directly from the database
            </p>
          </div>
        </div>

      <Card className="p-6">
        <div className="flex gap-4 mb-6">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search vector database..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="pl-9 h-12"
              onKeyDown={(e) => e.key === "Enter" && searchVectors()}
            />
          </div>
          <Button onClick={searchVectors} disabled={isLoading} className="h-12 px-8">
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Search Vectors"}
          </Button>
        </div>

        {dbStats && (
            <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="p-4 bg-secondary/30 rounded-lg text-center">
                    <div className="text-sm text-muted-foreground">Collection</div>
                    <div className="font-bold">{dbStats.collection_name}</div>
                </div>
                <div className="p-4 bg-secondary/30 rounded-lg text-center">
                    <div className="text-sm text-muted-foreground">Total Vectors</div>
                    <div className="font-bold">{dbStats.vectors_count}</div>
                </div>
                <div className="p-4 bg-secondary/30 rounded-lg text-center">
                    <div className="text-sm text-muted-foreground">Status</div>
                    <div className="font-bold text-green-500">{dbStats.status}</div>
                </div>
            </div>
        )}

        <div className="space-y-4">
          <h3 className="font-semibold flex items-center gap-2">
            <Database className="w-4 h-4" />
            Results {chunks.length > 0 && `(${chunks.length})`}
          </h3>
          
          <ScrollArea className="h-[500px] pr-4">
            {chunks.length > 0 ? (
                <div className="space-y-3">
                {chunks.map((chunk, idx) => (
                    <Card key={idx} className="p-4 hover:bg-secondary/20 transition-colors">
                    <div className="flex items-start justify-between mb-2">
                        <Badge variant="outline" className="font-mono text-xs">
                         ID: {chunk.id?.substring(0, 8)}...
                        </Badge>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <FileText className="w-3 h-3" />
                            {chunk.word_count} words
                        </div>
                    </div>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">
                        {chunk.text}
                    </p>
                    {chunk.metadata && (
                        <div className="mt-3 pt-3 border-t flex flex-wrap gap-2">
                             {Object.entries(chunk.metadata).map(([k, v]) => (
                                 <Badge key={k} variant="secondary" className="text-xs">
                                     {k}: {String(v).substring(0, 20)}
                                 </Badge>
                             ))}
                        </div>
                    )}
                    </Card>
                ))}
                </div>
            ) : (
                <div className="text-center py-12 text-muted-foreground">
                    <Sparkles className="w-12 h-12 mx-auto mb-4 opacity-20" />
                    <p>Enter a query to explore semantic proximity in the vector space</p>
                </div>
            )}
            </ScrollArea>
        </div>
      </Card>
    </div>
  );
}
