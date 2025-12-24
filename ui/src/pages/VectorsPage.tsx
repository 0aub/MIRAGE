import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Search, ChevronLeft, ChevronRight, Loader2, Database, FileText, Hash, List } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { dbApi } from "@/lib/api";

interface VectorChunk {
  id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  char_count: number;
  word_count: number;
  sentence_count: number;
  metadata: Record<string, any>;
}

interface VectorStats {
  total_chunks: number;
  total_documents: number;
  avg_chunk_size: number;
  total_vectors: number;
}

export default function VectorsPage() {
  const [chunks, setChunks] = useState<VectorChunk[]>([]);
  const [stats, setStats] = useState<VectorStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingStats, setIsLoadingStats] = useState(true);
  const [total, setTotal] = useState(0);
  const [currentPage, setCurrentPage] = useState(0);
  const [limit] = useState(50); // Reduced from 100 to improve UI performance
  const [searchQuery, setSearchQuery] = useState("");
  const [filteredDocId, setFilteredDocId] = useState("");
  const { toast } = useToast();

  // Load stats on mount
  useEffect(() => {
    loadStats();
  }, []);

  // Load chunks when page or filter changes
  useEffect(() => {
    loadChunks();
  }, [currentPage, filteredDocId]);

  const loadStats = async () => {
    try {
      setIsLoadingStats(true);
      const response = await dbApi.vector.getStats();
      setStats(response);
    } catch (error: any) {
      console.error('Error loading vector stats:', error);
      // Don't show error toast for stats, they're optional
    } finally {
      setIsLoadingStats(false);
    }
  };

  const loadChunks = async () => {
    try {
      setIsLoading(true);
      const offset = currentPage * limit;
      const response = await dbApi.vector.getChunks(limit, offset, filteredDocId || undefined);

      setChunks(response.chunks);
      setTotal(response.total);
    } catch (error: any) {
      console.error('Error loading vector chunks:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to load vector chunks",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = () => {
    setFilteredDocId(searchQuery);
    setCurrentPage(0);
  };

  const handleClearFilter = () => {
    setSearchQuery("");
    setFilteredDocId("");
    setCurrentPage(0);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const totalPages = Math.ceil(total / limit);

  const truncateText = (text: string, maxLength: number = 300) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + "...";
  };

  return (
    <div className="space-y-6 animate-fade-in pb-20 md:pb-0">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2">Vector Store</h1>
        <p className="text-muted-foreground">Browse and manage document chunks in Qdrant vector database</p>
      </div>

      {/* Stats Cards */}
      {!isLoadingStats && stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <List className="w-5 h-5 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Chunks</p>
                <p className="text-2xl font-bold">{stats.total_chunks.toLocaleString()}</p>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-500/10 rounded-lg">
                <FileText className="w-5 h-5 text-green-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Documents</p>
                <p className="text-2xl font-bold">{stats.total_documents.toLocaleString()}</p>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-500/10 rounded-lg">
                <Hash className="w-5 h-5 text-purple-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Avg. Chunk Size</p>
                <p className="text-2xl font-bold">{Math.round(stats.avg_chunk_size)}</p>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-orange-500/10 rounded-lg">
                <Database className="w-5 h-5 text-orange-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Vectors</p>
                <p className="text-2xl font-bold">{stats.total_vectors.toLocaleString()}</p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Search and Filter */}
      <Card className="p-6">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search by document ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              className="pl-10"
            />
          </div>
          <div className="flex gap-2">
            <Button onClick={handleSearch} variant="default">
              <Search className="w-4 h-4 mr-2" />
              Search
            </Button>
            {filteredDocId && (
              <Button onClick={handleClearFilter} variant="outline">
                Clear Filter
              </Button>
            )}
          </div>
        </div>

        {filteredDocId && (
          <div className="mt-4">
            <Badge variant="secondary">
              Filtered by: {filteredDocId}
            </Badge>
          </div>
        )}
      </Card>

      {/* Chunks List */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold">
            Chunks ({total.toLocaleString()})
          </h2>
          <p className="text-sm text-muted-foreground">
            Page {currentPage + 1} of {totalPages || 1}
          </p>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
            <p className="ml-3 text-muted-foreground">Loading chunks...</p>
          </div>
        ) : chunks.length === 0 ? (
          <div className="text-center py-12">
            <Database className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-xl font-semibold mb-2">No chunks found</h3>
            <p className="text-muted-foreground">
              {filteredDocId ? 'No chunks match your filter criteria' : 'No chunks in the vector store yet'}
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {chunks.map((chunk) => (
              <Card key={chunk.id} className="p-4 hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant="outline" className="font-mono text-xs">
                        {chunk.document_id}
                      </Badge>
                      <Badge variant="secondary" className="text-xs">
                        Chunk {chunk.chunk_index}
                      </Badge>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <span title="Word count">{chunk.word_count}w</span>
                    <span title="Character count">{chunk.char_count} chars</span>
                  </div>
                </div>

                <div className="text-sm leading-relaxed bg-muted/30 rounded-lg p-3 font-mono">
                  {truncateText(chunk.text)}
                </div>

                {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                  <div className="mt-3 pt-3 border-t text-xs text-muted-foreground">
                    <span className="font-semibold">Metadata:</span>{' '}
                    {JSON.stringify(chunk.metadata)}
                  </div>
                )}
              </Card>
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-2 mt-6">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
              disabled={currentPage === 0}
            >
              <ChevronLeft className="w-4 h-4 mr-1" />
              Previous
            </Button>

            <div className="flex items-center gap-2 px-4">
              <span className="text-sm text-muted-foreground">
                Page {currentPage + 1} of {totalPages}
              </span>
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
              disabled={currentPage >= totalPages - 1}
            >
              Next
              <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          </div>
        )}
      </Card>
    </div>
  );
}
