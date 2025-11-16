import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Search, ChevronLeft, ChevronRight, Loader2, Database } from "lucide-react";
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

export default function VectorsPage() {
  const [chunks, setChunks] = useState<VectorChunk[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [currentPage, setCurrentPage] = useState(0);
  const [limit] = useState(50);
  const [searchQuery, setSearchQuery] = useState("");
  const [filteredDocId, setFilteredDocId] = useState("");
  const { toast } = useToast();

  useEffect(() => {
    loadChunks();
  }, [currentPage, filteredDocId]);

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

  const totalPages = Math.ceil(total / limit);

  const filteredChunks = chunks.filter(chunk => {
    if (!searchQuery) return true;
    const lowerQuery = searchQuery.toLowerCase();
    return (
      chunk.document_id.toLowerCase().includes(lowerQuery) ||
      chunk.text.toLowerCase().includes(lowerQuery)
    );
  });

  const truncateText = (text: string, maxLength: number = 200) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + "...";
  };

  return (
    <div className="space-y-6 animate-fade-in pb-20 md:pb-0">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">Vector Store</h1>
          <p className="text-muted-foreground">Browse document chunks stored in Qdrant</p>
        </div>
        <div className="hidden md:flex items-center gap-2 text-sm text-muted-foreground">
          <Database className="w-4 h-4" />
          <span>{total.toLocaleString()} total chunks</span>
        </div>
      </div>

      {/* Search and Filter */}
      <Card className="p-6">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search by document ID or text..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              className="pl-9"
            />
          </div>
          <div className="flex gap-2">
            <Button onClick={handleSearch} variant="default">
              Filter
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
            <Badge variant="secondary">Filtered by: {filteredDocId}</Badge>
          </div>
        )}
      </Card>

      {/* Chunks List */}
      {isLoading ? (
        <div className="text-center py-12">
          <Loader2 className="w-12 h-12 mx-auto text-primary animate-spin mb-4" />
          <p className="text-muted-foreground">Loading chunks...</p>
        </div>
      ) : filteredChunks.length === 0 ? (
        <Card className="p-12 text-center">
          <Database className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-xl font-semibold mb-2">No chunks found</h3>
          <p className="text-muted-foreground">
            {filteredDocId
              ? "Try clearing the filter or using a different search query"
              : "Process some documents to populate the vector store"}
          </p>
        </Card>
      ) : (
        <>
          <div className="space-y-4">
            {filteredChunks.map((chunk) => (
              <Card key={chunk.id} className="p-6 hover:shadow-lg transition-shadow">
                <div className="space-y-4">
                  {/* Header */}
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="outline" className="font-mono text-xs">
                          Chunk {chunk.chunk_index}
                        </Badge>
                        <span className="text-sm text-muted-foreground truncate">
                          {chunk.document_id}
                        </span>
                      </div>
                    </div>
                    <div className="flex gap-2 text-xs text-muted-foreground">
                      <span>{chunk.word_count} words</span>
                      <span>•</span>
                      <span>{chunk.sentence_count} sentences</span>
                    </div>
                  </div>

                  {/* Chunk Text */}
                  <div className="bg-secondary/30 rounded-lg p-4">
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">
                      {truncateText(chunk.text, 300)}
                    </p>
                  </div>

                  {/* Metadata */}
                  {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                    <div className="pt-2 border-t">
                      <details className="text-sm">
                        <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                          View metadata
                        </summary>
                        <pre className="mt-2 p-3 bg-secondary/30 rounded text-xs overflow-x-auto">
                          {JSON.stringify(chunk.metadata, null, 2)}
                        </pre>
                      </details>
                    </div>
                  )}

                  {/* Footer Stats */}
                  <div className="flex items-center gap-4 text-xs text-muted-foreground pt-2 border-t">
                    <span>ID: {chunk.id.substring(0, 8)}...</span>
                    <span>•</span>
                    <span>{chunk.char_count.toLocaleString()} characters</span>
                  </div>
                </div>
              </Card>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div className="text-sm text-muted-foreground">
                  Page {currentPage + 1} of {totalPages}
                  <span className="ml-2">
                    (Showing {currentPage * limit + 1}-{Math.min((currentPage + 1) * limit, total)} of {total.toLocaleString()})
                  </span>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
                    disabled={currentPage === 0}
                  >
                    <ChevronLeft className="w-4 h-4 mr-1" />
                    Previous
                  </Button>
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
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
