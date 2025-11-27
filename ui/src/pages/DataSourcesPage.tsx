import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Upload,
  FileText,
  Youtube,
  Globe,
  Search,
  MoreVertical,
  Eye,
  RefreshCw,
  Trash2,
  FileJson,
  FileCode,
  Image as ImageIcon,
  CheckCircle2,
  Clock,
  XCircle,
  Loader2
} from "lucide-react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { useToast } from "@/hooks/use-toast";
import { documentsApi, urlApi } from "@/lib/api";
import FileUploadTab from "@/components/data-sources/FileUploadTab";
import WebPageTab from "@/components/data-sources/WebPageTab";
import YouTubeTab from "@/components/data-sources/YouTubeTab";

interface DataSource {
  id: string;
  title: string;
  type: "file" | "webpage" | "youtube";
  status: "pending" | "processing" | "completed" | "failed";
  fileType?: string;
  progress?: number;
  entities?: number;
  relationships?: number;
  chunks?: number;
  processingTime?: string;
  processingTimeSeconds?: number;
  processingStartTime?: number;
  extractedText?: string;
  thumbnail?: string;
  dateAdded: string;
  totalWords?: number;
  totalChars?: number;
  author?: string;
  language?: string;
  currentChunk?: number;
  totalChunks?: number;
}

// Helper function to format seconds as mm:ss
const formatProcessingTime = (seconds: number): string => {
  if (!seconds || seconds <= 0) return "00:00";
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
};

export default function DataSourcesPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState("all");
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isDetailsModalOpen, setIsDetailsModalOpen] = useState(false);
  const [selectedSource, setSelectedSource] = useState<DataSource | null>(null);
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isContentModalOpen, setIsContentModalOpen] = useState(false);
  const [contentText, setContentText] = useState("");
  const [rawText, setRawText] = useState("");
  const [processedText, setProcessedText] = useState("");
  const [contentType, setContentType] = useState<"raw" | "processed">("raw");
  const [isLoadingContent, setIsLoadingContent] = useState(false);
  const [currentTime, setCurrentTime] = useState(Date.now());
  const [isYouTubeProcessing, setIsYouTubeProcessing] = useState(false);
  const { toast} = useToast();

  // Update timer every second for processing items
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(Date.now());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Initial data load - fetch documents only (async jobs handle their own progress tracking)
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setIsLoading(true);

        // Fetch completed documents
        const documentsResponse = await documentsApi.list({ limit: 100 });

        // Convert to DataSource format
        const sources: DataSource[] = documentsResponse.documents.map(doc => {
          const dateAdded = doc.created_at
            ? new Date(doc.created_at).toLocaleDateString()
            : new Date().toLocaleDateString();

          const processingTime = doc.processing_time_seconds
            ? formatProcessingTime(doc.processing_time_seconds)
            : undefined;

          return {
            id: doc.document_id,
            title: doc.title,
            type: doc.content_type as "file" | "webpage" | "youtube",
            fileType: doc.content_type === 'youtube' ? 'video' : doc.content_type,
            status: "completed",
            progress: 100,
            entities: doc.entity_count,
            relationships: doc.relationship_count,
            totalWords: doc.total_words,
            totalChars: doc.total_chars,
            author: doc.author,
            language: doc.language,
            processingTime: processingTime,
            processingTimeSeconds: doc.processing_time_seconds,
            dateAdded: dateAdded,
            thumbnail: doc.content_type === 'youtube'
              ? `https://img.youtube.com/vi/${doc.document_id.replace('yt_', '')}/mqdefault.jpg`
              : undefined,
          };
        });

        setDataSources(sources);
      } catch (error) {
        console.error('Error loading data sources:', error);
        toast({
          title: "Error",
          description: "Failed to load data sources",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    };

    loadInitialData();
  }, [toast]);

  // Note: Polling removed - async jobs now handle their own progress tracking via WebPageTab/YouTubeTab

  const loadFiles = async () => {
    try {
      const response = await documentsApi.list({ limit: 100 });

      // Convert document data to DataSource format
      const sources: DataSource[] = response.documents.map(doc => {
        const dateAdded = doc.created_at
          ? new Date(doc.created_at).toLocaleDateString()
          : new Date().toLocaleDateString();

        const processingTime = doc.processing_time_seconds
          ? formatProcessingTime(doc.processing_time_seconds)
          : undefined;

        return {
          id: doc.document_id,
          title: doc.title,
          type: doc.content_type as "file" | "webpage" | "youtube",
          fileType: doc.content_type === 'youtube' ? 'video' : doc.content_type,
          status: "completed",
          progress: 100,
          entities: doc.entity_count,
          relationships: doc.relationship_count,
          totalWords: doc.total_words,
          totalChars: doc.total_chars,
          author: doc.author,
          language: doc.language,
          processingTime: processingTime,
          processingTimeSeconds: doc.processing_time_seconds,
          dateAdded: dateAdded,
          thumbnail: doc.content_type === 'youtube'
            ? `https://img.youtube.com/vi/${doc.document_id.replace('yt_', '')}/mqdefault.jpg`
            : undefined,
        };
      });

      // Simply replace all documents (no processing items to merge since jobs handle their own progress)
      setDataSources(sources);
    } catch (error) {
      console.error('Error loading documents:', error);
      toast({
        title: "Error",
        description: "Failed to load documents from server",
        variant: "destructive",
      });
    }
  };

  const handleDelete = async (documentId: string) => {
    try {
      await documentsApi.delete(documentId);
      toast({
        title: "Success",
        description: "Document deleted successfully",
      });
      loadFiles(); // Reload documents
    } catch (error) {
      console.error('Error deleting file:', error);
      toast({
        title: "Error",
        description: "Failed to delete file",
        variant: "destructive",
      });
    }
  };

  const handleReprocess = async (source: DataSource) => {
    try {
      // Show processing toast
      toast({
        title: "Processing",
        description: `Reprocessing ${source.title}...`,
      });

      // Update source status to processing with start time
      setDataSources(prev => prev.map(s =>
        s.id === source.id ? { ...s, status: "processing" as const, progress: 0, processingStartTime: Date.now() } : s
      ));

      if (source.type === "youtube") {
        // Extract video ID from document ID (format: yt_{video_id})
        const videoId = source.id.replace('yt_', '');
        const url = `https://www.youtube.com/watch?v=${videoId}`;

        // Delete old data first
        await documentsApi.delete(source.id);

        // Reprocess
        await urlApi.process(url);

        toast({
          title: "Success",
          description: "YouTube video reprocessed successfully",
        });
      } else if (source.type === "webpage") {
        // For webpages, we need to store the URL in the backend
        // For now, show an error
        toast({
          title: "Not Implemented",
          description: "Webpage reprocessing requires the original URL to be stored",
          variant: "destructive",
        });
        return;
      } else {
        // For files, we need the original file
        toast({
          title: "Not Supported",
          description: "File reprocessing is not yet supported. Please re-upload the file.",
          variant: "destructive",
        });
        return;
      }

      // Reload documents
      await loadFiles();
    } catch (error: any) {
      console.error('Error reprocessing:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to reprocess document",
        variant: "destructive",
      });

      // Reload to get correct status
      loadFiles();
    }
  };

  const handleViewDetails = (source: DataSource) => {
    setSelectedSource(source);
    setIsDetailsModalOpen(true);
  };

  const handleViewContent = async (documentId: string, type: "raw" | "processed" = "raw") => {
    setIsLoadingContent(true);
    setIsContentModalOpen(true);
    setContentType(type);

    try {
      const response = await documentsApi.getContent(documentId);
      setRawText(response.full_text || "");
      setProcessedText(response.processed_text || "");

      // Set the content based on the type requested
      if (type === "raw") {
        setContentText(response.full_text || "");
      } else {
        setContentText(response.processed_text || response.full_text || "");
      }
    } catch (error: any) {
      console.error('Error fetching content:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to fetch document content",
        variant: "destructive",
      });
      setIsContentModalOpen(false);
    } finally {
      setIsLoadingContent(false);
    }
  };

  const getStatusIcon = (status: DataSource["status"]) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="w-4 h-4 text-accent" />;
      case "processing":
        return <Loader2 className="w-4 h-4 text-primary animate-spin" />;
      case "pending":
        return <Clock className="w-4 h-4 text-muted-foreground" />;
      case "failed":
        return <XCircle className="w-4 h-4 text-destructive" />;
    }
  };

  const getFileIcon = (fileType?: string) => {
    switch (fileType) {
      case "pdf":
        return <FileText className="w-5 h-5 text-destructive" />;
      case "json":
        return <FileJson className="w-5 h-5 text-primary" />;
      case "html":
        return <FileCode className="w-5 h-5 text-accent" />;
      default:
        return <FileText className="w-5 h-5" />;
    }
  };

  const getTypeIcon = (type: DataSource["type"]) => {
    switch (type) {
      case "file":
        return <Upload className="w-4 h-4" />;
      case "webpage":
        return <Globe className="w-4 h-4" />;
      case "youtube":
        return <Youtube className="w-4 h-4" />;
    }
  };

  const filteredSources = dataSources.filter((source) => {
    const matchesSearch = source.title.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesTab = activeTab === "all" || source.type === activeTab;
    return matchesSearch && matchesTab;
  });

  return (
    <div className="space-y-6 animate-fade-in pb-20 md:pb-0">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">Data Sources</h1>
          <p className="text-muted-foreground">Manage your uploaded files, web pages, and videos</p>
        </div>
        
        <Dialog open={isAddModalOpen} onOpenChange={isYouTubeProcessing ? undefined : setIsAddModalOpen}>
          <DialogTrigger asChild>
            <Button className="gap-2">
              <Upload className="w-4 h-4" />
              Add Source
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl" hideClose={isYouTubeProcessing}>
            <DialogHeader>
              <DialogTitle>Add New Data Source</DialogTitle>
            </DialogHeader>

            <Tabs defaultValue="file" className="mt-4">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="file" className="gap-2">
                  <Upload className="w-4 h-4" />
                  Files
                </TabsTrigger>
                <TabsTrigger value="webpage" className="gap-2">
                  <Globe className="w-4 h-4" />
                  Web Page
                </TabsTrigger>
                <TabsTrigger value="youtube" className="gap-2">
                  <Youtube className="w-4 h-4" />
                  YouTube
                </TabsTrigger>
              </TabsList>

              <TabsContent value="file" className="mt-4 min-h-[400px]">
                <FileUploadTab onClose={() => { setIsAddModalOpen(false); loadFiles(); }} />
              </TabsContent>

              <TabsContent value="webpage" className="mt-4 min-h-[400px]">
                <WebPageTab onClose={() => { setIsAddModalOpen(false); loadFiles(); }} />
              </TabsContent>

              <TabsContent value="youtube" className="mt-4 min-h-[400px]">
                <YouTubeTab
                  onClose={() => { setIsAddModalOpen(false); loadFiles(); }}
                  onProcessingChange={setIsYouTubeProcessing}
                />
              </TabsContent>
            </Tabs>
          </DialogContent>
        </Dialog>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search data sources..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="all">All Sources</TabsTrigger>
          <TabsTrigger value="file">Files</TabsTrigger>
          <TabsTrigger value="webpage">Web Pages</TabsTrigger>
          <TabsTrigger value="youtube">YouTube</TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="mt-6">
          {/* Show loading spinner while fetching initial data */}
          {isLoading ? (
            <div className="text-center py-12">
              <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite]" role="status">
                <span className="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">Loading...</span>
              </div>
              <p className="mt-4 text-muted-foreground">Loading data sources...</p>
            </div>
          ) : (
            <>
              {/* Show items */}
              {filteredSources.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredSources.map((source) => (
              <Card key={source.id} className="p-6 hover:shadow-lg transition-shadow">
                <div className="space-y-4">
                  {/* Header */}
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-start gap-3 flex-1 min-w-0">
                      {source.type === "file" ? (
                        getFileIcon(source.fileType)
                      ) : source.type === "youtube" && source.thumbnail ? (
                        <img src={source.thumbnail} alt="" className="w-12 h-12 rounded object-cover" />
                      ) : (
                        getTypeIcon(source.type)
                      )}
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-sm truncate">{source.title}</h3>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant="outline" className="text-xs">
                            {source.type}
                          </Badge>
                          {getStatusIcon(source.status)}
                        </div>
                      </div>
                    </div>
                    
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                          <MoreVertical className="w-4 h-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem className="gap-2" onClick={() => handleViewDetails(source)}>
                          <Eye className="w-4 h-4" />
                          View Details
                        </DropdownMenuItem>
                        <DropdownMenuItem className="gap-2" onClick={() => handleReprocess(source)}>
                          <RefreshCw className="w-4 h-4" />
                          Reprocess
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          className="gap-2 text-destructive"
                          onClick={() => handleDelete(source.id)}
                        >
                          <Trash2 className="w-4 h-4" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>

                  {/* Progress Bar with Timer */}
                  {source.status === "processing" && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>
                          {source.currentChunk && source.totalChunks
                            ? `Processing chunk ${source.currentChunk}/${source.totalChunks}`
                            : 'Processing...'}
                        </span>
                        <span>{source.processingStartTime ? formatProcessingTime(Math.floor((currentTime - source.processingStartTime) / 1000)) : '00:00'}</span>
                      </div>
                      <Progress value={source.progress} className="h-2" />
                    </div>
                  )}

                  {/* Metrics */}
                  {source.status === "completed" && (
                    <>
                      <div className="grid grid-cols-3 gap-2 pt-2 border-t">
                        <div className="text-center">
                          <p className="text-lg font-bold" style={{ color: 'hsl(150, 55%, 50%)' }}>{source.entities || 0}</p>
                          <p className="text-xs text-muted-foreground">Entities</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold" style={{ color: 'hsl(165, 50%, 48%)' }}>{source.relationships || 0}</p>
                          <p className="text-xs text-muted-foreground">Relations</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold" style={{ color: 'hsl(24, 75%, 55%)' }}>
                            {source.totalWords ? (source.totalWords / 1000).toFixed(1) + 'k' : (source.chunks || 0)}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {source.totalWords ? 'Words' : 'Chunks'}
                          </p>
                        </div>
                      </div>

                      {/* Additional Details for YouTube/Webpages */}
                      {(source.type === "youtube" || source.type === "webpage") && (source.totalWords || source.processingTimeSeconds) && (
                        <div className="text-xs text-muted-foreground space-y-1 pt-2 border-t">
                          {source.author && (
                            <div className="flex justify-between">
                              <span>Author:</span>
                              <span className="font-medium truncate ml-2">{source.author}</span>
                            </div>
                          )}
                          {source.language && (
                            <div className="flex justify-between">
                              <span>Language:</span>
                              <span className="font-medium uppercase">{source.language}</span>
                            </div>
                          )}
                          {source.totalWords && (
                            <div className="flex justify-between">
                              <span>Content:</span>
                              <span className="font-medium">{source.totalWords.toLocaleString()} words</span>
                            </div>
                          )}
                          {source.processingTimeSeconds && source.processingTimeSeconds > 0 && (
                            <div className="flex justify-between">
                              <span>Processing Time:</span>
                              <span className="font-medium">{formatProcessingTime(source.processingTimeSeconds)}</span>
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  )}

                  {/* Error Message */}
                  {source.status === "failed" && (
                    <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20">
                      <p className="text-xs text-destructive">Processing failed. Please try again.</p>
                    </div>
                  )}

                  {/* Footer */}
                  <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
                    <span>{source.dateAdded}</span>
                    {source.status === "completed" && source.processingTime && (
                      <span className="font-medium">
                        <Clock className="w-3 h-3 inline mr-1" />
                        {source.processingTime}
                      </span>
                    )}
                  </div>
                </div>
              </Card>
                ))}
              </div>
              )}

              {/* Show empty state only after loading completes AND we have no items */}
              {filteredSources.length === 0 && (
                <div className="text-center py-12">
                  <FileText className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No data sources found</h3>
                  <p className="text-muted-foreground mb-4">
                    {searchQuery ? "Try a different search term" : "Add your first data source to get started"}
                  </p>
                  <Button onClick={() => setIsAddModalOpen(true)}>
                    <Upload className="w-4 h-4 mr-2" />
                    Add Source
                  </Button>
                </div>
              )}
            </>
          )}
        </TabsContent>
      </Tabs>

      {/* Details Modal */}
      <Dialog open={isDetailsModalOpen} onOpenChange={setIsDetailsModalOpen}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Document Details</DialogTitle>
          </DialogHeader>

          {selectedSource && (
            <div className="space-y-6">
              {/* Title and Type */}
              <div>
                <h3 className="text-lg font-semibold mb-2">{selectedSource.title}</h3>
                <div className="flex gap-2">
                  <Badge variant="outline">{selectedSource.type}</Badge>
                  {selectedSource.language && (
                    <Badge variant="outline">{selectedSource.language.toUpperCase()}</Badge>
                  )}
                  <Badge variant={selectedSource.status === "completed" ? "default" : "secondary"}>
                    {selectedSource.status}
                  </Badge>
                </div>
              </div>

              {/* Thumbnail for YouTube */}
              {selectedSource.type === "youtube" && selectedSource.thumbnail && (
                <img
                  src={selectedSource.thumbnail}
                  alt={selectedSource.title}
                  className="w-full rounded-lg"
                />
              )}

              {/* Statistics */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 p-4 bg-secondary/50 rounded-lg">
                <div>
                  <p className="text-sm text-muted-foreground">Entities</p>
                  <p className="text-2xl font-bold" style={{ color: 'hsl(150, 55%, 50%)' }}>{selectedSource.entities || 0}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Relationships</p>
                  <p className="text-2xl font-bold" style={{ color: 'hsl(165, 50%, 48%)' }}>{selectedSource.relationships || 0}</p>
                </div>
                {selectedSource.totalWords && (
                  <div>
                    <p className="text-sm text-muted-foreground">Words</p>
                    <p className="text-2xl font-bold" style={{ color: 'hsl(24, 75%, 55%)' }}>
                      {selectedSource.totalWords.toLocaleString()}
                    </p>
                  </div>
                )}
                {selectedSource.totalChars && (
                  <div>
                    <p className="text-sm text-muted-foreground">Characters</p>
                    <p className="text-2xl font-bold">{selectedSource.totalChars.toLocaleString()}</p>
                  </div>
                )}
                {selectedSource.chunks !== undefined && (
                  <div>
                    <p className="text-sm text-muted-foreground">Chunks</p>
                    <p className="text-2xl font-bold">{selectedSource.chunks}</p>
                  </div>
                )}
                {selectedSource.processingTimeSeconds && selectedSource.processingTimeSeconds > 0 && (
                  <div>
                    <p className="text-sm text-muted-foreground">Processing Time</p>
                    <p className="text-2xl font-bold" style={{ color: 'hsl(280, 60%, 60%)' }}>
                      {formatProcessingTime(selectedSource.processingTimeSeconds)}
                    </p>
                  </div>
                )}
                <div>
                  <p className="text-sm text-muted-foreground">Date Added</p>
                  <p className="text-sm font-medium">{selectedSource.dateAdded}</p>
                </div>
              </div>

              {/* Additional Metadata */}
              {selectedSource.author && (
                <div className="p-4 bg-secondary/30 rounded-lg">
                  <p className="text-sm text-muted-foreground mb-1">Author</p>
                  <p className="font-medium">{selectedSource.author}</p>
                </div>
              )}

              {/* Document ID */}
              <div className="p-4 bg-secondary/30 rounded-lg">
                <p className="text-sm text-muted-foreground mb-1">Document ID</p>
                <p className="font-mono text-sm">{selectedSource.id}</p>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col gap-2">
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    className="flex-1"
                    onClick={() => {
                      window.open(`/graph?document=${selectedSource.id}`, '_blank');
                    }}
                  >
                    View in Graph
                  </Button>
                </div>
                {(selectedSource.type === 'youtube' || selectedSource.type === 'webpage') && (
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      className="flex-1"
                      onClick={() => handleViewContent(selectedSource.id, 'raw')}
                    >
                      Raw Content
                    </Button>
                    <Button
                      variant="outline"
                      className="flex-1"
                      onClick={() => handleViewContent(selectedSource.id, 'processed')}
                    >
                      Processed Content
                    </Button>
                  </div>
                )}
                <Button
                  variant="destructive"
                  onClick={() => {
                    setIsDetailsModalOpen(false);
                    handleDelete(selectedSource.id);
                  }}
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Content Display Modal */}
      <Dialog open={isContentModalOpen} onOpenChange={setIsContentModalOpen}>
        <DialogContent className="max-w-4xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle>
              {selectedSource?.type === 'youtube' ? 'Video Transcript' : 'Content'} - {contentType === 'raw' ? 'Raw' : 'Processed'}
            </DialogTitle>
          </DialogHeader>

          {/* Content Type Toggle Buttons */}
          <div className="flex gap-2 mb-2">
            <Button
              variant={contentType === 'raw' ? 'default' : 'outline'}
              size="sm"
              onClick={() => {
                setContentType('raw');
                setContentText(rawText);
              }}
              disabled={isLoadingContent}
            >
              Raw Content
            </Button>
            <Button
              variant={contentType === 'processed' ? 'default' : 'outline'}
              size="sm"
              onClick={() => {
                setContentType('processed');
                setContentText(processedText || rawText);
              }}
              disabled={isLoadingContent}
            >
              Processed Content
            </Button>
          </div>

          <div className="overflow-y-auto max-h-[60vh] p-4 bg-secondary/30 rounded-lg">
            {isLoadingContent ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              </div>
            ) : (
              <div className="whitespace-pre-wrap text-sm leading-relaxed">
                {contentText || "No content available"}
              </div>
            )}
          </div>

          <div className="text-sm text-muted-foreground mt-2">
            {contentText && `${contentText.split(/\s+/).length.toLocaleString()} words`}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
