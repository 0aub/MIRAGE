import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Youtube, Loader2, ExternalLink, CheckCircle2, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { urlApi } from "@/lib/api";

interface YouTubeTabProps {
  onClose: () => void;
  onProcessingChange?: (isProcessing: boolean) => void;
}

type ProcessingStage = 'idle' | 'rewriting' | 'extraction' | 'completed' | 'error';

const STAGE_INFO: Record<ProcessingStage, { label: string; description: string }> = {
  idle: { label: 'Ready', description: 'Ready to process' },
  rewriting: {
    label: 'Preparing Content',
    description: 'Chunking and rewriting transcript content...'
  },
  extraction: {
    label: 'Generating Graph',
    description: 'Extracting entities and relationships...'
  },
  completed: { label: 'Completed', description: 'Processing completed successfully!' },
  error: { label: 'Error', description: 'Processing failed' },
};

export default function YouTubeTab({ onClose, onProcessingChange }: YouTubeTabProps) {
  const [url, setUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [videoInfo, setVideoInfo] = useState<any>(null);
  const [processingStage, setProcessingStage] = useState<ProcessingStage>('idle');
  const [processingProgress, setProcessingProgress] = useState(0);
  const [currentChunk, setCurrentChunk] = useState(0);
  const [totalChunks, setTotalChunks] = useState(0);
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [entitiesExtracted, setEntitiesExtracted] = useState(0);
  const [processingStartTime, setProcessingStartTime] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [totalProcessingTime, setTotalProcessingTime] = useState(0);
  const { toast } = useToast();

  // Helper function to format seconds into readable time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins > 0) {
      return `${mins}m ${secs}s`;
    }
    return `${secs}s`;
  };

  // Update elapsed time every second during processing
  useEffect(() => {
    if (!isProcessing || !processingStartTime) {
      return;
    }

    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() / 1000) - processingStartTime);
      setElapsedSeconds(elapsed);
    }, 1000);

    return () => clearInterval(interval);
  }, [isProcessing, processingStartTime]);

  // Notify parent about processing state changes
  useEffect(() => {
    if (onProcessingChange) {
      onProcessingChange(isProcessing);
    }
  }, [isProcessing, onProcessingChange]);

  // Poll processing status when processing
  useEffect(() => {
    if (!isProcessing || !documentId) return;

    const pollStatus = async () => {
      try {
        const response = await urlApi.getProcessingStatus();
        const item = response.processing.find(p => p.document_id === documentId);

        if (item) {
          // Update chunk progress
          setCurrentChunk(item.current_chunk || 0);
          setTotalChunks(item.total_chunks || 0);

          // Set stage based on phase from backend
          const phase = item.phase || 'rewriting';
          if (phase === 'rewriting') {
            setProcessingStage('rewriting');
            if (item.total_chunks && item.total_chunks > 0) {
              const progress = ((item.current_chunk || 0) / item.total_chunks) * 100;
              setProcessingProgress(progress);
            }
          } else if (phase === 'extraction') {
            setProcessingStage('extraction');
            if (item.total_chunks && item.total_chunks > 0) {
              const progress = ((item.current_chunk || 0) / item.total_chunks) * 100;
              setProcessingProgress(progress);
            }
          }
        } else {
          // Processing completed - item no longer in processing list
          const totalTime = processingStartTime ? Math.floor((Date.now() / 1000) - processingStartTime) : 0;
          setTotalProcessingTime(totalTime); // Store total processing time
          setProcessingStage('completed');
          setProcessingProgress(100);
          // Auto-close modal after showing success
          setTimeout(() => {
            setIsProcessing(false);
            onClose();
          }, 2000);
        }
      } catch (error) {
        console.error('Error polling status:', error);
      }
    };

    // Poll immediately, then every 2 seconds
    pollStatus();
    const interval = setInterval(pollStatus, 2000);
    return () => clearInterval(interval);
  }, [isProcessing, documentId, onClose]);

  const extractVideoId = (url: string) => {
    const regex = /(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]*)/;
    const match = url.match(regex);
    return match ? match[1] : null;
  };

  const handleFetchInfo = async () => {
    const videoId = extractVideoId(url);
    if (!videoId) {
      toast({
        title: "Invalid URL",
        description: "Please enter a valid YouTube URL",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setVideoInfo(null);

    try {
      const response = await urlApi.preview(url);

      if (response.content_type !== 'youtube') {
        toast({
          title: "Invalid URL",
          description: "This doesn't appear to be a YouTube URL",
          variant: "destructive",
        });
        setIsLoading(false);
        return;
      }

      setVideoInfo({
        id: videoId,
        title: response.title || "YouTube Video",
        thumbnail: `https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`,
        description: response.description,
        estimated_words: response.estimated_words,
        content_preview: response.content_preview,
      });

      // Removed annoying notification - user can see video info in the card
    } catch (error: any) {
      console.error('Error fetching YouTube video:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to fetch YouTube video. Make sure it has captions/subtitles available.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleGetTranscript = async () => {
    if (!url || !videoInfo) return;

    setIsProcessing(true);
    setProcessingStage('rewriting'); // Start with first phase
    setProcessingProgress(0);
    setCurrentChunk(0);
    setTotalChunks(0);
    setProcessingStartTime(Date.now() / 1000); // Record start time
    setElapsedSeconds(0); // Reset elapsed time
    setTotalProcessingTime(0); // Reset total time

    // Extract video ID to use as document ID
    const videoId = extractVideoId(url);
    const docId = `yt_${videoId}`;
    setDocumentId(docId);

    try {
      // Start processing (this is async and will take time)
      // The polling useEffect will track chunk progress automatically
      const response = await urlApi.process(url);

      // If we get here, processing completed successfully
      setEntitiesExtracted(response.phase2?.entities_extracted || 0);
      setProcessingStage('completed');
      setProcessingProgress(100);

      toast({
        title: "Success",
        description: `YouTube video processed successfully. ${response.phase2?.entities_extracted || 0} entities extracted from transcript.`,
      });

      // Auto-close after showing success
      setTimeout(() => {
        setIsProcessing(false);
        onClose();
      }, 2000);
    } catch (error: any) {
      console.error('Error processing YouTube video:', error);
      setProcessingStage('error');
      toast({
        title: "Error",
        description: error.message || "Failed to process YouTube video",
        variant: "destructive",
      });
      setIsProcessing(false);
    }
  };

  const stageInfo = STAGE_INFO[processingStage];

  return (
    <div className="space-y-4 min-h-[400px] flex flex-col">
      <div className="space-y-2">
        <label className="text-sm font-medium">YouTube URL</label>
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Youtube className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              type="url"
              placeholder="https://www.youtube.com/watch?v=..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="pl-9"
              disabled={isProcessing}
              onKeyDown={(e) => e.key === 'Enter' && !isLoading && !isProcessing && handleFetchInfo()}
            />
          </div>
          <Button onClick={handleFetchInfo} disabled={isLoading || isProcessing || !url}>
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Loading...
              </>
            ) : (
              "Load Video"
            )}
          </Button>
        </div>
      </div>

      {/* Empty State - when no video is loaded */}
      {!videoInfo && !isLoading && (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 border-2 border-dashed rounded-lg border-border">
          <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center mb-4">
            <Youtube className="w-8 h-8 text-red-500" />
          </div>
          <h4 className="text-lg font-semibold mb-2">Extract Knowledge from YouTube Videos</h4>
          <p className="text-sm text-muted-foreground mb-4 max-w-md">
            Enter a YouTube URL above to extract and process video transcripts. The system will identify entities, concepts, and relationships from the content.
          </p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle2 className="w-4 h-4 text-green-500" />
            <span>Automatic transcript extraction</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
            <CheckCircle2 className="w-4 h-4 text-green-500" />
            <span>Content rewriting and cleaning</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
            <CheckCircle2 className="w-4 h-4 text-green-500" />
            <span>Knowledge graph generation</span>
          </div>
        </div>
      )}

      {videoInfo && (
        <Card className="p-4 bg-secondary/50">
          <div className="flex gap-4 mb-4">
            <img
              src={videoInfo.thumbnail}
              alt={videoInfo.title}
              className="w-40 h-24 rounded object-cover"
              onError={(e) => {
                e.currentTarget.src = "/placeholder.svg";
              }}
            />
            <div className="flex-1 min-w-0">
              <h4 className="font-semibold mb-1 flex items-center gap-2">
                {videoInfo.title}
                <a href={url} target="_blank" rel="noopener noreferrer" className="inline-flex">
                  <ExternalLink className="w-3 h-3" />
                </a>
              </h4>
              <p className="text-sm text-muted-foreground mb-2 line-clamp-2">{videoInfo.description}</p>
              <p className="text-xs text-muted-foreground">
                ~{videoInfo.estimated_words} words in transcript
              </p>
            </div>
          </div>

          {videoInfo.content_preview && !isProcessing && (
            <div className="mt-4 p-3 rounded bg-background border">
              <h5 className="text-xs font-medium mb-2">Transcript Preview</h5>
              <p className="text-xs text-muted-foreground whitespace-pre-wrap">
                {videoInfo.content_preview}
              </p>
            </div>
          )}

          {/* Processing Progress Card */}
          {isProcessing && stageInfo && (
            <Card className="p-6 bg-secondary/50 border-primary/20 mt-4">
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  {processingStage === 'completed' ? (
                    <CheckCircle2 className="w-6 h-6 text-green-500" />
                  ) : processingStage === 'error' ? (
                    <AlertCircle className="w-6 h-6 text-destructive" />
                  ) : (
                    <Loader2 className="w-6 h-6 animate-spin text-primary" />
                  )}
                  <div className="flex-1">
                    <h4 className="font-semibold">{stageInfo.label}</h4>
                    <p className="text-sm text-muted-foreground">{stageInfo.description}</p>
                  </div>
                </div>

                {/* Elapsed time counter during processing */}
                {processingStage !== 'completed' && processingStage !== 'error' && elapsedSeconds > 0 && (
                  <div className="text-xs text-muted-foreground">
                    Processing for: {formatTime(elapsedSeconds)}
                  </div>
                )}

                {/* Progress bar for rewriting phase (with chunk count) */}
                {processingStage === 'rewriting' && totalChunks > 0 && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Processing chunk {currentChunk} of {totalChunks}</span>
                      <span>{Math.round(processingProgress)}%</span>
                    </div>
                    <Progress value={processingProgress} className="h-2" />
                  </div>
                )}

                {/* Progress bar for extraction phase (with chunk count) */}
                {processingStage === 'extraction' && totalChunks > 0 && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Processing chunk {currentChunk} of {totalChunks}</span>
                      <span>{Math.round(processingProgress)}%</span>
                    </div>
                    <Progress value={processingProgress} className="h-2" />
                  </div>
                )}

                {/* Success message */}
                {processingStage === 'completed' && (
                  <div className="space-y-1">
                    {entitiesExtracted > 0 && (
                      <div className="text-sm text-green-600 dark:text-green-400">
                        Successfully extracted {entitiesExtracted} entities!
                      </div>
                    )}
                    {totalProcessingTime > 0 && (
                      <div className="text-xs text-muted-foreground">
                        Completed in: {formatTime(totalProcessingTime)}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </Card>
          )}

          {!isProcessing && (
            <Button onClick={handleGetTranscript} className="w-full mt-4">
              Get Transcript & Process
            </Button>
          )}
        </Card>
      )}
    </div>
  );
}
