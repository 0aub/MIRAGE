import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Globe, Loader2, ExternalLink, CheckCircle2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { urlApi } from "@/lib/api";

interface WebPageTabProps {
  onClose: () => void;
}

export default function WebPageTab({ onClose }: WebPageTabProps) {
  const [url, setUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [preview, setPreview] = useState<any>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentPhase, setCurrentPhase] = useState<string>("");
  const { toast} = useToast();
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  // Poll job status
  const pollJobStatus = async (jid: string) => {
    try {
      const status = await urlApi.getJobStatus(jid);

      setProgress(status.progress);
      setCurrentPhase(status.current_phase || "");

      if (status.status === "completed") {
        // Job completed, stop polling
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }

        // Get final result
        try {
          const result = await urlApi.getJobResult(jid);
          toast({
            title: "Success!",
            description: `Web page processed successfully. ${result.phase2.entities_extracted} entities extracted.`,
          });
          setIsProcessing(false);
          onClose();
        } catch (error: any) {
          toast({
            title: "Error",
            description: "Failed to get processing result",
            variant: "destructive",
          });
          setIsProcessing(false);
        }
      } else if (status.status === "failed") {
        // Job failed, stop polling
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }

        toast({
          title: "Processing Failed",
          description: status.error || "Unknown error occurred",
          variant: "destructive",
        });
        setIsProcessing(false);
      }
    } catch (error: any) {
      console.error("Error polling job status:", error);
      // Continue polling even on error (might be temporary)
    }
  };

  const handleFetchContent = async () => {
    if (!url) {
      toast({
        title: "URL required",
        description: "Please enter a valid URL",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setPreview(null);

    try {
      const response = await urlApi.preview(url);
      setPreview({
        title: response.title || "Untitled",
        description: response.description || "No description available",
        content_preview: response.content_preview,
        estimated_words: response.estimated_words,
        content_type: response.content_type,
      });
      toast({
        title: "Content fetched",
        description: `Found ${response.estimated_words} words`,
      });
    } catch (error: any) {
      console.error('Error fetching URL:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to fetch content from URL",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleAdd = async () => {
    if (!url || !preview) return;

    setIsProcessing(true);
    setProgress(0);
    setCurrentPhase("queued");

    try {
      // Submit job for async processing
      const response = await urlApi.processAsync(url);
      setJobId(response.job_id);

      toast({
        title: "Processing Started",
        description: "Your web page is being processed in the background",
      });

      // Start polling for status
      pollIntervalRef.current = setInterval(() => {
        pollJobStatus(response.job_id);
      }, 2000); // Poll every 2 seconds

    } catch (error: any) {
      console.error('Error processing URL:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to process web page",
        variant: "destructive",
      });
      setIsProcessing(false);
    }
  };

  const getPhaseLabel = (phase: string) => {
    const labels: Record<string, string> = {
      "fetching": "Fetching content",
      "chunking": "Chunking text",
      "rewriting": "Rewriting content",
      "extracting": "Extracting entities",
      "storing_graph": "Storing graph",
      "compressing": "Compressing",
      "storing_vector": "Storing vectors",
      "finalizing": "Finalizing",
      "queued": "Queued"
    };
    return labels[phase] || phase;
  };

  return (
    <div className="space-y-4 min-h-[400px] flex flex-col">
      <div className="space-y-2">
        <label className="text-sm font-medium">Web Page URL</label>
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Globe className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              type="url"
              placeholder="https://example.com/article"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="pl-9"
              onKeyDown={(e) => e.key === 'Enter' && !isLoading && url && handleFetchContent()}
              disabled={isProcessing}
            />
          </div>
          <Button onClick={handleFetchContent} disabled={isLoading || !url || isProcessing}>
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Fetching...
              </>
            ) : (
              "Fetch Content"
            )}
          </Button>
        </div>
      </div>

      {/* Empty State - when no content is loaded */}
      {!preview && !isLoading && !isProcessing && (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 border-2 border-dashed rounded-lg border-border">
          <div className="w-16 h-16 rounded-full bg-blue-500/10 flex items-center justify-center mb-4">
            <Globe className="w-8 h-8 text-blue-500" />
          </div>
          <h4 className="text-lg font-semibold mb-2">Extract Knowledge from Web Pages</h4>
          <p className="text-sm text-muted-foreground mb-4 max-w-md">
            Enter a web page URL above to extract and process content. The system automatically removes navigation, ads, and focuses on the main content.
          </p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle2 className="w-4 h-4 text-green-500" />
            <span>Smart content extraction</span>
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

      {/* Processing Status */}
      {isProcessing && (
        <Card className="p-4 bg-secondary/50">
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <Loader2 className="w-5 h-5 animate-spin text-primary" />
              <div className="flex-1">
                <p className="text-sm font-medium">Processing...</p>
                <p className="text-xs text-muted-foreground">{getPhaseLabel(currentPhase)}</p>
              </div>
              <span className="text-sm font-medium">{progress}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
        </Card>
      )}

      {preview && !isProcessing && (
        <Card className="p-4 bg-secondary/50">
          <div className="flex items-start gap-3">
            <Globe className="w-8 h-8 text-primary flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <h4 className="font-semibold mb-1 flex items-center gap-2">
                {preview.title}
                <a href={url} target="_blank" rel="noopener noreferrer" className="inline-flex">
                  <ExternalLink className="w-3 h-3" />
                </a>
              </h4>
              <p className="text-sm text-muted-foreground line-clamp-2">{preview.description}</p>
              <p className="text-xs text-muted-foreground mt-1">
                ~{preview.estimated_words} words • {preview.content_type}
              </p>
            </div>
          </div>

          <div className="mt-4 p-3 rounded bg-background border">
            <h5 className="text-xs font-medium mb-2">Content Preview</h5>
            <p className="text-xs text-muted-foreground whitespace-pre-wrap">
              {preview.content_preview}
            </p>
          </div>

          <Button onClick={handleAdd} className="w-full mt-4" disabled={isProcessing}>
            {isProcessing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Processing ({progress}%)
              </>
            ) : (
              "Add to Data Sources"
            )}
          </Button>
        </Card>
      )}
    </div>
  );
}
