import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
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
  const { toast } = useToast();

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

    try {
      const response = await urlApi.process(url);
      toast({
        title: "Success",
        description: `Web page processed successfully. ${response.phase2.entities_extracted} entities extracted.`,
      });
      onClose();
    } catch (error: any) {
      console.error('Error processing URL:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to process web page",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
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
            />
          </div>
          <Button onClick={handleFetchContent} disabled={isLoading || !url}>
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
      {!preview && !isLoading && (
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

      {preview && (
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
                Processing...
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
