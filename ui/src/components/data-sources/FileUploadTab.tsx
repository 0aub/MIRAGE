import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Button } from "@/components/ui/button";
import { Upload, FileText, FileJson, FileCode } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface FileUploadTabProps {
  onClose: () => void;
}

export default function FileUploadTab({ onClose }: FileUploadTabProps) {
  const { toast } = useToast();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    console.log("Files dropped:", acceptedFiles);
    toast({
      title: "Files uploaded",
      description: `${acceptedFiles.length} file(s) uploaded successfully and queued for processing.`,
    });
    onClose();
  }, [onClose, toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/json': ['.json'],
      'text/html': ['.html'],
      'text/plain': ['.txt'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
  });

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors
          ${isDragActive 
            ? 'border-primary bg-primary/5' 
            : 'border-border hover:border-primary/50 hover:bg-secondary/50'
          }
        `}
      >
        <input {...getInputProps()} />
        <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
        {isDragActive ? (
          <p className="text-lg font-medium">Drop files here...</p>
        ) : (
          <>
            <p className="text-lg font-medium mb-2">Drag & drop files here</p>
            <p className="text-sm text-muted-foreground mb-4">or click to browse</p>
            <Button variant="outline" size="sm">
              Select Files
            </Button>
          </>
        )}
      </div>

      <div className="space-y-2">
        <h4 className="text-sm font-medium">Supported file types:</h4>
        <div className="grid grid-cols-2 gap-2">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <FileText className="w-4 h-4 text-destructive" />
            PDF Documents
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <FileJson className="w-4 h-4 text-primary" />
            JSON Files
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <FileCode className="w-4 h-4 text-accent" />
            HTML Files
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <FileText className="w-4 h-4" />
            Text & DOCX
          </div>
        </div>
      </div>
    </div>
  );
}
