import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Upload, FileText, CheckCircle2, XCircle, Clock } from "lucide-react";
import { toast } from "sonner";

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  metadata?: {
    pages?: number;
    entities?: number;
    language?: string;
  };
}

export default function UploadPage() {
  const [files, setFiles] = useState<UploadedFile[]>([]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = acceptedFiles.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      status: "pending",
      progress: 0,
    }));
    setFiles((prev) => [...prev, ...newFiles]);
    toast.success(`${acceptedFiles.length} file(s) added`);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/json": [".json"],
      "text/html": [".html"],
    },
  });

  const handleStartProcessing = () => {
    toast.success("Processing started!");
    // Simulate processing
    files.forEach((file, index) => {
      setTimeout(() => {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === file.id
              ? {
                  ...f,
                  status: "processing",
                  progress: 50,
                }
              : f
          )
        );
      }, index * 1000);

      setTimeout(() => {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === file.id
              ? {
                  ...f,
                  status: "completed",
                  progress: 100,
                  metadata: {
                    pages: Math.floor(Math.random() * 50) + 5,
                    entities: Math.floor(Math.random() * 100) + 20,
                    language: "ar",
                  },
                }
              : f
          )
        );
      }, index * 1000 + 3000);
    });
  };

  const getStatusIcon = (status: UploadedFile["status"]) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="w-5 h-5 text-accent" />;
      case "failed":
        return <XCircle className="w-5 h-5 text-destructive" />;
      case "processing":
        return <Clock className="w-5 h-5 text-primary animate-spin" />;
      default:
        return <FileText className="w-5 h-5 text-muted-foreground" />;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <div className="space-y-6 animate-fade-in pb-20 md:pb-0">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">Upload Documents</h1>
          <p className="text-muted-foreground">Upload PDF, JSON, or HTML files for processing</p>
        </div>
        {files.length > 0 && (
          <Button onClick={handleStartProcessing} className="gap-2">
            <Upload className="w-4 h-4" />
            Start Processing
          </Button>
        )}
      </div>

      {/* Upload Area */}
      <Card className="p-12 border-2 border-dashed transition-colors hover:border-primary">
        <div
          {...getRootProps()}
          className="text-center cursor-pointer space-y-4"
        >
          <input {...getInputProps()} />
          <div className="flex justify-center">
            <div className="p-6 rounded-full bg-primary/10">
              <Upload className="w-12 h-12 text-primary" />
            </div>
          </div>
          <div>
            <p className="text-lg font-medium mb-2">
              {isDragActive ? "Drop files here" : "Drag & drop files here"}
            </p>
            <p className="text-sm text-muted-foreground">
              or click to browse • Supports PDF, JSON, HTML
            </p>
          </div>
        </div>
      </Card>

      {/* File List */}
      {files.length > 0 && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Uploaded Files ({files.length})</h2>
          <div className="space-y-4">
            {files.map((file) => (
              <div key={file.id} className="p-4 rounded-lg border border-border bg-card">
                <div className="flex items-start gap-4">
                  {getStatusIcon(file.status)}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-2">
                      <p className="font-medium truncate">{file.name}</p>
                      <span className="text-sm text-muted-foreground ml-2">
                        {formatFileSize(file.size)}
                      </span>
                    </div>
                    {file.status === "processing" && (
                      <Progress value={file.progress} className="h-2 mb-2" />
                    )}
                    {file.metadata && (
                      <div className="flex gap-4 text-xs text-muted-foreground">
                        <span>{file.metadata.pages} pages</span>
                        <span>{file.metadata.entities} entities</span>
                        <span>Language: {file.metadata.language.toUpperCase()}</span>
                      </div>
                    )}
                    <p className="text-sm text-muted-foreground capitalize mt-1">{file.status}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
