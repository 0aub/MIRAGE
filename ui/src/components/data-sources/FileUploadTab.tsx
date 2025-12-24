import { useCallback, useState, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Upload, FileText, FileJson, FileCode, CheckCircle, Loader2, XCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { filesApi, urlApi } from "@/lib/api";

interface FileUploadTabProps {
  onClose: () => void;
}

interface UploadingFile {
  file: File;
  uploadProgress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  jobId?: string;
  processingProgress?: number;
  processingPhase?: string;
  error?: string;
}

export default function FileUploadTab({ onClose }: FileUploadTabProps) {
  const { toast } = useToast();
  const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([]);

  const uploadFile = async (file: File) => {
    const fileIndex = uploadingFiles.length;

    // Add file to uploading list
    setUploadingFiles(prev => [
      ...prev,
      {
        file,
        uploadProgress: 0,
        status: 'uploading',
      },
    ]);

    try {
      // Create FormData and upload with progress tracking
      const formData = new FormData();
      formData.append('file', file);

      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          setUploadingFiles(prev => {
            const updated = [...prev];
            if (updated[fileIndex]) {
              updated[fileIndex].uploadProgress = progress;
            }
            return updated;
          });
        }
      });

      const uploadPromise = new Promise<any>((resolve, reject) => {
        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            reject(new Error(`Upload failed: ${xhr.statusText}`));
          }
        });
        xhr.addEventListener('error', () => reject(new Error('Upload failed')));
        xhr.addEventListener('abort', () => reject(new Error('Upload aborted')));
      });

      xhr.open('POST', `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/files/files/upload`);
      xhr.send(formData);

      const uploadResponse = await uploadPromise;

      // Update upload complete - now trigger processing
      setUploadingFiles(prev => {
        const updated = [...prev];
        if (updated[fileIndex]) {
          updated[fileIndex].uploadProgress = 100;
        }
        return updated;
      });

      // Call the process endpoint to start background processing
      const processResponse = await filesApi.process(uploadResponse.file_id);

      // Update status to processing
      setUploadingFiles(prev => {
        const updated = [...prev];
        if (updated[fileIndex]) {
          updated[fileIndex].status = 'processing';
          updated[fileIndex].jobId = processResponse.job_id;
          updated[fileIndex].processingProgress = 0;
        }
        return updated;
      });

      // Poll for processing status
      if (processResponse.job_id) {
        pollJobStatus(processResponse.job_id, fileIndex);
      }

    } catch (error: any) {
      console.error('Upload error:', error);
      setUploadingFiles(prev => {
        const updated = [...prev];
        if (updated[fileIndex]) {
          updated[fileIndex].status = 'error';
          updated[fileIndex].error = error.message;
        }
        return updated;
      });

      toast({
        title: "Upload failed",
        description: error.message || "Failed to upload file",
        variant: "destructive",
      });
    }
  };

  const pollJobStatus = async (jobId: string, fileIndex: number) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await urlApi.getJobStatus(jobId);

        setUploadingFiles(prev => {
          const updated = [...prev];
          if (updated[fileIndex]) {
            updated[fileIndex].processingProgress = status.progress;
            updated[fileIndex].processingPhase = status.current_phase;

            if (status.status === 'completed') {
              updated[fileIndex].status = 'completed';
              clearInterval(pollInterval);

              toast({
                title: "Processing complete",
                description: `${updated[fileIndex].file.name} has been processed successfully`,
              });
            } else if (status.status === 'failed') {
              updated[fileIndex].status = 'error';
              updated[fileIndex].error = status.error || 'Processing failed';
              clearInterval(pollInterval);

              toast({
                title: "Processing failed",
                description: status.error || "Failed to process file",
                variant: "destructive",
              });
            }
          }
          return updated;
        });

      } catch (error) {
        console.error('Failed to poll job status:', error);
        clearInterval(pollInterval);
      }
    }, 2000); // Poll every 2 seconds

    // Stop polling after 10 minutes
    setTimeout(() => clearInterval(pollInterval), 600000);
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    console.log("Files dropped:", acceptedFiles);

    // Upload each file
    acceptedFiles.forEach(file => {
      uploadFile(file);
    });

    toast({
      title: "Upload started",
      description: `Uploading ${acceptedFiles.length} file(s)...`,
    });
  }, []);

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
      {/* Upload Area */}
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

      {/* Upload Progress */}
      {uploadingFiles.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium">Upload Progress</h4>
          {uploadingFiles.map((uploadingFile, index) => (
            <div key={index} className="space-y-2 p-3 border rounded-lg bg-card">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  {uploadingFile.status === 'uploading' && (
                    <Loader2 className="w-4 h-4 animate-spin text-blue-500 flex-shrink-0" />
                  )}
                  {uploadingFile.status === 'processing' && (
                    <Loader2 className="w-4 h-4 animate-spin text-orange-500 flex-shrink-0" />
                  )}
                  {uploadingFile.status === 'completed' && (
                    <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                  )}
                  {uploadingFile.status === 'error' && (
                    <XCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
                  )}
                  <span className="text-sm font-medium truncate">
                    {uploadingFile.file.name}
                  </span>
                </div>
                <span className="text-xs text-muted-foreground flex-shrink-0 ml-2">
                  {uploadingFile.status === 'uploading' && `${uploadingFile.uploadProgress}%`}
                  {uploadingFile.status === 'processing' && (
                    uploadingFile.processingProgress !== undefined
                      ? `${Math.round(uploadingFile.processingProgress * 100)}%`
                      : 'Processing...'
                  )}
                  {uploadingFile.status === 'completed' && 'Complete'}
                  {uploadingFile.status === 'error' && 'Failed'}
                </span>
              </div>

              {/* Progress Bar */}
              {(uploadingFile.status === 'uploading' || uploadingFile.status === 'processing') && (
                <div>
                  <Progress
                    value={
                      uploadingFile.status === 'uploading'
                        ? uploadingFile.uploadProgress
                        : (uploadingFile.processingProgress ?? 0) * 100
                    }
                    className="h-2"
                  />
                  {uploadingFile.status === 'processing' && uploadingFile.processingPhase && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {uploadingFile.processingPhase}
                    </p>
                  )}
                </div>
              )}

              {/* Error Message */}
              {uploadingFile.status === 'error' && uploadingFile.error && (
                <p className="text-xs text-red-500">{uploadingFile.error}</p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Supported File Types */}
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
