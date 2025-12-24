# Upload Progress Feature

## Overview

Added real-time upload and processing progress tracking to the file upload interface. Users can now see:
- Upload progress (0-100%)
- Processing status polling
- Current processing phase
- Success/failure indicators

## What Changed

### Updated Component: [FileUploadTab.tsx](../ui/src/components/data-sources/FileUploadTab.tsx)

#### Before
- Only showed a toast notification
- No actual file upload
- No progress tracking

#### After
- **Real upload functionality** using XMLHttpRequest with progress events
- **Upload progress bar** showing percentage during file transfer
- **Processing status polling** checks job status every 2 seconds
- **Phase indicators** showing current processing step (e.g., "Extracting entities...")
- **Visual status icons**:
  - 🔄 Spinning loader (blue) - Uploading
  - 🔄 Spinning loader (orange) - Processing
  - ✓ Green check - Completed
  - ✗ Red X - Failed

## Features

### 1. Upload Progress Tracking

```typescript
// XMLHttpRequest with progress event
xhr.upload.addEventListener('progress', (event) => {
  if (event.lengthComputable) {
    const progress = Math.round((event.loaded / event.total) * 100);
    // Update UI with progress
  }
});
```

### 2. Processing Status Polling

```typescript
// Poll job status every 2 seconds
const pollJobStatus = async (jobId: string, fileIndex: number) => {
  const pollInterval = setInterval(async () => {
    const status = await urlApi.getJobStatus(jobId);
    // Update progress, phase, completion status
  }, 2000);
};
```

### 3. Multi-File Support

- Upload multiple files simultaneously
- Track each file independently
- Show progress for all files in a list

### 4. Error Handling

- Connection errors
- Upload failures
- Processing errors
- User-friendly error messages

## User Experience

### Upload Flow

1. **User drops/selects files**
   ```
   Toast: "Upload started - Uploading 1 file(s)..."
   ```

2. **Upload progress** (0-100%)
   ```
   [Filename.pdf] [🔄] 45%
   [━━━━━━━━━-----] 45%
   ```

3. **Processing starts**
   ```
   [Filename.pdf] [🔄] 15%
   [━━━━━----------] 15%
   Phase: "Extracting entities..."
   ```

4. **Processing complete**
   ```
   [Filename.pdf] [✓] Complete
   Toast: "Processing complete - Filename.pdf has been processed successfully"
   ```

5. **If error occurs**
   ```
   [Filename.pdf] [✗] Failed
   Error: "Connection timeout - The server took too long to respond"
   ```

## Technical Details

### API Integration

**Upload Endpoint**: `POST /files/files/upload`
- Accepts `multipart/form-data`
- Returns `file_id` (file uploaded but not yet processed)

**Process Endpoint**: `POST /files/files/{file_id}/process`
- Triggers background processing
- Returns `job_id` for status tracking

**Status Endpoint**: `GET /url/jobs/{job_id}/status`
- Returns:
  - `status`: `queued` | `processing` | `completed` | `failed`
  - `progress`: 0.0 to 1.0
  - `current_phase`: e.g., "Extracting entities"
  - `error`: Error message if failed

### Upload Flow

1. **Upload file** → `POST /files/files/upload` → Returns `file_id`
2. **Start processing** → `POST /files/files/{file_id}/process` → Returns `job_id`
3. **Poll status** → `GET /url/jobs/{job_id}/status` → Returns progress

### State Management

```typescript
interface UploadingFile {
  file: File;                       // Original file object
  uploadProgress: number;           // 0-100 (upload %)
  status: 'uploading' | 'processing' | 'completed' | 'error';
  jobId?: string;                   // Backend job ID
  processingProgress?: number;      // 0-1 (processing %)
  processingPhase?: string;         // Current phase name
  error?: string;                   // Error message
}
```

### Polling Strategy

- **Interval**: 2 seconds
- **Timeout**: 10 minutes (auto-stop)
- **Cleanup**: Clears interval on completion or failure

## Testing

### How to Test

1. **Access the UI**: [http://localhost:3000](http://localhost:3000)

2. **Navigate to Data Sources**

3. **Upload a PDF file**:
   - Drag & drop or click to select
   - Watch the progress bar during upload
   - See processing status update every 2 seconds
   - Get notification when complete

### Expected Behavior

✅ **Small files (<1MB)**:
- Upload: <1 second
- Processing: 10-30 seconds
- Progress updates smoothly

✅ **Medium files (1-10MB)**:
- Upload: 1-5 seconds
- Processing: 30-120 seconds
- Progress visible during upload

✅ **Large files (>10MB)**:
- Upload: 5-30 seconds
- Processing: 2-10 minutes
- Clear progress indication throughout

## Benefits

✅ **Better UX**: Users know what's happening
✅ **Transparency**: See upload and processing progress
✅ **Error visibility**: Clear error messages
✅ **Multi-file**: Track multiple uploads at once
✅ **Professional**: Polished, modern interface

## Integration with Ollama

This works seamlessly with the Ollama integration:
- Files are processed using **gemma3:4b**
- Entity extraction happens during "Processing" phase
- Progress updates show extraction progress
- No API rate limits (local processing)

## Future Enhancements

Potential improvements:
- [ ] Pause/resume upload
- [ ] Cancel processing
- [ ] Detailed phase breakdown (chunking, extraction, storage)
- [ ] Estimated time remaining
- [ ] Upload queue management
- [ ] Retry failed uploads

---

*Generated: 2025-12-24*
*MIRAGE - Upload Progress Feature*
