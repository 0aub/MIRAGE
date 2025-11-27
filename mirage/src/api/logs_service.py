"""
Logs Streaming Service API
Provides real-time log streaming via Server-Sent Events (SSE)
Streams Docker container logs to the UI for monitoring
Uses Docker Python SDK for container log access
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from loguru import logger
import docker
import asyncio
import re
from datetime import datetime

router = APIRouter()

# Initialize Docker client (connects via Unix socket mounted from host)
try:
    docker_client = docker.from_env()
    logger.info("Docker client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Docker client: {e}")
    docker_client = None


def parse_log_level(line: str) -> str:
    """
    Parse log level from log line

    Returns: "INFO", "WARNING", "ERROR", "DEBUG", or "INFO" (default)
    """
    # Match loguru format: <level>LEVEL</level>
    level_match = re.search(r'<level>(\w+)\s*</level>', line)
    if level_match:
        return level_match.group(1).strip()

    # Match common patterns
    if 'ERROR' in line or 'error' in line or 'Error' in line:
        return 'ERROR'
    elif 'WARNING' in line or 'warning' in line or 'Warning' in line:
        return 'WARNING'
    elif 'DEBUG' in line or 'debug' in line or 'Debug' in line:
        return 'DEBUG'
    else:
        return 'INFO'


def format_log_entry(line: str) -> dict:
    """
    Format a log line into a structured entry

    Returns: Dict with timestamp, level, message
    """
    # Remove ANSI color codes
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    clean_line = ansi_escape.sub('', line)

    # Remove HTML-like tags (from loguru format)
    html_tags = re.compile(r'<[^>]+>')
    clean_line = html_tags.sub('', clean_line)

    # Extract timestamp if present (loguru format: YYYY-MM-DD HH:MM:SS)
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', clean_line)
    timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Parse log level
    level = parse_log_level(line)

    # Clean up the message (remove timestamp and level markers)
    message = clean_line
    if timestamp_match:
        message = message.replace(timestamp_match.group(0), '').strip()

    # Remove level prefix if present
    message = re.sub(r'^\s*\|\s*\w+\s*\|', '', message).strip()

    return {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "raw": line.strip()
    }


async def stream_docker_logs(
    container: str = "mirage-api",
    tail: int = 100,
    follow: bool = True,
    level_filter: Optional[str] = None
):
    """
    Stream logs from Docker container using Docker Python SDK

    Args:
        container: Docker container name
        tail: Number of initial lines to retrieve
        follow: Whether to follow logs (continuous streaming)
        level_filter: Optional log level filter (INFO, WARNING, ERROR, DEBUG)

    Yields:
        Server-Sent Events formatted log entries
    """
    import json

    if not docker_client:
        error_msg = "Docker client not initialized. Check Docker socket mount."
        logger.error(error_msg)
        yield f"data: {json.dumps({'timestamp': datetime.now().isoformat(), 'level': 'ERROR', 'message': error_msg})}\n\n"
        return

    try:
        # Get container
        cont = docker_client.containers.get(container)
        logger.info(f"Starting log stream from container: {container}")

        # Stream logs
        log_stream = cont.logs(
            stream=True,
            follow=follow,
            tail=tail,
            stdout=True,
            stderr=True
        )

        try:
            for log_bytes in log_stream:
                # Decode bytes to string
                line = log_bytes.decode('utf-8', errors='replace').strip()

                if not line:
                    continue

                # Format log entry
                entry = format_log_entry(line)

                # Apply level filter if specified
                if level_filter and entry["level"] != level_filter:
                    continue

                # Yield as SSE event
                yield f"data: {json.dumps(entry)}\n\n"

                # Allow other tasks to run
                await asyncio.sleep(0.01)

        except GeneratorExit:
            logger.info("Client disconnected from log stream")

    except docker.errors.NotFound:
        error_msg = f"Container '{container}' not found"
        logger.error(error_msg)
        yield f"data: {json.dumps({'timestamp': datetime.now().isoformat(), 'level': 'ERROR', 'message': error_msg})}\n\n"
    except Exception as e:
        logger.error(f"Error streaming logs: {e}")
        yield f"data: {json.dumps({'timestamp': datetime.now().isoformat(), 'level': 'ERROR', 'message': f'Error streaming logs: {e}'})}\n\n"


@router.get("/stream")
async def stream_logs(
    container: str = Query(default="mirage-api", description="Container name to stream logs from"),
    tail: int = Query(default=100, ge=10, le=1000, description="Number of initial log lines"),
    level: Optional[str] = Query(default=None, description="Filter by log level (INFO, WARNING, ERROR, DEBUG)")
):
    """
    Stream real-time logs via Server-Sent Events (SSE)

    Usage:
        ```javascript
        const eventSource = new EventSource('/logs/stream?container=mirage-api&tail=100');
        eventSource.onmessage = (event) => {
            const log = JSON.parse(event.data);
            console.log(`[${log.timestamp}] ${log.level}: ${log.message}`);
        };
        ```

    Returns:
        StreamingResponse with Server-Sent Events
    """
    logger.info(f"Log stream requested: container={container}, tail={tail}, level={level}")

    return StreamingResponse(
        stream_docker_logs(
            container=container,
            tail=tail,
            follow=True,
            level_filter=level
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.get("/recent")
async def get_recent_logs(
    container: str = Query(default="mirage-api", description="Container name"),
    tail: int = Query(default=100, ge=10, le=1000, description="Number of log lines"),
    level: Optional[str] = Query(default=None, description="Filter by log level")
):
    """
    Get recent logs (non-streaming)

    Returns:
        List of recent log entries
    """
    if not docker_client:
        raise HTTPException(status_code=503, detail="Docker client not initialized. Check Docker socket mount.")

    try:
        # Get container and logs
        cont = docker_client.containers.get(container)
        log_bytes = cont.logs(
            tail=tail,
            stdout=True,
            stderr=True
        )

        # Decode and split logs
        logs = log_bytes.decode('utf-8', errors='replace').strip().split('\n')

        # Format entries
        entries = []
        for line in logs:
            if not line.strip():
                continue

            entry = format_log_entry(line)

            # Apply level filter
            if level and entry["level"] != level:
                continue

            entries.append(entry)

        logger.info(f"Retrieved {len(entries)} log entries from {container}")

        return {
            "container": container,
            "count": len(entries),
            "logs": entries
        }

    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Container '{container}' not found")
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching logs: {e}")


@router.get("/containers")
async def list_containers():
    """
    List available MIRAGE containers for log streaming

    Returns:
        List of container names with status
    """
    if not docker_client:
        raise HTTPException(status_code=503, detail="Docker client not initialized. Check Docker socket mount.")

    try:
        # Get all MIRAGE-related containers
        all_containers = docker_client.containers.list(all=True, filters={"name": "mirage"})

        containers = []
        for cont in all_containers:
            containers.append({
                "name": cont.name,
                "status": cont.status,
                "running": cont.status == "running"
            })

        logger.info(f"Found {len(containers)} MIRAGE containers")

        return {
            "count": len(containers),
            "containers": containers
        }

    except Exception as e:
        logger.error(f"Error listing containers: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing containers: {e}")
