"""
MIRAGE - Multilingual Information Retrieval with Accelerated Graph Embeddings
Main FastAPI application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import sys

from src.config import settings
from src.core.retrieval import preload_reranker
from src.core.models.embedding_manager import get_embedding_manager
from src.api import (
    document_service,
    chat_service,
    graph_service,
    refrag_service,
    file_service,
    db_service,
    url_service,
    benchmark_service,
    graphrag_service,
    logs_service,
)

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.log_level,
)

# Initialize FastAPI app
app = FastAPI(
    title="MIRAGE API",
    description="Multilingual Information Retrieval with Accelerated Graph Embeddings",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/")
async def root():
    """API health check and information"""
    return {
        "name": "MIRAGE API",
        "version": "0.1.0",
        "status": "operational",
        "environment": settings.environment,
        "services": {
            "documents": "/documents",
            "chat": "/chat",
            "graph": "/graph",
            "graphrag": "/graphrag",
            "refrag": "/refrag",
            "files": "/files",
            "database": "/db",
            "url": "/url",
            "benchmark": "/benchmark",
        },
        "docs": "/docs" if settings.debug else "disabled",
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Service health check for monitoring"""
    return {
        "status": "healthy",
        "services": {
            "api": "up",
            # Add database checks here later
            # "neo4j": await check_neo4j(),
            # "qdrant": await check_qdrant(),
            # "redis": await check_redis(),
        },
    }


# Include API routers
app.include_router(
    document_service.router,
    prefix="/documents",
    tags=["Documents Registry"],
)

app.include_router(
    chat_service.router,
    prefix="/chat",
    tags=["Chat & Conversation"],
)

app.include_router(
    graph_service.router,
    prefix="/graph",
    tags=["Knowledge Graph"],
)

app.include_router(
    refrag_service.router,
    prefix="/refrag",
    tags=["REFRAG Compression"],
)

app.include_router(
    file_service.router,
    prefix="/files",
    tags=["File Management"],
)

app.include_router(
    db_service.router,
    prefix="/db",
    tags=["Database Management"],
)

app.include_router(
    url_service.router,
    prefix="/url",
    tags=["URL Processing"],
)

app.include_router(
    benchmark_service.router,
    prefix="/benchmark",
    tags=["Benchmarking & Evaluation"],
)

app.include_router(
    graphrag_service.router,
    # Prefix already defined in graphrag_service.router
    tags=["GraphRAG Search"],
)

app.include_router(
    logs_service.router,
    prefix="/logs",
    tags=["System Logs"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An error occurred",
        },
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize connections and services on startup"""
    logger.info("🚀 Starting MIRAGE API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Port: {settings.port}")

    # Initialize database connections here
    # await init_neo4j()
    # await init_qdrant()
    # await init_redis()

    # Preload embedding model (eliminates 20s+ cold-start on first query)
    try:
        logger.info("Loading embedding model for vector search...")
        embedder = get_embedding_manager()
        # Trigger actual model load by making a test embedding
        _ = embedder.embed("test")
        logger.info("✅ Embedding model preloaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Embedding model preload error: {e}")

    # Preload cross-encoder model for semantic mode (eliminates 35s cold-start)
    try:
        logger.info("Loading cross-encoder model for semantic mode...")
        if preload_reranker():
            logger.info("✅ Cross-encoder model preloaded successfully")
        else:
            logger.warning("⚠️ Cross-encoder preload failed - semantic mode may have cold-start")
    except Exception as e:
        logger.warning(f"⚠️ Cross-encoder preload error: {e}")

    logger.info("✅ MIRAGE API started successfully")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down MIRAGE API")

    # Close database connections here
    # await close_neo4j()
    # await close_qdrant()
    # await close_redis()

    logger.info("✅ MIRAGE API shutdown complete")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
