# Development Dockerfile for MIRAGE UI
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install dependencies first (for better caching)
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Expose port (will be overridden by docker-compose)
EXPOSE ${PORT:-3000}

# Start Vite dev server
# --host 0.0.0.0 allows external connections (required for Docker)
# --port uses environment variable
CMD ["sh", "-c", "npm run dev -- --host 0.0.0.0 --port ${PORT:-3000}"]
