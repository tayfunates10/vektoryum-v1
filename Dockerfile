# Build stage
FROM node:20-slim AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM node:20-slim
WORKDIR /app

# Ensure /app is owned by the node user
RUN chown -R node:node /app

# Switch to non-root node user (UID/GID 1000)
USER node

COPY --chown=node:node package.json package-lock.json ./
RUN npm ci --only=production

COPY --from=builder --chown=node:node /app/dist ./dist
COPY --from=builder --chown=node:node /app/engine/app/static ./engine/app/static

# Pre-create directory paths to avoid runtime permission issues
RUN mkdir -p /app/vector_jobs /app/vektoryum_data

# Hugging Face runs with its own port, which is exposed to the container via the PORT env variable.
ENV PORT=7860
EXPOSE 7860

CMD ["node", "dist/server.cjs"]
