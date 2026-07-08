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
COPY package.json package-lock.json ./
RUN npm ci --only=production
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/engine/app/static ./engine/app/static

# Hugging Face runs with its own port, which is exposed to the container via the PORT env variable.
ENV PORT=8000
EXPOSE 8000

CMD ["node", "dist/server.cjs"]
