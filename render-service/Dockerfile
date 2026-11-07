# Base images are pinned by digest (@sha256): the tag is only a label.
# To update one: docker buildx imagetools inspect <tag> and paste the new digest.
FROM node:18-bookworm-slim@sha256:f9ab18e354e6855ae56ef2b290dd225c1e51a564f87584b9bd21dd651838830e

# Install Chromium and FFmpeg for Remotion rendering
RUN apt-get update && \
    apt-get install -y --no-install-recommends chromium ffmpeg && \
    rm -rf /var/lib/apt/lists/*

ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
ENV REMOTION_BUNDLE_PATH=/app/remotion
ENV OUTPUT_DIR=/output

WORKDIR /app

# Copy render service source
COPY render-service/package.json render-service/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY render-service/tsconfig.json ./
COPY render-service/src/ ./src/

# Copy remotion source for bundling
COPY remotion/package.json remotion/package-lock.json /app/remotion/
COPY remotion/tsconfig.json /app/remotion/tsconfig.json
COPY remotion/src/ /app/remotion/src/
COPY remotion/public/ /app/remotion/public/

# Install remotion deps
RUN cd /app/remotion && npm ci --no-audit --no-fund

# Build TypeScript
RUN npm run build

EXPOSE 3100
CMD ["node", "dist/server.js"]
