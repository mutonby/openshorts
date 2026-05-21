FROM node:18-bookworm-slim

# Install Chromium and FFmpeg for Remotion rendering
RUN apt-get update && \
    apt-get install -y --no-install-recommends chromium ffmpeg && \
    rm -rf /var/lib/apt/lists/*

ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
ENV REMOTION_BUNDLE_PATH=/app/remotion
ENV OUTPUT_DIR=/output

WORKDIR /app

# Copy render service source
COPY renderer/service/package.json ./
RUN npm install

COPY renderer/service/tsconfig.json ./
COPY renderer/service/src/ ./src/

# Copy remotion source for bundling
COPY renderer/compositions/package.json /app/remotion/package.json
COPY renderer/compositions/tsconfig.json /app/remotion/tsconfig.json
COPY renderer/compositions/src/ /app/remotion/src/
COPY renderer/compositions/public/ /app/remotion/public/

# Install remotion deps
RUN cd /app/remotion && npm install

# Build TypeScript
RUN npm run build

EXPOSE 3100
CMD ["node", "dist/server.js"]
