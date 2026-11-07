# Base images are pinned by digest (@sha256): the tag is only a label.
# To update one: docker buildx imagetools inspect <tag> and paste the new digest.
# Multi-stage: `dev` for local docker-compose (HMR), `prod` for Coolify/production.
# The prod stage is last, so `docker build` (and Coolify) produce the static build
# by default. Local docker-compose targets the `dev` stage explicitly.

# ---- deps + dev server (local) ----
FROM node:18-alpine@sha256:8d6421d663b4c28fd3ebc498332f249011d118945588d0a35cb9bc4b8ca09d9e AS dev
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]

# ---- production build ----
FROM node:18-alpine@sha256:8d6421d663b4c28fd3ebc498332f249011d118945588d0a35cb9bc4b8ca09d9e AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
# Vite bakes env at BUILD time. Set VITE_API_URL to the backend's public URL
# (Coolify: add it as a build variable on the frontend service). Empty = same-origin.
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
# Optional product analytics (OpenPanel). Left empty, index.html initialises
# nothing and the build phones home to nobody — see the comment there.
ARG VITE_OPENPANEL_API_URL=""
ENV VITE_OPENPANEL_API_URL=$VITE_OPENPANEL_API_URL
ARG VITE_OPENPANEL_CLIENT_ID=""
ENV VITE_OPENPANEL_CLIENT_ID=$VITE_OPENPANEL_CLIENT_ID
RUN npm run build

# ---- static serving (nginx) ----
FROM nginx:alpine@sha256:713fc2ebfd92e8217db451477182ee91944b123d2b1ee9fd5a8c6e91ca6c3a11 AS prod
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
