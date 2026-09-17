# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Frontend. Two stages: `dev` runs the Vite dev server with hot reload;
# `production` builds static assets and serves them with nginx.
#
#   docker compose --profile frontend up          -> dev
#   docker build --target production -f docker/frontend.Dockerfile frontend
# ---------------------------------------------------------------------------
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install

FROM deps AS dev
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

FROM deps AS build
COPY . .
RUN npm run build

FROM nginx:alpine AS production
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
