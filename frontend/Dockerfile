FROM node:18-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./
RUN npm install

# Copy the frontend source code
COPY . .

EXPOSE 5173

# Run Vite dev server, accessible from outside the container
CMD ["npm", "run", "dev"]
