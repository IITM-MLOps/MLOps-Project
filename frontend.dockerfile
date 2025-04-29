# Use official nginx image
FROM nginx:alpine

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy your static files to nginx's public folder
COPY . /usr/share/nginx/html/

# Expose port 80
EXPOSE 80

# No CMD needed, nginx runs by default
# From the frontend directory
# docker build -f Dockerfile.frontend -t doodle-frontend .

# Run the container, mapping port 80 in the container to 3000 on your host (optional)
# docker run -d -p 3000:80 --name=doodle-frontend doodle-frontend
