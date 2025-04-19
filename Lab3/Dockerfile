# Use TensorFlow base image with Python 3
FROM tensorflow/tensorflow:latest-py3

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app port
EXPOSE 4000

# Define the command to run the Flask application
CMD ["python", "flask_iris_api.py"]
