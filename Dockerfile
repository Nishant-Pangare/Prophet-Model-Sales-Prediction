# Use the official Python image as a base
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port number on which the Flask app will run
EXPOSE 1000

# Define environment variable
ENV FLASK_APP="Prophet_ModelAPI.py"

# Run the Flask API using the python command
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=1000"]
