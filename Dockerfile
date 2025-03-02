# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY api_requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r api_requirements.txt

# Download NLTK resources
RUN python -c "import nltk; nltk.download(['punkt', 'stopwords', 'wordnet', 'punkt_tab', 'averaged_perceptron_tagger'])"

# Copy the application code
COPY . .

# Expose the port the application will run on
EXPOSE 8000

# Run the command to start the development server when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
