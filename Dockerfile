# Use the official Python image from the Docker Hub
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the entire project to the working directory
COPY . .

# Create and activate the virtual environment
RUN python -m venv .venv \
    && . .venv/bin/activate

# Set environment variable for the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
