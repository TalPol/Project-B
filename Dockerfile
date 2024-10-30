FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y ninja-build
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt (if any) and install dependencies
# You can skip this if you don't have dependencies yet
COPY . /app
RUN xargs apt-get -y install < packages.txt
RUN pip install -r requirements.txt
RUN pip install flash-attn --no-build-isolation
RUN pip install mamba-ssm --no-cache-dir
RUN pip install causal-conv1d
SHELL ["/bin/bash", "-c"]
# Keep the container running, allowing you to connect and develop
CMD ["bash"]