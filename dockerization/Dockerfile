FROM  tensorflow/tensorflow:2.8.4-gpu
WORKDIR /workspace/project
RUN apt-get update -y
RUN apt-get install -y libx11-dev
RUN apt-get install -y python3-tk
RUN addgroup --gid 4321 vscode
RUN adduser --uid 1234 --gid 4321 vscode
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt