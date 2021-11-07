# FROM pytorch/pytorch
#FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

WORKDIR "/workspace"

RUN apt-get clean \
        && apt-get update \
        && apt-get install -y ffmpeg libportaudio2 python3-pyqt5 \
        && apt-get -y autoremove


ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENV PYTHONIOENCODING=utf8

COPY . .

CMD ["/bin/bash"]
