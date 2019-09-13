FROM pytorch/pytorch

WORKDIR "/workspace"

RUN apt-get clean \
        && apt-get update \
        && apt-get install -y ffmpeg libportaudio2 python3-pyqt5 \
        && apt-get -y autoremove
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["/bin/bash"]
