FROM python:3.8.3

WORKDIR '/usr/src/app'

COPY /requirements.txt .

#RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app/main.py"]