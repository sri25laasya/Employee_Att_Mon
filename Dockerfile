FROM python:3.11
WORKDIR /code
COPY . /code
RUN pip install -r /code/requirements.txt
RUN  apt update
RUN apt install -y libgl1-mesa-glx libglib2.0-0
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]