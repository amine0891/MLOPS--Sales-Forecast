FROM --platform=linux/amd64 python:3.10.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install  -r requirements.txt


COPY . .

EXPOSE 9000

# CMD ["python", "main.py"]

CMD gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:9000