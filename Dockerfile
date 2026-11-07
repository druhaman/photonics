FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY doe doe
COPY weights weights
COPY *.py ./

ENTRYPOINT ["python", "synth.py"]
CMD ["/data/input", "/data/output"]
