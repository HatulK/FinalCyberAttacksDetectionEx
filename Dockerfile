FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

ENV NAME FlaskApp

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
