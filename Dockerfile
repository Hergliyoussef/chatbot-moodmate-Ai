FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
# Port pour Gradio
EXPOSE 7860 
CMD ["python", "app_moodmate_local.py"]