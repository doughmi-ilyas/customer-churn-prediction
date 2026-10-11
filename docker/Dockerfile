FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir $(grep -E '^(pandas|numpy|scikit-learn|imbalanced-learn|streamlit|joblib)==' requirements.txt)

COPY app/ app/
COPY artifacts/ artifacts/

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
