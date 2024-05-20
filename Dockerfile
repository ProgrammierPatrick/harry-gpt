FROM python:3.12
WORKDIR /app

RUN pip3 install numpy torch flask --extra-index-url https://download.pytorch.org/whl/cpu

COPY server .

CMD [ "python3" , "server.py"]

EXPOSE 5000