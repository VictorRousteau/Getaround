FROM continuumio/miniconda3

WORKDIR /home/app

RUN apt-get update
# install nano and unzip
RUN apt-get install nano unzip
# install curl
RUN apt install curl -y

# install aws cli
RUN curl -fsSL https://get.deta.dev/cli.sh | sh

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

# copy requirements.txt in the container and install dependencies with pip
COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

# set up environment variables declared using the
# docker run -e var_name="var_value" image_name
# or that we will declare using heroku later
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV BACKEND_STORE_URI=$BACKEND_STORE_URI
ENV ARTIFACT_ROOT=$ARTIFACT_ROOT

# run the mlflow server in the container
CMD mlflow server -p $PORT \
    --host 0.0.0.0 \
    --backend-store-uri $BACKEND_STORE_URI \
    --default-artifact-root $ARTIFACT_ROOT