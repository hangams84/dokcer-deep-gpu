###build
sudo docker build -t docker-keti-deep:conda-py36-gpu .

### run
sudo docker run --gpus all -it --runtime=nvidia --name conda-py36-conda 
-p 5000:8888 -p 5001:6006 -p 5002:22 docker-keti-deep:conda-py36-gpu jupyter notebook --no-browser 
--ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/root'

### ssh service start
sudo service ssh start
