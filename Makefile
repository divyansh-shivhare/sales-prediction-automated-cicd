.PHONY: install retrain build run deploy clean

 install:
 	python -m pip install -r requirements.txt

 retrain:
	python retrain.py

 build:
	docker build -t cicd-sales-prediction:latest .

 run:
	docker run -p 5000:5000 cicd-sales-prediction:latest

 deploy: build
	./deploy.sh

 clean:
	rm -rf models model.pkl
