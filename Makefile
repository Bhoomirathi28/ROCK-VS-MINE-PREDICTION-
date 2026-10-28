.PHONY: install train test lint run docker

install:      ## install runtime + dev dependencies
	pip install -r requirements-dev.txt

train:        ## clean data, train & compare 4 models, regenerate figures
	python -m src.train

test:         ## run the test-suite
	pytest -q

lint:         ## style check
	flake8 src app tests

run:          ## start the web app on http://localhost:5000
	python app/app.py

docker:       ## build and run the container
	docker build -t rock-vs-mine . && docker run -p 5000:5000 rock-vs-mine
