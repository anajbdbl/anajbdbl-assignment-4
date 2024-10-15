install:
	. venv/bin/activate && pip install -r requirements.txt

run:
	. venv/bin/activate && flask run --host=0.0.0.0 --port=3000

# python3.8 -m venv venv