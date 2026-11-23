PY ?= python3
.PHONY: data data-real train analyze report notebook test run stop api sim dash all clean

data:            ## build dataset (uses real Open-Meteo files if data/raw has them, else SYNTHETIC)
	$(PY) -m src.build_dataset

data-real:       ## download REAL data (needs internet), then rebuild
	$(PY) -m src.fetch_data
	$(PY) -m src.build_dataset --require-real

train:           ## tune, compare, calibrate, save models/ and docs/figures
	$(PY) -m src.train --trials 10

analyze:         ## error analysis -> docs/error_analysis.md
	$(PY) -m src.analyze

report:          ## write measured results into README.md and docs/REPORT.md
	$(PY) -m src.report

notebook:        ## execute the EDA / model comparison notebook
	$(PY) scripts/make_notebook.py

test:
	$(PY) -m pytest -q tests

run:             ## everything: broker + API + dashboard + simulator
	docker compose up --build

stop:
	docker compose down

api:             ## local (no docker): needs a running mosquitto on localhost
	MQTT_HOST=localhost ALLOW_SIM_TIME=1 EVAL_INTERVAL_S=1 API_KEY=dev-key $(PY) -m uvicorn backend.app.main:app --port 8000

sim:
	$(PY) -m iot.simulator --mqtt-host localhost --speed 1

dash:
	API_URL=http://localhost:8000 API_KEY=dev-key streamlit run dashboard/app.py

all: data train analyze report test
clean:
	rm -rf data/processed/* data/synthetic/* models/* .pytest_cache
