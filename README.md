# Capstone Project of IBM AI Enterprise Workflow 

Forecasting the time series revenue for the next 30 days from aavail dataset

### Please note

All commands are run in this directory.

### 1. Data Ingesting and EDA 

Run part1.ipynb

### 2. Baseline Model and Performance 

Run part2.ipynb
    
### 3. To train and test the model 

```bash
~$ ! python model/model.py
```

### 4. To run API test, opening the API first 

```bash
~$ python app.py
```

```bash
~$ ! python unittests/ApiTests.py
```

### 5. To run all of the tests

```bash
~$ python run-tests.py
```

### 6. To build the docker container

```bash
~$ docker build -t YOUR_IMAGE_NAME .
```

### 7. Run the container to test that it is working  

```bash
~$ docker run -p 4000:8080 YOUR_IMAGE_NAME
```

Go to http://0.0.0.0:4000/ 




