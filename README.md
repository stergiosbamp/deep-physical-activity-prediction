# Physical Activity Prediction with Deep Learning
This is the GitHub repository for the Thesis entitled "Physical Activity Prediction with Deep Learning" of MSc Data and Web Science.


## Project setup

Create virtual environment

```
$ python3 -m venv venv
$ source venv/bin/activate
```

Upgrade pip

```
$ python -m pip install --upgrade pip
```

Install dependencies

```
$ pip install -r requirements.txt
```


## Setting up the MongoDB

For the sake of simplicity this repository contains a `docker-compose.yml`
file that provides easy setup of a required mongo database in order to
download and store the data from the "MyHeart Counts" study.

**Spin up the database**
```
$ cd docker/
$ docker-compose up -d
```

**Export collection for backup**

* Exec inside container as root
`docker exec -it --user root mongodb bash`
* Create directory with `$ mkdir /dump/`
* Dump the collection `$ mongodump --db=deep_physical_activity_prediction_db --collection=healthkit_stepscount_singles`
* Exit container `$ exit`

Copy from inside the container the binary files to the computer
`$ docker cp  mongodb:/dump/deep_physical_activity_prediction_db/ .`


## Download the data from Synapse and import files to the database

Setup your Synapse credentials in the .env file: `SYNAPSE_USER, SYNAPSE_PASSWORD`

Then download the Healthkit data table

```
$ cd src/data/
$ python download.py 
```

This downloads and stores in database in two different collections:
1. the original table from the Healthkit data table.
2. the embedded files associated with each record of the above table.

However, in order to use these data easily they need 
to be merged. To do this

```
$ python importer.py
```

This merges the embedded data for each user and 
stores them in the `healthkit_stepscound_singles` collection.
