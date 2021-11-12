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

### Spin up the database

```
$ cd docker/
$ docker-compose up -d
```

### Export collection for backup

* Exec inside container as root
`docker exec -it --user root mongodb bash`
* Create directory with `$ mkdir /dump/`
* Dump the collection with compressed mode to reduce the size 
`$ mongodump --gzip --db=deep_physical_activity_prediction_db --collection=healthkit_stepscount_singles`
* Exit container `$ exit`

Copy from inside the container the binary files to your computer's desired path.

`$ cd DESIRED/PATH/`

`$ docker cp mongodb:/dump/deep_physical_activity_prediction_db/ .`

### Import collection into a new database from backup

In case you want the data ready to be imported in a database and to be used for data pre-processing and model building,
and overcome the step of download, request them from contributors of this GitHub repository.

* Spin up a **new** database with the following `docker-compose.yml` file. Note that the exposed port is different
* to void conflicts in case a MongoDB already runs in port 27017.

``` 
version: '3.1'

services:

  mongodb:
    image: bitnami/mongodb
    container_name: mongodb-backup
    environment:
        MONGODB_DATABASE: deep_physical_activity_prediction_db
        ALLOW_EMPTY_PASSWORD: "yes"
    volumes: 
        - 'mongodb_data_backup:/bitnami/mongodb'
    ports:
      - 27018:27017

volumes: 
    mongodb_data_backup:
```

* Make a directory `/dump/` inside with `docker exec -it --user root mongodb-backup mkdir /dump/`
* Copy inside the container the above dumped file with `docker cp deep_physical_activity_prediction_db/ mongodb-backup:/dump/`
* Go inside container `docker exec -it --user root mongodb-backup bash`
* Restore it with
`
$ mongorestore --gzip /dump/
`

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
stores them in a clean `healthkit_stepscount_singles` collection.
This is the main collection and source of data of the project, that all data pre-processing and ML/DL model building
is based upon.

