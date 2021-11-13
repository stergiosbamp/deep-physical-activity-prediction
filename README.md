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
file that provides easy setup of a required MongoDB database in order to
download and store the data from the "MyHeart Counts" study.

In case you want the data ready to be imported in a database and to be used for data pre-processing and model building,
and skip the step of downloading, request them from contributors of this GitHub repository.
Then follow the instructions of how to import the data.

### Spin up the database

The following docker-compose file sets up a default database named `deep_physical_activity_prediction`.  
```
$ cd docker/
$ docker-compose up -d
```

### Import data

* Make a directory `/dump/` inside with `docker exec -it --user root mongodb mkdir /dump/`
* Copy inside the container both dumped binary files (`bson.gz` and `metadata.json.gz`) 
which are stored in a directory for example: `deep_physical_activity_prediction_db/` with `docker cp deep_physical_activity_prediction_db/ mongodb:/dump/`
* Go inside container `docker exec -it --user root mongodb bash`
* Restore it with `$ mongorestore --gzip /dump/`

### Export data

* Exec inside container as root
`docker exec -it --user root mongodb bash`

* Create directory with `$ mkdir /dump/`

* Dump the desired collection (example: `healthkit_stepscount_singles`) with compressed mode to reduce the size 
`$ mongodump --gzip --db=deep_physical_activity_prediction_db --collection=healthkit_stepscount_singles`

* Exit container with `$ exit`

* Copy from inside the container the binary files to your computer's desired path.
    ```
    $ cd DESIRED/PATH/
    $ docker cp mongodb:/dump/deep_physical_activity_prediction_db/ .
    ```



### Setup a backup MongoDB

Spin up a **new** database with the following `docker-compose.yml` file. Note that the exposed port is different
to void conflicts in case a MongoDB already runs in port 27017.

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

Then you can import the data in the same way as described above. Rememeber to use the
new name of container which is `mongodb-backup`.

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

