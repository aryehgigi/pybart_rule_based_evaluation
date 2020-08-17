#!/bin/bash

cd spike/server

### lets start with bringing up odin-wrapper
sudo docker-compose up odin-wrapper &

### Generate (for train) and Annotate (for dev/test):
python3.7 rule_based.py -a generate SOME_PARAM_LIST
python3 rule_based.py -a annotate -d dev &
python3 rule_based.py -a annotate -d test &

wait
echo "finished annotating"

### (you might need to do the following instructions, for each odinson folder you want to use e.g. train/test or UD/enhancedUD/BART-config1/etc).
### copy the annotated data to 
cp -r spike/server/resources/datasets/SPECIFIC_NAME/ann ODINSON_LOCATION/docs


cd ODINSON_LOCATION
### before running the indexing you should fix some things in the odinson folder (just replace FILL1/FILL2 with the strings to replace,
###		notice that we want to replcae whatever they have there (something like ${user.home}/data/odinson) to our ODINSON_LOCATION):
sed -i "s/FILL1/FILL2/g" extra/src/main/resources/application.conf
sed -i "s/FILL1/FILL2/g" core/src/main/resources/reference.conf

### to run the indexing process:
sbt "extra/runMain ai.lum.odinson.extra.IndexDocuments"


### Choose a port for the backend e.g. 9000
### to run an odinson backend :
sbt "project backend" "run $PORT_NUMBER"

### but if you dont want it block use:
# screen -d -m sbt "project backend" "run $PORT_NUMBER"

### evaluate:
cd ../spike/server
python3.7 rule_based.py -a eval SOME_PARAM_LIST_2


