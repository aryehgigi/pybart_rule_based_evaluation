# pyart_rule_based_evaluation

This project, evaluates the BART representation using the evaluation procedure, both described in this [paper](https://arxiv.org/abs/2005.01306).<br/>
We use the [pyBART](https://github.com/allenai/pybart) converter in order to convert the basic UD graphs to our BART, and thus the evaluation needs to use spaCy.<br/>
The evaluation needs the [Spike](https://github.com/allenai/spike){might be private yet, so just look [here](https://spike.covid-19.apps.allenai.org/search/covid19)} tool in order to generate patterns and match them on a dataset. Its best to clone Spike and put our python script under spike/server/


In order to run this, you need odin-wrapper up:
```bash
cd
sudo docker-compose up odin-wrapper
```

configure it to as many ports you want if you want to run a few evaluations parallely, by adding ports under services.odin-wrapper.ports:
```
- "9090:9090"
- "9091:9090"
.
.
- "9103:9090"
etc
```
