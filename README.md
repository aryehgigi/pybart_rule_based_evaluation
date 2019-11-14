# ud2ude_rule_based_evaluation

in order to tun this, you need odin-wrapper up:
cd
sudo docker-compose up odin-wrapper

configure it to as many ports you want if you want to run a few evaluations parallely, by adding ports under services.odin-wrapper.ports:
- "9090:9090"
- "9091:9090"
.
.
- "9103:9090"
etc
