# intel_in_air_task

##
1. create python venv (make sure you have python3.8)

```
make
```

2. to activate the venv run
```
source  .venv/current/bin/activate
```

3. Change config file for dataloader to work on you PC
```
configs/dataloader.yaml
```

5. to train the network
```
python intel_in_air_task/train.py
```

4. ro evaluate and test the model you can use 
```
python intel_in_air_task/eval.py
```