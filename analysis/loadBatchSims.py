import pandas as pd
import json
import glob

workingDir = '../batchData/optuna_batch/'

outFiles = glob.glob(workingDir+'*.out')

data = []
for filename in outFiles:
    with open(filename) as f:
        d = json.loads(f.read())
        data.append(d)     # Collect them in a list

# Convert list of dicts to DataFrame
df = pd.DataFrame(data)
df = df.sort_values('loss')
print(df[df['loss']<=100])
quit()