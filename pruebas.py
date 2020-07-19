import pandas as pd

data = pd.DataFrame({'A':[0,1,2,3,4],'B':[5,6,7,8,9]})
data['C'] = data.apply(lambda x: 1 if x['A'] != 0 else 0,axis=1)
print(data.apply(lambda x: 1 if x['A'] != 0 else 0,axis=1))

