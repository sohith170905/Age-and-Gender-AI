from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
z=np.load('tmp/age_tuned_predictions.npz')
ages=np.array([int(p.stem.split('_')[0]) for p in sorted(Path('data/UTKFace').glob('*.jpg.chip.jpg'))])
print('ages',ages.shape)
probs=z['age_probs']; reg=z['age_reg']; val=z['val_idx']; test=z['test_idx']
centers=np.arange(10)*10+4.5
for name,pred in [('reg',reg),('expected',probs@centers),('blend50',.5*reg+.5*(probs@centers)),('blend25',.25*reg+.75*(probs@centers))]:
 print(name,'val mae',mean_absolute_error(ages[val],pred[val]),'test mae',mean_absolute_error(ages[test],pred[test]))
for base_name,base in [('reg',reg),('expected',probs@centers),('blend',.5*reg+.5*(probs@centers))]:
 m=LinearRegression().fit(base[val,None],ages[val]); p=np.clip(m.predict(base[test,None]),0,116)
 print('cal',base_name,'coef',m.coef_[0],m.intercept_,'val',mean_absolute_error(ages[val],np.clip(m.predict(base[val,None]),0,116)),'test',mean_absolute_error(ages[test],p))
pred=np.clip(reg,0,116); cal=pred.copy()
for c in range(10):
 mask=(np.floor(pred/10)==c)
 if mask[val].sum():
  d=np.median(ages[val][mask[val]]-pred[val][mask[val]])
  cal[mask]+=d
  print('class',c,'n',mask[val].sum(),'median offset',d)
print('class median test',mean_absolute_error(ages[test],np.clip(cal[test],0,116)), 'val',mean_absolute_error(ages[val],np.clip(cal[val],0,116)))
print('age mean',ages[val].mean(),ages[test].mean(),'reg means',reg[val].mean(),reg[test].mean())
