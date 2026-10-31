from pathlib import Path
import numpy as np
z=np.load('tmp/age_tuned_predictions.npz')
ages=np.array([int(p.stem.split('_')[0]) for p in sorted(Path('data/UTKFace').glob('*.jpg.chip.jpg'))])
genders=np.array([int(p.stem.split('_')[1]) for p in sorted(Path('data/UTKFace').glob('*.jpg.chip.jpg'))])
reg=z['age_reg']; val=z['val_idx']; test=z['test_idx']
for split,idx in [('val',val),('test',test)]:
 print(split)
 for g in [0,1]:
  m=genders[idx]==g
  err=reg[idx][m]-ages[idx][m]
  print('gender',g,'n',m.sum(),'bias',err.mean(),'mae',np.abs(err).mean())
 for lo,hi in [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,79),(80,89),(90,116)]:
  m=(ages[idx]>=lo)&(ages[idx]<=hi)
  if m.sum(): print(f'{lo}-{hi}',m.sum(),'bias',round((reg[idx][m]-ages[idx][m]).mean(),2),'mae',round(np.abs(reg[idx][m]-ages[idx][m]).mean(),2))
