import sys, pathlib, numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import tensorflow as tf
from data_utils import make_arrays
import gra_net
from sklearn.metrics import mean_absolute_error, accuracy_score
root=pathlib.Path(__file__).resolve().parent
X, labels=make_arrays(str(root/'data/UTKFace'))
split=np.load(root/'results/dataset_split.npz')
idx=np.concatenate([split['val_idx'],split['test_idx']])
age=np.asarray(labels['age_regression'])[idx]
cls=np.asarray(labels['age_class'])[idx]
model=tf.keras.models.load_model(root/'models/gra_net_age_tuned.keras',compile=False,custom_objects={'_GatedBlend':gra_net._GatedBlend})
orig=np.asarray(model.predict(X[idx],batch_size=128,verbose=0))
flip=np.asarray(model.predict(X[idx,:,::-1,:],batch_size=128,verbose=0))
for name,outs in [('orig',orig),('flip',flip),('avg',[(orig[i]+flip[i])/2 for i in range(3)])]:
 g,c,r=[np.asarray(v) for v in outs]
 agep=np.clip(r.reshape(-1),0,116)
 cp=np.argmax(c,axis=1)
 print(name,'age_mae',mean_absolute_error(age,agep),'age_cls',accuracy_score(cls,cp),'gender',accuracy_score(labels['gender'][idx],np.argmax(g,axis=1)))
