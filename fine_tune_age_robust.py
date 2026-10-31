"""Train a candidate that gives the continuous age head a more robust objective.
The verified production checkpoint is never overwritten; selection is done only after
fixed validation/test evaluation.
"""
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from data_utils import make_arrays
from gra_net import _GatedBlend

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--base',default='models/gra_net_age_tuned.keras')
    p.add_argument('--out',default='models/gra_net_age_robust.keras')
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=5e-5)
    args=p.parse_args()
    tf.keras.utils.set_random_seed(42)
    X, labels=make_arrays(args.data)
    split=np.load('results/dataset_split.npz')
    tr,va=split['train_idx'],split['val_idx']
    age=np.asarray(labels['age_class'],dtype=np.int32)
    counts=np.bincount(age[tr],minlength=10).astype(np.float32)
    weights=np.sqrt(counts.mean()/np.maximum(counts,1.0)); weights/=weights.mean()
    y={
      'gender':np.asarray(labels['gender'],dtype=np.int32),
      'age_class':age,
      'age_regression':np.asarray(labels['age_regression'],dtype=np.float32),
    }
    model=tf.keras.models.load_model(args.base,compile=False,custom_objects={'_GatedBlend':_GatedBlend})
    model.compile(
      optimizer=tf.keras.optimizers.Nadam(learning_rate=args.lr,clipnorm=1.0),
      loss={'gender':'sparse_categorical_crossentropy','age_class':'sparse_categorical_crossentropy','age_regression':tf.keras.losses.Huber(delta=5.0)},
      loss_weights={'gender':0.7,'age_class':1.4,'age_regression':1.2},
      metrics={'gender':'accuracy','age_class':'accuracy','age_regression':'mae'},
    )
    train_w={'gender':np.ones(len(tr),np.float32),'age_class':weights[age[tr]],'age_regression':np.ones(len(tr),np.float32)}
    val_w={'gender':np.ones(len(va),np.float32),'age_class':np.ones(len(va),np.float32),'age_regression':np.ones(len(va),np.float32)}
    names=['gender','age_class','age_regression']
    hist=model.fit(X[tr],[y[n][tr] for n in names],sample_weight=[train_w[n] for n in names],validation_data=(X[va],[y[n][va] for n in names],[val_w[n] for n in names]),epochs=args.epochs,batch_size=args.batch_size,shuffle=True,callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_age_regression_mae',mode='min',factor=.5,patience=1,min_lr=1e-6,verbose=1),tf.keras.callbacks.EarlyStopping(monitor='val_age_regression_mae',mode='min',patience=2,restore_best_weights=True,verbose=1)],verbose=1)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); model.save(args.out)
    Path('results/age_robust_history.json').write_text(json.dumps({k:[float(x) for x in v] for k,v in hist.history.items()},indent=2),encoding='utf-8')
    print('saved',args.out)
if __name__=='__main__': main()
