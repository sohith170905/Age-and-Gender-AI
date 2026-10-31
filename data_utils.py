"""Data helpers for UTKFace-style filenames: [age]_[gender]_[race]_[date].jpg."""
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_SIZE=(32,32)

def parse_utkface_filename(path):
    try:
        parts=Path(path).stem.split('_')
        return int(parts[0]), int(parts[1])
    except Exception:
        return None, None

def load_image(path, size=IMAGE_SIZE):
    img=Image.open(path).convert('RGB').resize(size)
    return np.asarray(img,dtype=np.float32)/255.0

def make_arrays(root, max_items=None, size=IMAGE_SIZE):
    paths=sorted(list(Path(root).rglob('*.jpg.chip.jpg')))
    if max_items: paths=paths[:max_items]
    X=[]; genders=[]; ages=[]
    for p in paths:
        age, gender=parse_utkface_filename(p)
        if age is None or gender not in (0,1): continue
        try: X.append(load_image(p, size=size)); ages.append(age); genders.append(gender)
        except Exception: pass
    if not X: raise ValueError('No compatible UTKFace-style images found.')
    X=np.asarray(X); ages=np.asarray(ages,dtype=np.float32); genders=np.asarray(genders,dtype=np.int32)
    age_class=np.clip((ages//10).astype(np.int32),0,9)
    return X, {'gender':genders,'age_class':age_class,'age_regression':ages}
