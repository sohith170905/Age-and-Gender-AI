from pathlib import Path
from PIL import Image
import numpy as np
q=np.asarray(Image.open('tmp_user_face_crop.png').convert('RGB').resize((32,32)),dtype=np.float32)/255
paths=sorted(Path('data/UTKFace').glob('*.jpg.chip.jpg'))
# Use grayscale normalized low-res and color for nearest-neighbor diagnostic.
qg=np.asarray(Image.fromarray((q*255).astype('uint8')).convert('L').resize((16,16)),dtype=np.float32)/255
qg=(qg-qg.mean())/(qg.std()+1e-6)
vals=[]
for p in paths:
 try:
  a=np.asarray(Image.open(p).convert('RGB').resize((32,32)),dtype=np.float32)/255
  ag=np.asarray(Image.fromarray((a*255).astype('uint8')).convert('L').resize((16,16)),dtype=np.float32)/255
  ag=(ag-ag.mean())/(ag.std()+1e-6)
  vals.append((float(np.mean((qg-ag)**2)),p.name))
 except: pass
for d,n in sorted(vals)[:20]: print(round(d,4),n)
