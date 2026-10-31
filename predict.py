import argparse, numpy as np
from PIL import Image
from gra_net import build_gra_net

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--image',required=True); ap.add_argument('--model',default='models/gra_net.keras'); args=ap.parse_args()
    import tensorflow as tf
    m=tf.keras.models.load_model(args.model,compile=False); x=np.asarray(Image.open(args.image).convert('RGB').resize((32,32)),dtype='float32')[None]/255.; p=m.predict(x,verbose=0)
    print({'gender':'Female' if np.argmax(p[0])==1 else 'Male','age_band':f'{np.argmax(p[1])*10}-{np.argmax(p[1])*10+9}','estimated_age':round(float(p[2][0,0]),1)})
if __name__=='__main__': main()
