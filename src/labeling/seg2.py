import numpy as np, sys, os, json
from PIL import Image
from skimage import color, filters, morphology, measure
from scipy import ndimage as ndi
sys.path.insert(0,os.path.dirname(__file__))
from seg import load, specimen_mask, features, overlay

P=dict(blur=0.55, Td=13, Tg=7, warm_lo=0.9, warm=1.8, smooth_k=0.62, smooth_Lmin=20, min_sand=60, min_dark=30,
       circ=0.80, solid=0.92, ecc=0.80)

def clean(b, r_open, min_size, close=0):
    if r_open: b=morphology.binary_opening(b,morphology.disk(r_open))
    if close: b=morphology.binary_closing(b,morphology.disk(close))
    b=ndi.binary_fill_holes(b)
    return morphology.remove_small_objects(b,min_size)

def classify(rgb,m,P=P):
    F=features(rgb,m)
    Ls=filters.gaussian(F['Lf'],1.5); bs=filters.gaussian(F['bf'],2)
    tex=F['tex']; w=filters.gaussian(m.astype(float),50)
    tloc=filters.gaussian(np.where(m,tex,0),50)/np.maximum(w,1e-3)
    sharp=tloc/np.median(tloc[m])
    blurry=(sharp<P['blur'])&m          # out-of-focus zones -> unknown
    tex=tex/np.maximum(tloc,1e-3); tmed=1.0
    dark=filters.apply_hysteresis_threshold(-Ls,-(P['Td']+P['Tg']),-P['Td'])&m
    dark=clean(dark,2,P['min_dark'])
    warm=filters.apply_hysteresis_threshold(bs,P['warm_lo'],P['warm'])&m
    smooth=(filters.gaussian(tex,2)<P['smooth_k']*tmed)&(Ls>P['smooth_Lmin'])&m
    sand=clean((warm|smooth)&~dark,2,P['min_sand'],close=2)&~dark&m
    out=np.zeros(m.shape,np.uint8); out[sand]=1
    lbl=measure.label(dark)
    for r in measure.regionprops(lbl):
        circ=4*np.pi*r.area/max(r.perimeter,1)**2
        isv= circ>P['circ'] and r.solidity>P['solid'] and r.eccentricity<P['ecc'] and r.area>=50
        sl=r.slice; sub=out[sl]; sub[r.image]=3 if isv else 2
    out[blurry]=4
    out[~m]=255
    return out

def stats(out):
    tot=(out!=255).sum()
    p={k:100*(out==v).sum()/tot for k,v in [('paste',0),('sand',1),('sand_void',2),('void',3),('unknown',4)]}
    return p

if __name__=='__main__':
    f=sys.argv[1]; od=sys.argv[2]; os.makedirs(od,exist_ok=True)
    rgb=load(f); m=specimen_mask(color.rgb2lab(rgb/255.)[...,0])
    out=classify(rgb,m)
    base=os.path.splitext(os.path.basename(f))[0]
    Image.fromarray(overlay(rgb,out)).save(f'{od}/{base}_overlay.png')
    Image.fromarray(out).save(f'{od}/{base}_mask.png')
    print(base,{k:round(v,1) for k,v in stats(out).items()})
