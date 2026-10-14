import numpy as np, sys, os, json
from PIL import Image
from skimage import color, filters, morphology, measure, segmentation
from scipy import ndimage as ndi

def load(f, ds=2):
    im=Image.open(f).convert('RGB'); im=im.resize((im.size[0]//ds, im.size[1]//ds), Image.LANCZOS)
    return np.asarray(im)

def specimen_mask(L):
    Lb=filters.gaussian(L,6)
    m=Lb<60
    lbl=measure.label(m); r=max(measure.regionprops(lbl),key=lambda r:r.area)
    m=ndi.binary_fill_holes(lbl==r.label)
    m=morphology.binary_opening(m,morphology.disk(15))
    # trim margin (edges are blurred / shadowed)
    marg=int(0.03*min(L.shape))
    # robust square-ish box from row/col coverage (drops grid-paper strips and chipped corners)
    rows=m.mean(1); cols=m.mean(0)
    r=np.where(rows>0.6*rows.max())[0]; c=np.where(cols>0.6*cols.max())[0]
    box=np.zeros_like(m); box[r[0]:r[-1]+1, c[0]:c[-1]+1]=True
    m=m&box
    marg=int(0.045*min(L.shape))
    m=ndi.distance_transform_edt(np.pad(m,1))[1:-1,1:-1]>marg
    return m

def features(rgb, m):
    lab=color.rgb2lab(rgb/255.)
    L,a,b=lab[...,0],lab[...,1],lab[...,2]
    # lighting flatten: normalized blur within mask
    w=filters.gaussian(m.astype(float),60); 
    Lbg=filters.gaussian(np.where(m,L,0),60)/np.maximum(w,1e-3)
    med=np.median(L[m])
    Lf=33.0*L/np.maximum(Lbg,3)
    bbg=filters.gaussian(np.where(m,b,0),60)/np.maximum(w,1e-3)
    abg=filters.gaussian(np.where(m,a,0),60)/np.maximum(w,1e-3)
    bf=b-bbg; af=a-abg
    hp=np.abs(L-filters.gaussian(L,1.5))
    tex=filters.gaussian(hp,3)
    return dict(L=L,Lf=Lf,af=af,bf=bf,tex=tex,lab=lab)

def classify(rgb, m, P):
    F=features(rgb,m)
    Lf,af,bf,tex=F['Lf'],F['af'],F['bf'],F['tex']
    seg=segmentation.slic(np.dstack([Lf,af*2,bf*2]), n_segments=int(m.size/80), compactness=8, channel_axis=-1, convert2lab=False, max_num_iter=5, start_label=1)
    n=seg.max()+1
    idx=seg.ravel()
    cnt=np.bincount(idx,minlength=n).astype(float); cnt[cnt==0]=1
    mean=lambda x: np.bincount(idx,x.ravel(),n)/cnt
    mL,ma,mb,mt=mean(Lf),mean(af),mean(bf),mean(tex)
    sL=np.sqrt(np.maximum(mean(Lf**2)-mL**2,0))
    texmed=np.median(tex[m])
    cls=np.zeros(n,np.uint8)  # 0 paste
    dark = mL < P['dark_L']
    warm = (mb > P['warm_b']) | ((mb>P['warm_b2']) & (ma>P['warm_a']))
    smooth = (mt < P['smooth_t']*texmed) & (sL < P['smooth_s']) & (mL < P['gray_hi']) & (mL > P['dark_L'])
    cls[warm & ~dark]=1
    cls[smooth]=1
    cls[dark]=2
    lab=cls[seg]; lab[~m]=255
    # cleanup sand
    sand=lab==1
    sand=morphology.binary_opening(sand,morphology.disk(2))
    sand=ndi.binary_fill_holes(sand)
    sand=morphology.remove_small_objects(sand,P['min_sand'])
    dk=(lab==2)
    dk=morphology.binary_opening(dk,morphology.disk(1))
    dk=morphology.remove_small_objects(dk,P['min_dark'])
    out=np.zeros(lab.shape,np.uint8)
    out[sand]=1
    # dark regions -> sand void (2) or round air void (3)
    lbl=measure.label(dk)
    for r in measure.regionprops(lbl):
        circ=4*np.pi*r.area/max(r.perimeter,1)**2
        is_void = circ>P['circ'] and r.solidity>P['solid'] and r.eccentricity<P['ecc']
        out[lbl==r.label]=3 if is_void else 2
    out[~m]=255
    return out, F

P=dict(dark_L=14, warm_b=3.0, warm_b2=1.8, warm_a=1.5, smooth_t=0.6, smooth_s=6.0, gray_hi=45,
       min_sand=40, min_dark=25, circ=0.85, solid=0.93, ecc=0.75)

def overlay(rgb, out, alpha=0.55):
    o=rgb.astype(float).copy()
    cols={1:(255,230,0),2:(255,130,0),3:(0,200,255),4:(150,60,200)}
    for k,c in cols.items():
        s=out==k; o[s]=(1-alpha)*o[s]+alpha*np.array(c)
    o[out==255]*=0.35
    return o.clip(0,255).astype(np.uint8)

if __name__=='__main__':
    f=sys.argv[1]; od=sys.argv[2]; os.makedirs(od,exist_ok=True)
    rgb=load(f); 
    m=specimen_mask(color.rgb2lab(rgb/255.)[...,0])
    out,F=classify(rgb,m,P)
    base=os.path.splitext(os.path.basename(f))[0]
    Image.fromarray(overlay(rgb,out)).save(f'{od}/{base}_overlay.png')
    Image.fromarray(out).save(f'{od}/{base}_mask.png')
    tot=(out!=255).sum()
    print(base, {k:round(100*(out==v).sum()/tot,1) for k,v in [('paste',0),('sand',1),('sandvoid',2),('void',3)]})
