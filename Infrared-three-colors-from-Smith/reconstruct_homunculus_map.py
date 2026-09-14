#!/usr/bin/env python3
"""Deterministic, explicitly provisional figure reconstruction (Smith 2003)."""
from __future__ import annotations
import argparse, hashlib, json, importlib.metadata
from io import BytesIO
from pathlib import Path
import cv2, fitz, numpy as np
from astropy.io import fits
from scipy import ndimage, optimize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PDF=Path("Smith-Homonculus-mass-2003.pdf"); INS=Path("instructions.md")
PBOX=(173.3704071,503.3273926,315.2308044,645.1878052)
LEVELS=np.array([30,60,120,200,250,300,375,450,520,590,750,870,970,1050,1200,1350,1450,1550,1700,2000,2500,3200.])
STAR=np.array([344.5,350.5])
# Major Fig1e x-axis ticks measured in PDF user units (drawing 252 axes): the
# 2-arcsec spacing is 21.14 PDF units.  These values are persisted rather than
# treating the raster frame as a nominal 13 arcsec field.
FIG1_MAJOR_TICKS_PDF=np.array([166.1955084,187.3355084,208.4755084,229.6155084,250.7555084,271.8955084,293.0355084,314.1755084])
FIG1_MAJOR_TICK_ARCSEC=2.0
FIG1_ARCSEC_PER_PDF=FIG1_MAJOR_TICK_ARCSEC/21.14

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def version(name):
 try:return importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:return "installed-without-package-metadata"
def image(doc,xref): return np.asarray(Image.open(BytesIO(doc.extract_image(xref)["image"])).convert("RGB"))
def segments(d): return np.asarray([[[a.x,a.y],[b.x,b.y]] for it in d["items"] if it[0]=="l" for a,b in [it[1:3]]],float)
def fig1_pixel(p, shape=(221,221)):
 h,w=shape; x0,y0,x1,y1=PBOX; p=np.asarray(p,float)
 return np.stack(((p[...,0]-x0)/(x1-x0)*(w-1),(p[...,1]-y0)/(y1-y0)*(h-1)),axis=-1)
def detect_fig1_cross(drawing):
 s=segments(drawing); hs=[q for q in s if abs(q[0,1]-q[1,1])<.02 and abs(q[1,0]-q[0,0])>10]; vs=[q for q in s if abs(q[0,0]-q[1,0])<.02 and abs(q[1,1]-q[0,1])>10]
 if len(hs)!=1 or len(vs)!=1: raise RuntimeError("Could not uniquely detect Fig1e drawing-252 cross")
 h,v=hs[0],vs[0]; p=np.array([v[0,0],h[0,1]])
 if not(min(h[:,0])<=p[0]<=max(h[:,0]) and min(v[:,1])<=p[1]<=max(v[:,1])): raise RuntimeError("Cross arms do not intersect")
 return p,np.asarray([h,v])
def detect_fig3_star(rgb):
 """Programmatic central black-plus centroid; restrict search to avoid annotations."""
 g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); yy,xx=np.indices(g.shape); roi=(abs(xx-STAR[0])<25)&(abs(yy-STAR[1])<25)
 dark=(g<12)&roi
 # the marker's long orthogonal arms give the largest row/column projections
 x=float(np.argmax(dark.sum(0))); y=float(np.argmax(dark.sum(1)))
 if abs(x-STAR[0])>3 or abs(y-STAR[1])>3: raise RuntimeError("Fig3 stellar plus not found near expected central control")
 return np.array([x+.5,y+.5])
def detect_fig3_ticks(rgb):
 """Derive regular tick combs from raster border strips, never stored centres."""
 g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float);h,w=g.shape
 def edge(axis, side):
  # Perpendicular-stroke response: remove continuous frame via row/column
  # median, then sum local contrast in a narrow inward strip.
  if axis=='x': a=g[:28] if side=='top' else g[-28:]; sig=np.abs(np.diff(a,axis=0)).sum(0); n=w
  else: a=g[:,:28] if side=='left' else g[:,-28:]; sig=np.abs(np.diff(a,axis=1)).sum(1); n=h
  sig[:30]=sig[-30:]=0
  best=None
  for period in np.linspace(36,41,101):
   for phase in np.linspace(0,period,80,endpoint=False):
    q=phase+np.arange(-2,n/period+2)*period;q=q[(q>=30)&(q<n-30)];score=sum(sig[max(0,int(round(v))-2):min(n,int(round(v))+3)].max() for v in q)
    if best is None or score>best[0]:best=(score,period,phase)
  _,period,phase=best; q=phase+np.arange(-2,n/period+2)*period;q=q[(q>=30)&(q<n-30)]; centres=[];lengths=[]
  for v in q:
   lo=max(0,int(round(v))-4);hi=min(n,int(round(v))+5);p=lo+int(np.argmax(sig[lo:hi]));centres.append(float(p));lengths.append(float(sig[p]))
  c=np.array(centres);k=np.arange(len(c));co=np.polyfit(k,c,1);res=c-np.polyval(co,k)
  return {'edge':side,'centers':c.tolist(),'strengths':lengths,'spacing_px':float(co[0]),'rms_px':float(np.sqrt(np.mean(res**2))),'residuals':res.tolist(),'quality':len(c)>=10}
 top=edge('x','top');bot=edge('x','bottom');left=edge('y','left');right=edge('y','right')
 def combine(a,b):
  z=np.array([a['spacing_px'],b['spacing_px']]);e=np.array([max(a['rms_px'],.1),max(b['rms_px'],.1)]);return float(np.average(z,weights=1/e**2)),float(np.sqrt(np.mean(e**2)))
 sx,ux=combine(top,bot);sy,uy=combine(left,right)
 return {'per_edge':{'top':top,'bottom':bot,'left':left,'right':right},'x_px_per_tick':sx,'y_px_per_tick':sy,'x_uncertainty_px':ux,'y_uncertainty_px':uy,'arcsec_per_pixel_x':1/sx,'arcsec_per_pixel_y':1/sy,'assumed_arcsec_per_regular_tick':1.,'angular_assignment':'inferred from publication tick hierarchy and field-size plausibility; Fig3 ticks are unlabeled','alternatives':'0.5 or 2 arcsec/tick imply approximately 8.8 or 35.2 arcsec fields; neither is adopted','method':'image-derived border-strip comb search, local peak refinement, linear regular-sequence fits'}
def direct_registration(srcstar,dststar,ticks):
 fig1scale=FIG1_ARCSEC_PER_PDF*(PBOX[2]-PBOX[0])/220; sx=fig1scale/ticks['arcsec_per_pixel_x'];sy=fig1scale/ticks['arcsec_per_pixel_y'];M=affine_from_params([sx,sy,0],srcstar,dststar)
 return {"authority":"Fig3 border ticks measured from raster; 1 arcsec per regular tick is inferred","parameters":{"scale_x":sx,"scale_y":sy,"rotation_deg":0.},"matrix":M.tolist(),"inverse_matrix":invert_affine(M).tolist(),"fig1_star_pixel":srcstar.tolist(),"fig3_star_xy":dststar.tolist(),"fig1_arcsec_per_pixel":fig1scale,"fig3_arcsec_per_pixel_x":ticks['arcsec_per_pixel_x'],"fig3_arcsec_per_pixel_y":ticks['arcsec_per_pixel_y'],"axis_scale_ratio":{"x":sx,"y":sy},"validation_only":"contour/morphology overlay; no optimizer objective or astrometric claim"}
def marker_mask(shape, star=STAR):
 m=np.zeros(shape,np.uint8); x,y=np.rint(star).astype(int); cv2.rectangle(m,(x-10,y-10),(x+11,y+11),1,-1); return m.astype(bool)
def annotation_mask(shape):
 h,w=shape;m=np.zeros(shape,np.uint8);m[int(.76*h):int(.94*h),int(.76*w):int(.96*w)]=1;m[int(.90*h):int(.97*h),int(.48*w):int(.74*w)]=1;return m.astype(bool)
def rank_feature(a, mask=None):
 """Blurred rank brightness: resistant to colour transfer and Fig1 posterization."""
 a=cv2.GaussianBlur(np.asarray(a,np.float32),(0,0),2.0)
 ok=np.isfinite(a) if mask is None else np.asarray(mask,bool)
 vals=a[ok]; order=np.sort(vals)
 out=np.zeros_like(a); out[ok]=np.searchsorted(order,a[ok],side="right")/max(len(order),1)
 return out
def affine_from_params(params, source_star, target_star):
 sx,sy,theta=params; c,s=np.cos(theta),np.sin(theta); A=np.array([[c*sx,-s*sy],[s*sx,c*sy]])
 return np.column_stack((A,target_star-A@source_star))
def apply_affine(p, M): return np.asarray(p)@M[:,:2].T+M[:,2]
def invert_affine(M):
 A=M[:,:2]; return np.column_stack((np.linalg.inv(A),-np.linalg.inv(A)@M[:,2]))
def transform(s, shape=None, control=None, registration=None):
 """Map Fig1 PDF segments with the star-anchored border-tick transform."""
 if registration is None: raise ValueError("registration is required")
 return apply_affine(fig1_pixel(np.asarray(s)),np.asarray(registration["matrix"],float))
def transform_point(p, shape=None, control=None, registration=None): return transform(np.asarray(p,float),shape,control,registration)
def raster(shape,segs,width=2):
 h,w=shape;m=np.zeros(shape,np.uint8)
 for a,b in np.rint(segs).astype(int): cv2.line(m,tuple(np.clip(a,[0,0],[w-1,h-1])),tuple(np.clip(b,[0,0],[w-1,h-1])),255,width)
 return m

def registration_fit(fig1gray, rgb, srcstar, dststar, ann):
 """Star-anchored isotropic fit on a fixed ROI; no deformation or shear."""
 h,w=rgb.shape[:2]; source=rank_feature(255-fig1gray)
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);seed=(hsv[:,:,1]>40)&(hsv[:,:,2]>25)&~ann;lab,n=ndimage.label(cv2.morphologyEx(seed.astype('u1'),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8)));choices=[]
 for k in range(1,n+1):
  yy,xx=np.where(lab==k)
  if len(xx):choices.append((np.min((xx-dststar[0])**2+(yy-dststar[1])**2),k))
 tmask=ndimage.binary_fill_holes(ndimage.binary_dilation(lab==min(choices)[1],iterations=6)) if choices else seed
 target=rank_feature(rgb[:,:,0],tmask);target[~tmask]=0
 # Guaranteed source coverage for every scale in the prespecified search range.
 roi=np.zeros((h,w),bool);roi[190:511,150:540]=True;roi &= ~ann & ~marker_mask((h,w),dststar)
 sedge=cv2.Canny((source*255).astype('u1'),35,80);tedge=cv2.Canny((target*255).astype('u1'),35,80);dist=cv2.distanceTransform((~tedge).astype('u1'),cv2.DIST_L2,3)
 variants=[(.45,.45),(.55,.55),(.65,.65)]
 def objective(p,detail=False):
  s,theta=p; M=affine_from_params([s,s,theta],srcstar,dststar); q=cv2.warpAffine(source,M,(w,h),flags=cv2.INTER_LINEAR,borderValue=0); e=cv2.warpAffine(sedge,M,(w,h),flags=cv2.INTER_NEAREST)>0
  vals=[]
  for a,b in variants:
   corr=np.corrcoef(q[roi],target[roi])[0,1]; qa=(q>a)&roi; tb=(target>b)&roi; dice=2*(qa&tb).sum()/max(qa.sum()+tb.sum(),1); edge=float(dist[e&roi].mean()/np.hypot(*rgb.shape[:2])) if np.any(e&roi) else 1.; vals.append(-corr-.20*dice+.10*edge)
  val=float(np.median(vals)); return (val,vals,float(np.corrcoef(q[roi],target[roi])[0,1]),float(dist[e&roi].mean()),int(roi.sum())) if detail else val
 bounds=[(1.5,3.2),(-np.deg2rad(2),np.deg2rad(2))]
 de=optimize.differential_evolution(objective,bounds,seed=20403,popsize=14,maxiter=60,tol=2e-5,polish=True,workers=1); candidate=de.x
 # Test rotation, but retain the parallel axes unless its aggregate gain is material.
 zero=optimize.minimize_scalar(lambda s:objective([s,0.]),bounds=bounds[0],method='bounded',options={'xatol':1e-6}); p=np.array([zero.x,0.])
 if objective(candidate)<objective(p)-.01: p=candidate
 old=np.array([(w-1)/220,0.]); base=objective(p);oldscore=objective(old);delta=max(oldscore-base,1e-6)
 grid=np.linspace(*bounds[0],121);vals=np.array([objective([s,p[1]]) for s in grid]);keep=np.r_[grid[vals<=base+.05*delta],p[0]]
 surface_scales=np.linspace(*bounds[0],41);surface_rot=np.linspace(*bounds[1],17);surface=[[float(objective([s,t])) for s in surface_scales] for t in surface_rot]
 M=affine_from_params([p[0],p[0],p[1]],srcstar,dststar)
 return {"parameters":{"scale":float(p[0]),"rotation_deg":float(np.rad2deg(p[1]))},"matrix":M.tolist(),"inverse_matrix":invert_affine(M).tolist(),"fig1_star_pixel":srcstar.tolist(),"fig3_star_xy":dststar.tolist(),"old_parameters":{"scale_x":float(old[0]),"scale_y":float((h-1)/220),"rotation_deg":0.},"objective":{"name":"median over fixed-ROI rank variants of -NCC - 0.20 Dice + 0.10 normalized edge distance","old":float(oldscore),"fitted":float(base),"improvement":float(oldscore-base),"old_detail":objective(old,True),"fitted_detail":objective(p,True),"variants":variants,"fixed_roi_xyxy":[150,190,540,511],"surface":{"scale":surface_scales.tolist(),"rotation_deg":[float(np.rad2deg(t)) for t in surface_rot],"values":surface}},"bounds":{"scale":bounds[0],"rotation_deg":[-2.,2.]},"profile_5pct":{"scale":[float(keep.min()),float(keep.max())],"rotation_deg":[float(np.rad2deg(p[1])),float(np.rad2deg(p[1]))]},"optimum_interior":bool(bounds[0][0]<p[0]<bounds[0][1] and bounds[1][0]<p[1]<bounds[1][1]),"fit_note":"Fixed target ROI and coverage are identical for all trials. Isotropic scale is required by square parallel angular frames; no shear. Edge/Dice are supplementary terms from the same figure data, not independent data."}, source,target

def fig1_stats(gray,segs):
 h,w=gray.shape; out=[]
 for aa,bb in segs:
  a=fig1_pixel(aa,(h,w));b=fig1_pixel(bb,(h,w)); d=b-a;L=np.hypot(*d)
  if L<.2:continue
  n=np.array([-d[1],d[0]])/L
  for t in np.linspace(.05,.95,max(3,int(L))):
   p=a+t*d
   for off in (2,3):
    q=np.rint(p+off*n).astype(int);r=np.rint(p-off*n).astype(int)
    if min(*q,*r)>=0 and q[0]<w and r[0]<w and q[1]<h and r[1]<h:out += [gray[q[1],q[0]],gray[r[1],r[0]]]
 return (float(np.median(out)),float(np.median(abs(np.asarray(out)-np.median(out))),),len(out)) if out else (np.nan,np.nan,0)
def partition(items):
 z=np.array([q["gray"] for q in items]);n=len(z);k=len(LEVELS);span=max(np.ptp(z),1);D={(0,0):(0.,[])}
 for g in range(k):
  N={}
  for (gg,e),(cost,old) in D.items():
   if gg!=g:continue
   for b in range(e+1,min(n,e+4)+1):
    if n-b<k-g-1:continue
    med=np.median(z[e:b]);c=np.median(abs(z[e:b]-med));pen=0 if not old else 6*max(0,med-old[-1][2])**2/span; key=(g+1,b);v=(cost+c+pen,old+[(e,b,float(med))])
    if key not in N or v[0]<N[key][0]:N[key]=v
  D=N
 cost,a=D[(k,n)];return [list(range(x,y)) for x,y,_ in a],{"method":"Fig1e xref35 normal-offset grayscale contiguous DP with monotonic anchors","objective":cost,"status":"provisional: inferred grouping, no independent contour labels","medians":[z[x:y].tolist() for x,y,_ in a]}
def fit(x,y,w):
 o=np.argsort(x);x=x[o];y=y[o];w=w[o];blocks=[[y[i],w[i],[i]] for i in range(len(x))];i=0
 while i<len(blocks)-1:
  if blocks[i][0]>blocks[i+1][0]:a,b=blocks[i],blocks[i+1];blocks[i:i+2]=[[(a[0]*a[1]+b[0]*b[1])/(a[1]+b[1]),a[1]+b[1],a[2]+b[2]]];i=max(0,i-1)
  else:i+=1
 yy=np.zeros(len(y))
 for v,_,ii in blocks:yy[ii]=v
 ux=np.unique(x);return ux,np.array([yy[x==u].max() for u in ux])
def cv(items,g,weight):
 s=np.array([q["rgb"]@weight for q in items]);gid=np.empty(41,int)
 for j,a in enumerate(g):gid[a]=j
 e=[]
 for j in range(22):
  tr=gid!=j;x,y=fit(s[tr],LEVELS[gid[tr]],np.array([q["n"] for q in items])[tr]);e.extend((np.interp(s[~tr],x,y,left=y[0],right=y[-1])-LEVELS[j])**2)
 return float(np.sqrt(np.mean(e)))
def footprint(shape,rgb,ann,star):
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);sig=(hsv[:,:,1]>40)&(hsv[:,:,2]>25)&~ann;sig=cv2.morphologyEx(sig.astype("u1"),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8));lab,n=ndimage.label(sig); choices=[]
 for k in range(1,n+1):
  yy,xx=np.where(lab==k)
  if len(xx):choices.append((float(np.min((xx-star[0])**2+(yy-star[1])**2)),k))
 fp=(lab==min(choices)[1]) if choices else sig;fp=ndimage.binary_fill_holes(ndimage.binary_dilation(fp,iterations=6));fp &= ~ann
 edge_count=int(fp[:2].sum()+fp[-2:].sum()+fp[2:-2,:2].sum()+fp[2:-2,-2:].sum());fp[:2]=fp[-2:]=False;fp[:,:2]=fp[:,-2:]=False
 return fp,{"method":"native Fig3 RGB-composite saturation morphology; no spatial resampling","area_fig3_pixels":int(fp.sum()),"excluded_two_pixel_border_count":edge_count}

def extent(mask, x, y):
 yy,xx=np.where(mask)
 return {"x":[float(x[xx].min()),float(x[xx].max())],"y":[float(y[yy].min()),float(y[yy].max())]} if len(xx) else None
def write_decisions(out,d):
 (out/"decisions.json").write_text(json.dumps(d,indent=2))
 r=d["registration"];n=d["normalization"]
 lines=["# Reconstruction decisions","",f"- Raster-measured Fig3 border ticks give X={r['fig3_arcsec_per_pixel_x']:.8f}, Y={r['fig3_arcsec_per_pixel_y']:.8f} arcsec/pixel under the assumed 1 arcsec regular-tick interval.",f"- Star-anchored Fig1-to-Fig3 contour scales are X={r['parameters']['scale_x']:.6f}, Y={r['parameters']['scale_y']:.6f}; axes parallel, rotation zero.",f"- Native grid {n['shape']} has bounds {n['angular_bounds']} and no spatial resampling.",f"- Extents: {r['extents']}.","- The Fig3 ticks are unlabeled: the 1 arcsec interval is an explicit figure-derived inference, not an independent astrometric calibration."]
 (out/"DECISIONS.md").write_text("\n".join(lines)+"\n")

def run(out="outputs"):
 out=Path(out);out.mkdir(exist_ok=True)
 for p in out.glob("*"):
  if p.is_file():p.unlink()
 doc=fitz.open(PDF); composite=image(doc,69);rgb=composite[9:685,11:691].copy();h,w=rgb.shape[:2];fig1=image(doc,35);gray=cv2.cvtColor(fig1,cv2.COLOR_RGB2GRAY)
 cross_pdf,cross_segments=detect_fig1_cross(doc[1].get_drawings()[252]);srcstar=fig1_pixel(cross_pdf,gray.shape);dststar=detect_fig3_star(rgb);ann=annotation_mask((h,w));inp=marker_mask((h,w),dststar);clean=cv2.inpaint(rgb,inp.astype("u1"),3,cv2.INPAINT_TELEA)
 ticks=detect_fig3_ticks(clean);reg=direct_registration(srcstar,dststar,ticks);source=rank_feature(255-gray);target=rank_feature(clean[:,:,0])
 oldreg={"matrix":affine_from_params([(w-1)/220,(h-1)/220,0],srcstar,dststar).tolist()}
 raw=[]
 for i in range(212,253):
  s=segments(doc[1].get_drawings()[i]);s=s[np.all((s[:,:,0]>=PBOX[0])&(s[:,:,0]<=PBOX[2])&(s[:,:,1]>=PBOX[1])&(s[:,:,1]<=PBOX[3]),axis=1)]
  if i==252:s=np.asarray([q for q in s if not any(np.allclose(q,c,atol=.02) for c in cross_segments)],float)
  raw.append((i,s))
 items=[]
 for i,s0 in raw:
  gr,gm,gn=fig1_stats(gray,s0);s=transform(s0,registration=reg);m=raster((h,w),s);y,x=np.where((m>0)&~ann);vals=clean[y,x].astype(float);items.append({"id":i,"segs":s,"gray":gr,"gray_mad":gm,"gray_n":gn,"rgb":np.median(vals,0),"rgb_mad":np.median(abs(vals-np.median(vals,0)),0),"n":len(vals)})
 groups,meta=partition(items); red=np.array([1.,0,0]);lum=np.array([.2126,.7152,.0722]);grid=np.array([[r,g,1-r-g] for r in (.5,.625,.75,.875,1.) for g in (0,.125,.25,.375,.5) if 1-r-g>=0]);cr=cv(items,groups,red);cl=cv(items,groups,lum);cgrid=[cv(items,groups,a) for a in grid];best=grid[int(np.argmin(cgrid))];cb=min(cgrid)
 scal=np.array([q["rgb"][0] for q in items]);gid=np.empty(41,int)
 for j,g in enumerate(groups):gid[g]=j
 xk,yk=fit(scal,LEVELS[gid],np.array([q["n"] for q in items]));bg=np.median(clean[:30,:30,0]);input_terminal=float(yk[-1])
 if yk[-1]>=3200:keep=yk<3200;xk,yk=xk[keep],yk[keep]
 flux=np.interp(clean[:,:,0],xk,yk,left=0,right=yk[-1]);clip=clean[:,:,0]>xk[-1];fp,fpmeta=footprint((h,w),clean,ann,dststar);flux[~fp]=0;clip &= fp
 inv=np.asarray(reg["inverse_matrix"],float);oh,ow=h,w;fluxo=flux;fpo=fp;clipo=clip;modelo=clean[:,:,0];sxang=ticks['arcsec_per_pixel_x'];syang=ticks['arcsec_per_pixel_y'];x=(np.arange(ow)-dststar[0])*sxang;y=(np.arange(oh)-dststar[1])*(-syang);weights=fluxo/fluxo.sum() if fluxo.sum() else fluxo
 hdr=fits.Header();hdr["BUNIT"]="Jy/arcsec2";hdr["FLXSCALE"]="FIGURE_EST";hdr["CTYPE1"]="XOFFSET";hdr["CTYPE2"]="YOFFSET";hdr["CUNIT1"]="arcsec";hdr["CUNIT2"]="arcsec";hdr["CRPIX1"]=dststar[0]+1;hdr["CRPIX2"]=dststar[1]+1;hdr["CRVAL1"]=0.;hdr["CRVAL2"]=0.;hdr["CDELT1"]=sxang;hdr["CDELT2"]=-syang;hdr["CD1_1"]=sxang;hdr["CD2_2"]=-syang;hdr['F1U0']=0;hdr['F1V0']=0;hdr.add_history("Native Fig3 OFFSET grid calibrated by image-derived regular border ticks; not celestial WCS.")
 fits.writeto(out/"i18_map.fits",fluxo.astype("f4"),hdr,overwrite=True)
 np.savez_compressed(out/"i18_map.npz",i18_map=fluxo.astype("f4"),weights_unit_sum=weights,footprint=fpo,clipped=clipo,model_red=modelo,upper_signal_anchor=yk[-1],segments=np.array([q["segs"] for q in items],dtype=object),marker_mask=marker_mask((h,w),dststar),annotation_mask=ann,registration_matrix=np.asarray(reg["matrix"]),inverse_registration_matrix=inv)
 Image.fromarray(clean).save(out/"fig3_left_rgb_cleaned.png");Image.fromarray((fpo*255).astype("u1")).save(out/"footprint_mask.png");Image.fromarray((clipo*255).astype("u1")).save(out/"clipped_mask.png");Image.fromarray((fluxo/fluxo.max()*255 if fluxo.max() else fluxo).astype("u1")).save(out/"i18_map_peak_norm.png")
 # Registration diagnostics: independent segments remain independent plot calls.
 oldsegs=[transform(s,registration=oldreg) for _,s in raw]
 fig,axs=plt.subplots(1,2,figsize=(12,6));
 for ax,title,ss in zip(axs,["Old full-frame mapping","Direct border-tick transform"],[oldsegs,[q['segs'] for q in items]]):
  ax.imshow(clean)
  for s in ss:
   for a,b in s:ax.plot([a[0],b[0]],[a[1],b[1]],"c-",lw=.35)
  ax.plot(dststar[0],dststar[1],"w+");ax.set_title(title);ax.axis("off")
 fig.savefig(out/"diag_registration_before_after.png",dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));[ax.plot(q['centers'],q['residuals'],'o-',label=name) for name,q in ticks['per_edge'].items()];ax.axhline(0,color='k');ax.set(xlabel='tick centre (pixel)',ylabel='fit residual (pixel)',title='Direct Fig3 border-tick residuals');ax.legend();fig.savefig(out/'diag_tick_residuals.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,7));ax.imshow(fluxo,cmap="inferno",extent=[x[0]-sxang/2,x[-1]+sxang/2,y[-1]-syang/2,y[0]+syang/2],origin="upper",vmin=0,vmax=3200);ax.set(xlabel="R.A. offset (arcsec)",ylabel="Declination offset (arcsec)",aspect="equal");fig.savefig(out/"diag_reconstructed_contours_overlay.png",dpi=160);plt.close(fig)
 fig,ax=plt.subplots();ax.imshow(gray,cmap="gray");[ax.plot(*fig1_pixel(np.array([a,b])).T,color=plt.cm.turbo(j/21),lw=.5) for j,g in enumerate(groups) for i in g for a,b in [raw[i][1][0]] if len(raw[i][1])];ax.axis("off");fig.savefig(out/"diag_fig1e_grouping.png",dpi=180);plt.close(fig)
 fig,ax=plt.subplots();ax.scatter(scal,LEVELS[gid],s=12,alpha=.5);ax.plot(xk,yk,"k-");ax.set(xlabel="Fig3 red",ylabel="Jy arcsec-2");fig.savefig(out/"diag_calibration_robust_spreads.png",dpi=160);plt.close(fig)
 fig,ax=plt.subplots();ax.bar(["red","luminance","RGB constrained"],[cr,cl,cb]);ax.set_ylabel("leave-one-level-out RMSE");fig.savefig(out/"diag_cv_model_comparison.png",dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,7));ax.imshow(clean);ax.contour(fp,[.5],colors="cyan");ax.axis("off");fig.savefig(out/"diag_footprint_overlay.png",dpi=160);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(12,4));[a.axis("off") for a in axs];axs[0].imshow(rgb);axs[0].set_title("before");axs[1].imshow(inp,cmap="gray");axs[1].set_title("inpaint mask");axs[2].imshow(clean);axs[2].set_title("after");fig.savefig(out/"diag_marker_removal.png",dpi=160);plt.close(fig)
 logs=[{"level":float(LEVELS[j]),"path_ids":[items[i]["id"] for i in g],"fig1_gray_median":float(np.median([items[i]["gray"] for i in g])),"fig1_gray_mad":float(np.median([items[i]["gray_mad"] for i in g])),"fig1_samples":int(sum(items[i]["gray_n"] for i in g))} for j,g in enumerate(groups)];grayraw=np.array([q['fig1_gray_median'] for q in logs]);
 for q,v in zip(logs,np.minimum.accumulate(grayraw)):q['fig1_gray_effective_nonincreasing']=float(v)
 # Report prominent morphology, not the intentionally conservative footprint.
 # This is the diagnostic behind the formerly +/-4.5 arcsec apparent extent.
 prominent=fp & (clean[:,:,0]>=np.percentile(clean[:,:,0][fp],55))
 xold=(np.arange(ow)-dststar[0])*13/ow;yold=(np.arange(oh)-dststar[1])*(-13/oh)
 reg['extents']={"definition":"prespecified Fig3 red >=55th percentile within footprint","old_13arcsec":extent(prominent,xold,yold),"direct_ticks":extent(prominent,x,y),"threshold_sensitivity":{"red_percentiles":[45,55,65]}};u1=ow-1;v1=oh-1
 d={"status":"provisional figure-derived normalized morphology product","hashes":{"pdf":digest(PDF),"instructions":digest(INS)},"registration":reg,"grouping":{**meta,"gray_direction":"higher flux is darker in Fig1e","groups":logs},"cv":{"red_rmse":cr,"luminance_rmse":cl,"constrained_rgb_best_rmse":cb,"constrained_rgb_weights":best.tolist(),"selection":"red_only"},"calibration":{"anchors_signal":xk.tolist(),"anchors_flux_effective_pooled":yk.tolist(),"sub30":"unsupported","highest_supported_effective_anchor":float(yk[-1]),"upper_extrapolation_policy":"high pixels held at supported anchor and flagged clipped/lower-bound"},"footprint":fpmeta,"tick_calibration":{"fig1":{"major_tick_pdf_positions":FIG1_MAJOR_TICKS_PDF.tolist(),"arcsec_per_major_interval":2.0,"pdf_units_per_major_interval":21.14,"arcsec_per_pdf":FIG1_ARCSEC_PER_PDF,"provenance":"Measured Fig1e labeled major-vector-tick spacing; zero coincides with detected stellar cross."},"fig3":ticks},"normalization":{"weight_sum":float(weights.sum()),"shape":[oh,ow],"fig1_arcsec_per_pixel":reg['fig1_arcsec_per_pixel'],"fig3_arcsec_per_pixel_x":sxang,"fig3_arcsec_per_pixel_y":syang,"angular_bounds":{"x":[float(x[0]),float(x[-1])],"y":[float(y[-1]),float(y[0])]},"spatial_interpolation":"none; native Fig3 raster"},"visualization":{"decision":"Display-only smoothing; side-by-side clips to exact Fig1e xref35/PBOX bounds while standalone shows the full native grid."},"limitations":["Figure-derived morphology is not archival photometry.","Fig3 angular interval is inferred from unlabeled ticks; no astrometric accuracy claim."]}
 write_decisions(out,d);(out/"contours.json").write_text(json.dumps({"levels":LEVELS.tolist(),"groups":logs},indent=2));(out/"README.md").write_text("# Figure-derived 18 um morphology\n\nRun `python reconstruct_homunculus_map.py --output outputs`, `python plot_reconstructed_contours.py --outputs outputs`, and `python plot_major_axis_profile.py --outputs outputs`, then `pytest -q`.\n\nFig1e uses its measured labeled scale (21.14 PDF units per 2 arcsec). Fig3 border-tick positions and separate X/Y raster spacings are measured from the image; assigning 1 arcsec to each unlabeled regular interval is an explicit figure-derived inference. The star-anchored, axis-specific transform is used only to transfer Fig1e contours onto the native Fig3 raster. No science array is spatially resampled. The side-by-side comparison uses exact Fig1e bounds, while the standalone plot shows the full native map. Display smoothing never changes science arrays. This product is provisional morphology, not archival photometry or an astrometric solution.\n")
 return d
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",default="outputs");run(a.parse_args().output)
