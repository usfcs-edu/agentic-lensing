import numpy as np, pandas as pd, os, re
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
D9="/raid/benson/git/agentic-lensing/reproductions/huang-2020/data"
D8="/raid/benson/git/agentic-lensing/reproductions/huang-2021/data"
gz=pd.read_parquet("/tmp/gz5.parquet", columns=["ra","dec"])
gzc=SkyCoord(gz.ra.values*u.deg, gz.dec.values*u.deg)
print("GZ-DECaLS:", len(gz))

f7=os.listdir(D9+"/cutouts_fits_dr7")
rd=[]
pat=re.compile(r"DESI-(\d+\.\d+)([+-]\d+\.\d+)\.fits")
for n in f7:
    m=pat.match(n)
    if m: rd.append((float(m.group(1)), float(m.group(2))))
rd=np.array(rd); print("dr7 cutouts parsed:", len(rd))
if len(rd):
    c7=SkyCoord(rd[:,0]*u.deg, rd[:,1]*u.deg)
    _,sep,_=match_coordinates_sky(gzc,c7)
    print("GZ-DECaLS matched to dr7 (<2arcsec):", int((sep<2*u.arcsec).sum()))

pdr8=pd.read_parquet(D8+"/parent_dr8.parquet", columns=["RA","DEC","BRICKID","OBJID"])
print("parent_dr8 rows:", len(pdr8))
pc=SkyCoord(pdr8.RA.values*u.deg, pdr8.DEC.values*u.deg)
idx2,sep2,_=match_coordinates_sky(gzc,pc)
inpar=sep2<2*u.arcsec
print("GZ-DECaLS in parent_dr8 (<2arcsec):", int(inpar.sum()))
saved=set(os.path.splitext(n)[0] for n in os.listdir(D8+"/cutouts_fits_dr8"))
print("saved dr8 cutouts:", len(saved))
key=(pdr8.BRICKID.astype(str)+"_"+pdr8.OBJID.astype(str))
has=key.isin(saved).values
print("parent rows w/ saved cutout:", int(has.sum()))
print("GZ-DECaLS w/ saved dr8 cutout (<2arcsec):", int((inpar & has[idx2]).sum()))
