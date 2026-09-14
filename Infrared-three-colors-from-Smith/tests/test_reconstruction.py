import numpy as np
from pathlib import Path
from astropy.io import fits
import fitz
from reconstruct_homunculus_map import run, image, detect_fig3_ticks, FIG1_MAJOR_TICKS_PDF, apply_affine, fig1_pixel, fig1_raster_pixel
from plot_major_axis_profile import major_axis_profile

def test_native_tick_calibrated_product(tmp_path, monkeypatch):
 monkeypatch.chdir(Path(__file__).parents[1]);raw=image(fitz.open('Smith-Homonculus-mass-2003.pdf'),69)[9:685,11:691];ticks=detect_fig3_ticks(raw)
 for q in ticks['per_edge'].values():assert q['quality'] and len(q['centers'])>=15 and q['rms_px']<.6
 assert abs(ticks['per_edge']['top']['spacing_px']-ticks['per_edge']['bottom']['spacing_px'])<.05
 assert abs(ticks['per_edge']['left']['spacing_px']-ticks['per_edge']['right']['spacing_px'])<.05
 assert 38.7<ticks['x_px_per_tick']<38.8 and 38.2<ticks['y_px_per_tick']<38.3
 a,b=tmp_path/'a',tmp_path/'b';d=run(a);run(b);z=np.load(a/'i18_map.npz',allow_pickle=True);z2=np.load(b/'i18_map.npz',allow_pickle=True)
 assert np.array_equal(z['i18_map'],z2['i18_map']) and np.isclose(z['weights_unit_sum'].sum(),1)
 assert np.isclose(np.diff(FIG1_MAJOR_TICKS_PDF).mean(),21.14)
 assert .060<d['normalization']['fig1_arcsec_per_pixel']<.062
 r=d['registration'];M=np.asarray(r['matrix']);assert not np.isclose(abs(M[0,0]),abs(M[1,1]))
 assert np.linalg.norm(apply_affine(np.asarray(r['fig1_star_pixel']),M)-np.asarray(r['fig3_star_xy']))<1e-6
 m=z['footprint'];assert m.shape==(676,680) and not(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any())
 h=fits.getheader(a/'i18_map.fits');assert h['CDELT1']>0 and h['CDELT2']<0 and not np.isclose(abs(h['CDELT1']),abs(h['CDELT2']))
 assert np.isclose(h['CDELT1'],ticks['arcsec_per_pixel_x']) and np.isclose(-h['CDELT2'],ticks['arcsec_per_pixel_y'])
 assert np.array_equal(fits.getdata(a/'i18_map.fits'),z['i18_map']) and d['normalization']['spatial_interpolation'].startswith('none')
 p,profile,edges=major_axis_profile(z['i18_map'],m,h)
 pixel_area=abs(h['CDELT1']*h['CDELT2'])
 assert len(p)==len(profile)==len(edges)-1 and np.all(profile>=0)
 assert np.isclose(np.sum(profile*np.diff(edges)),np.sum(z['i18_map'][m])*pixel_area)

def test_fig1_extracted_raster_rows_are_reversed_from_pdf_page_coordinates():
 p=np.array([[173.3704071,503.3273926],[315.2308044,645.1878052]])
 page=fig1_pixel(p);raster=fig1_raster_pixel(p)
 assert np.allclose(page[:,0],raster[:,0])
 assert np.allclose(page[:,1]+raster[:,1],220)
