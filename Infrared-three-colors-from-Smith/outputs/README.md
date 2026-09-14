# Figure-derived 18 um morphology

Run `python reconstruct_homunculus_map.py --output outputs`, `python plot_reconstructed_contours.py --outputs outputs`, and `python plot_major_axis_profile.py --outputs outputs`, then `pytest -q`.

Fig1e uses its measured labeled scale (21.14 PDF units per 2 arcsec). Fig3 border-tick positions and separate X/Y raster spacings are measured from the image; assigning 1 arcsec to each unlabeled regular interval is an explicit figure-derived inference. The star-anchored, axis-specific transform is used only to transfer Fig1e contours onto the native Fig3 raster. No science array is spatially resampled. The side-by-side comparison uses exact Fig1e bounds, while the standalone plot shows the full native map. Display smoothing never changes science arrays. This product is provisional morphology, not archival photometry or an astrometric solution.
