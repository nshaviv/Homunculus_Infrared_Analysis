
Consider Smith 2003 (attached pdf), figures 1e (bottom left panel) and Fig 3 left panel.

We are interested in obtaining a normalized 2D map at 18microns of the homunculus.

The problem, fig 3 left panel is an RGB unnormalized image, while fig 1e is a contour plot of the 18microns.

The idea, extract the three chanels from fig 3 left. The catch there might be some color mixing and it may be nonlinear. 

To "de-mix" perhaps we can use SVD analysis to relate the RGB to the 3 colors. And then use the contour plots to calibrate the brightness scale to the flux values. 

Use GPT 6 Astra to design. GPT 5.6 Terra to orchestrate and GPT 5.3 Spark for the actual coding.
