Readme for the V4 data in Cowley et al., bioRxiv, 2023.

This directory contains the V4 responses and images from all recording sessions. 
A basic description of the data is below and in Ext. Data Table 1.

Run ../script1_plot_V4_responses_and_images.py to plot images and responses per session.

Images for each session are stored in a zip file in ./images. Note
that you can access the images without unzipping (see ../classes/class_images.py to do this).
Each image is saved as a jpeg and has resolution 112 x 112 x 3.

Raw responses are stored in ./responses_raw. 
responses_raw: (num_neurons, num_images, num_repeats)
These are used for evaluating test performance with noise-corrected R2s.
Note that some images had different numbers of repeats. We use NaNs to fill in the empty repeats.
E.g., responses_raw[0,0,:num_repeats] will have spike counts, whereas responses_raw[0,0,num_repeats:] will be NaNs.
Spike count bins are in 100ms.

Repeat-averaged responses are stored in ./responses_repeataveraged.
responses_avg: (num_neurons, num_images)
These are the exact same as responses_raw except averaged over repeats.

NOTE:
images[iimage], responses_raw[:,iimage,:], and responses_avg[:,iimage] all correspond to each other.



Data table:


   session ID	| animal ID |	train/test |	# neurons	 | # repeats/image | # images | image types
	190923 		WE 		  test 		    33			14			1200		normal (1,200)
    	190924 		WE 		  train 		    25			13			900		normal (600), active learning (300) 
	190925 		WE 		  train 		    36			14			900		normal (600), active learning (300)
	190926 		WE 		  train 		    27 			15			900		normal (600), active learning (300)
	190927 		WE 		  train 		    31			11			900		normal (600), active learning (300)
	190928 		WE 		  train 		    24			13			900		normal (600), active learning (300) 
	190929 		WE 		  train 		    33			15			900		normal (600), active learning (300)  
	201016 		PE 		  train 		    82			6			900		normal (300), gaudy (300)
	201017 		PE 		  train 		    88			7			900		normal (300), active learning (300), gaudy (300)  
	201018 		PE 		  train 		    80			5			600		normal (600)
	201019 		PE 		  train 		    82			4			1600		normal (800), active learning (400), gaudy (400)
	201020 		PE 		  train 		    87			4			1600		normal (800), active learning (400), gaudy (400)
	201021 		PE 		  train 		    88			7			1600		normal (800), active learning (400), gaudy (400)
	201022 		PE		  train 		    79 			5			2000		normal (1,000), active learning (500), gaudy (500)
	201023 		PE 		  train 		    86 			7			2000		normal (1,000), active learning (500), gaudy (500)
	201024 		PE 		  train 		    85 			7			2000		normal (1,000), active learning (500), gaudy (500)
	201025 		PE 		  test 		    89			12			1200		normal (1,200)
	210224 		PE 		  validation 	    67			9			800		normal (800)
	210225 		PE 		  test 		    55			6			1200		normal (1,200)
	210226 		PE 		  train 		    78			6			1600		normal (600), maximizing (1,000)
	210301 		PE 		  train 		    61			4			2000		maximizing (650), validation exps. (1,350)
	210302 		PE 		  train 		    70			5			2000		maximizing (650), validation exps. (1,350)
	210303 		PE 		  train 		    67			3			2049		maximizing (2,049)
	210304 		PE 		  train 		    65			5			1968		maximizing (1,968)
	210305 		PE 		  train 		    70			22			400		validation exps. (400)
	210308 		PE 		  train 		    70			4			2000		normal (500), validation exps. (1,500)
	210309 		PE 		  train 		    56			4			2000		normal (600), validation exps. (1,200), artificial (200)
	210310 		PE 		  train 		    64			3			2000		normal (1,000), gaudy (1,000)
	210312 		PE 		  train 		    81			4			2000		maximizing (2,000)
	210315 		PE 		  train 		    65			2			2000		maximizing (2,000)
	210316 		PE 		  train 		    62			3			2000		normal (1,000), gaudy (1,000)
	210322 		PE 		  train 		    59			4			1200		normal (600), Bashivan et al., 2019 (600)
	210323 		PE 		  train 		    59			4			1200		normal (600), gaudy (600)
	210324 		PE 		  train 		    54			4			1200		normal (600), gaudy (600)
	210325 		PE 		  test 		    50			8			640		Bashivan et al., 2019 (640)
	210326 		PE 		  train 		    60			4			1200		normal (600), gaudy (600)
	210620 		RA 		  train 		    42			9			1600		normal (1,600)
	210621		RA		  train 		    51			16			1200		normal (1,200)
	211008		RA 		  train 		    21	 		24			1200		normal (1,200)
	211012		RA 		  train 		    51	 		15			2000		normal (1,000), gaudy (1,000)
	211013		RA 		  train 		    52	 		15			2000		normal (1,000), gaudy (1,000)
	211014		RA 		  train 		    46	 		11			2000		normal (1,000), gaudy (1,000)
	211015		RA 		  train 		    49	 		18			2000		normal (1,000), gaudy (1,000)
	211018		RA 		  train 		    56	 		10			3000		normal (1,500), gaudy (1,500)
	211022 		RA 		  test 		    42	 		20			1600		normal (1,600) & ~repeats \\
	211025 		RA 		  train 		    54	 		16			2000		normal (233), maximizing (150), gaudy (233), artificial (1,384)
	211026 		RA 		  train 		    56	 		13			2000		normal (660), validation exps. (840), active learning (250), gaudy (250)
	211027 		RA 		  train 		    56	 		18			2000		normal (240), validation exps. (1,260), \\ active learning (250), gaudy (250)
	211028 		RA 		  train 		    55	 		14			2000		normal (660), validation exps. (840), active learning (250), gaudy (250)
	211103 		RA 		  train 		    49	 		13			3000		normal (750), maximizing (750),  active learning (750), gaudy (750)

