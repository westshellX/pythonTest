import os
import sys
from osgeo import ogr
import ospybook as pb
from ospybook.vectorplotter import VectorPlotter


# Set this variable to your osgeopy-data directory so that the following
# examples will work without editing. We'll use the os.path.join() function
# to combine this directory and the filenames to make a complete path. Of
# course, you can type the full path to the file for each example if you'd
# prefer.
# data_dir = r'D:\osgeopy-data'
data_dir =r'C:\Users\westshell_ASUS\Documents\GisData\osgeopy-data'



########################  5.1  Attribute filters  #############################

# Set up an interactive plotter. Because this is the most fun if you do it
# interactively, you'll probably want to do it from the Python interactive
# prompt. If you're going to run it as a script instead, you might want to use
# a non-interactive plotter instead. Just remember to call draw() when you're
# done.
vp = VectorPlotter(True)

# Get the countries shapefile layer
ds = ogr.Open(os.path.join(data_dir, 'global'))
lyr = ds.GetLayer('ne_50m_admin_0_countries')

# Plot the countries with no fill and also print out the first 4 attribute
# records.
vp.plot(lyr, fill=False)
pb.print_attributes(lyr, 4, ['name'], geom=False)

# Apply a filter that finds countries in Asia and see how many records there
# are now.
lyr.SetAttributeFilter('continent = "Asia"')
lyr.GetFeatureCount()

# Draw the Asian countries in yellow and print out a few features.
vp.plot(lyr, 'y')
pb.print_attributes(lyr, 4, ['name'], geom=False)

# You can still get a feature that is not in Asia by using its FID.
lyr.GetFeature(2).GetField('name')

# Set a new filter that selects South American countries and show the results
# in blue. The old filter is no longer in effect.
lyr.SetAttributeFilter('continent = "South America"')
vp.plot(lyr, 'b')

# Clear all attribute filters.
lyr.SetAttributeFilter(None)
lyr.GetFeatureCount()

