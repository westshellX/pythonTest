# I use the print function in this code, even though I don't in the book text,
# so that you can run it as a regular script and still get the output. You only
# get output without using print if you're using the interactive window.


# Set this variable to your osgeopy-data directory so that the following
# examples will work without editing. We'll use the os.path.join() function
# to combine this directory and the filenames to make a complete path. Of
# course, you can type the full path to the file for each example if you'd
# prefer.
import os
import sys
data_dir = r'C:\Users\westshell_ASUS\Documents\GisData\osgeopy-data'
# data_dir =



##########################  3.2 Introduction to OGR  ##########################

# Import the module.
from osgeo import ogr

# Get the GeoJSON driver.
driver = ogr.GetDriverByName('GeoJSON')
print(driver)

# It's not case sensitive, so this also works.
driver = ogr.GetDriverByName('geojson')
print(driver)

# This does not work because the real name is 'Esri shapefile'.
driver = ogr.GetDriverByName('shapefile')
print(driver)

# Print out a list of drivers.
import ospybook as pb
pb.print_drivers()

###########################  3.3 Reading vector data  #########################

####################  3.3.1 Accessing specific features  ######################

# Open the data source for the examples.
fn = os.path.join(data_dir, 'global', 'ne_50m_populated_places.shp')
ds = ogr.Open(fn, 0)
if ds is None:
    sys.exit('Could not open {0}.'.format(fn))
lyr = ds.GetLayer(0)

# Get the total number of features and the last one.
num_features = lyr.GetFeatureCount()
last_feature = lyr.GetFeature(num_features - 1)
print(last_feature.NAME)

# Test what happens if you try to loop through a layer twice. The second
# loop should not print anything. (This is actually why in later examples we
# reopen the data source and get the layer for each little code snippet.
# If you ran them all at once without doing that, they wouldn't work.)
fn = os.path.join(data_dir, 'Washington', 'large_cities.geojson')
ds = ogr.Open(fn, 0)
if ds is None:
	sys.exit('Could not open {0}.'.format(fn))
lyr = ds.GetLayer(0)
print('First loop')
for feat in lyr:
		print(feat.GetField('Name'), feat.GetField('Population'))
print('Second loop')
for feat in lyr:
		pt = feat.geometry()
		print(feat.GetField('Name'), pt.GetX(), pt.GetY())

# # But it will if you reset reading first.
ds = ogr.Open(fn, 0)
if ds is None:
	sys.exit('Could not open {0}.'.format(fn))
lyr = ds.GetLayer(0)
print('First loop')
for feat in lyr:
    print(feat.GetField('Name'), feat.GetField('Population'))
print('Second loop')
lyr.ResetReading() # This is the important line.
for feat in lyr:
    pt = feat.geometry()
    print(feat.GetField('Name'), pt.GetX(), pt.GetY())

	#########################  3.3.2 Viewing your data  ###########################

# Print name and population attributes.
import ospybook as pb
fn = os.path.join(data_dir, 'global', 'ne_50m_populated_places.shp')
pb.print_attributes(fn, 3, ['NAME', 'POP_MAX'])

# Turn off geometries but skip field list parameters that come before the
# "geom" one.
#pb.print_attributes(fn, 3, geom=False)

# If you want to see what happens without the "geom" keyword in the last
# example, try this:
#pb.print_attributes(fn, 3, False)

# Import VectorPlotter and change directories
from ospybook.vectorplotter import VectorPlotter
os.chdir(os.path.join(data_dir, 'global'))

# Plot populated places on top of countries from an interactive session.
vp = VectorPlotter(True)
vp.plot('ne_50m_admin_0_countries.shp', fill=False)
vp.plot('ne_50m_populated_places.shp', 'bo')

# Plot populated places on top of countries non-interactively. Delete the vp
# variable if you tried the interactive one first.
del vp
vp = VectorPlotter(False)
vp.plot('ne_50m_admin_0_countries.shp', fill=False)
vp.plot('ne_50m_populated_places.shp', 'bo')
vp.draw()

#########################  3.4 Getting metadata  ##############################

# Open the large_cities data source.
fn = os.path.join(data_dir, 'Washington', 'large_cities.geojson')
ds = ogr.Open(fn)
if ds is None:
    sys.exit('Could not open {0}.'.format(fn))

# Get the spatial extent.
lyr = ds.GetLayer(0)
extent = lyr.GetExtent()
print(extent)
print('Upper left corner: {}, {}'.format(extent[0], extent[3]))
print('Lower right corner: {}, {}'.format(extent[1], extent[2]))

# Get geometry type
print(lyr.GetGeomType())
print(lyr.GetGeomType() == ogr.wkbPoint)
print(lyr.GetGeomType() == ogr.wkbPolygon)

# Get geometry type as human-readable string.
feat = lyr.GetFeature(0)
print(feat.geometry().GetGeometryName())

# Get spatial reference system. The output is also in listing3_2.py.
print(lyr.GetSpatialRef())

# Get field names and types
for field in lyr.schema:
    print(field.name, field.GetTypeName())
########################  3.5 Writing vector data  ############################

# Check the results from listing 3.2.
os.chdir(os.path.join(data_dir, 'global'))
vp = VectorPlotter(True)
vp.plot('ne_50m_admin_0_countries.shp', fill=False)
vp.plot('capital_cities.shp', 'bo')