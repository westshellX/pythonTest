import os
from osgeo import ogr
from ospybook.vectorplotter import VectorPlotter


# Set this variable to your osgeopy-data directory so that the following
# examples will work without editing. We'll use the os.path.join() function
# to combine this directory and the filenames to make a complete path. Of
# course, you can type the full path to the file for each example if you'd
# prefer.
data_dir = r'C:\Users\westshell_ASUS\Documents\GisData\osgeopy-data'
# data_dir =



#########################  6.2  Working with points  ##########################

###########################  6.2.1  Single points  ############################

# Create the firepit point.
firepit = ogr.Geometry(ogr.wkbPoint)
firepit.AddPoint(59.5, 11.5)

# Try out GetX and GetY.
x, y = firepit.GetX(), firepit.GetY()
print('{}, {}'.format(x, y))

# Take a look at the point.
print(firepit)
vp = VectorPlotter(True)
vp.plot(firepit, 'bo')

# Edit the point coordinates.
firepit.AddPoint(59.5, 13)
vp.plot(firepit, 'rs')
print(firepit)

# Or edit the point using SetPoint instead of AddPoint.
firepit.SetPoint(0, 59.5, 13)
print(firepit)

# Make a 2.5D point.
firepit = ogr.Geometry(ogr.wkbPoint25D)
firepit.AddPoint(59.5, 11.5, 2)
print(firepit)

###########################  6.2.2  Multiple points  ##########################

# Create the multipoint to hold the water spigots. Create multipoint and point
# geometries. For each spigot, edit the point coordinates and add the point to
# the multipoint.
faucets = ogr.Geometry(ogr.wkbMultiPoint)
faucet = ogr.Geometry(ogr.wkbPoint)
faucet.AddPoint(67.5, 16)
faucets.AddGeometry(faucet)
faucet.AddPoint(73, 31)
faucets.AddGeometry(faucet)
faucet.AddPoint(91, 24.5)
faucets.AddGeometry(faucet)

# Take a look at the multipoint.
vp.clear()
vp.plot(faucets, 'bo')
vp.zoom(-5)
print(faucets)

# Edit the coordinates for the second faucet.
faucets.GetGeometryRef(1).AddPoint(75, 32)
vp.plot(faucets, 'k^', 'tmp')
print(faucets)

# Change the coordinates back for the next example.
faucets.GetGeometryRef(1).AddPoint(73, 31)
vp.remove('tmp')

# Move all spigots two units to the east. After plotting, you will probably
# have to zoom out a bit in order to really see what happened.
for i in range(faucets.GetGeometryCount()):
    pt = faucets.GetGeometryRef(i)
    pt.AddPoint(pt.GetX() + 2, pt.GetY())
vp.plot(faucets, 'rs')
vp.zoom(-5)
#########################  6.3  Working with lines  ###########################

###########################  6.3.1  Single lines  #############################

# Create the sidewalk line. Make sure to add the vertices in order.
sidewalk = ogr.Geometry(ogr.wkbLineString)
sidewalk.AddPoint(54, 37)
sidewalk.AddPoint(62, 35.5)
sidewalk.AddPoint(70.5, 38)
sidewalk.AddPoint(74.5, 41.5)

# Take a look at the line.
vp = VectorPlotter(True)
vp.plot(sidewalk, 'b-')
print(sidewalk)

# Change the last vertex.
sidewalk.SetPoint(3, 76, 41.5)
vp.plot(sidewalk, 'k--', 'tmp')
print(sidewalk)

# Change the coordinates back for the next example.
sidewalk.SetPoint(3, 74.5, 41.5)
vp.remove('tmp')

# Move the line one unit to the north.
for i in range(sidewalk.GetPointCount()):
    sidewalk.SetPoint(i, sidewalk.GetX(i), sidewalk.GetY(i) + 1)
vp.plot(sidewalk, 'r--')
print(sidewalk)

# Try out GetGeometryCount to prove it that it returns zero for a single
# geometry.
print(sidewalk.GetPointCount()) # vertices
print(sidewalk.GetGeometryCount()) # sub-geometries

# Move the sidewalk back to its original location for the next example.
for i in range(sidewalk.GetPointCount()):
    sidewalk.SetPoint(i, sidewalk.GetX(i), sidewalk.GetY(i) - 1)

# Look at the list of tuples containing vertex coordinates.
print(sidewalk.GetPoints())

# Insert a new vertex between the 2nd and 3rd vertices.
vertices = sidewalk.GetPoints()
vertices[2:2] = [(66.5, 35)]
print(vertices)

# Create a new line geometry from the list of vertices.
new_sidewalk = ogr.Geometry(ogr.wkbLineString)
for vertex in vertices:
    new_sidewalk.AddPoint(*vertex)
vp.plot(new_sidewalk, 'g:')

# Get the original line for the multiple vertices example.
ds = ogr.Open(os.path.join(data_dir, 'misc', 'line-example.geojson'))
if ds is not None:
	lyr = ds.GetLayer()
	feature = lyr.GetFeature(0)
	line = feature.geometry().Clone()
	vp.clear()
	vp.plot(line, 'b-')
	
	# Add a bunch of vertices at different locations. Start from the end so that
	# # earlier indices don't get messed up.
	vertices = line.GetPoints()
	vertices[26:26] = [(87, 57)]
	vertices[19:19] = [(95, 38), (97, 43), (101, 42)]
	vertices[11:11] = [(121, 18)]
	vertices[5:5] = [(67, 32), (74, 30)]
	new_line = ogr.Geometry(ogr.wkbLineString)
	for vertex in vertices:
		new_line.AddPoint(*vertex)
	vp.plot(new_line, 'b--')

# Insert a vertex without creating a new line.
vertices = sidewalk.GetPoints()
vertices[2:2] = [(66.5, 35)]
for i in range(len(vertices)):
    sidewalk.SetPoint(i, *vertices[i])
vp.plot(sidewalk, 'k-', lw=3)