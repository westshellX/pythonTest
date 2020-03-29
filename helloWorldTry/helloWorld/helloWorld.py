from osgeo import ogr
from osgeo import gdal

data_dir =r'C:\Users\westshell_ASUS\Documents\GisData\S57Data\CN444121.000'
ds=ogr.Open(data_dir)
print(ds.GetDescription())
print(ds.GetMetadataDomainList())
print(ds)

layCount=ds.GetLayerCount()
layerIndex=0
while layerIndex<layCount:
	layer=ds.GetLayerByIndex(layerIndex)
	print('layer index: {0}. layer name: {1}'.format(layerIndex,layer.GetName()))
	layerIndex+=1

layer=ds.GetLayerByName('DSID')
print('layer 0 name:{0}'.format(layer.GetName()))
layer=ds.GetLayerByName('Point')
if layer is not None:
	print('layer 0 name:{0}'.format(layer.GetName()))
layer=ds.GetLayerByIndex(10000)
print(layer)

driver=gdal.GetDriverByName('S57')
print(driver)
dds=gdal.OpenEx(data_dir,gdal.OF_VECTOR);
layCount=dds.GetLayerCount()
layerIndex=0
while layerIndex<layCount:
	layer=dds.GetLayerByIndex(layerIndex)
	print('layer index: {0}. layer name: {1}'.format(layerIndex,layer.GetName()))
	layerIndex+=1

layer=dds.GetLayerByName('DSID')
print('layer 0 name:{0}'.format(layer.GetName()))
layer=dds.GetLayerByName('Point')
if layer is not None:
	print('layer 0 name:{0}'.format(layer.GetName()))
layer=dds.GetLayerByIndex(10000)
print(layer)