// Define the shapefile of the area of interest (AOI)
var aoi = ee.FeatureCollection("projects/thesis-473708/assets/cph_great_area");
var aoiGeometry = aoi.geometry(); // Ensures the use of the full area as a single geometry

// Define the range of months for analysis
var months = ['2023-08', '2023-09', '2023-10', '2023-11', '2023-12', '2024-01',
              '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07'];

// Function to apply cloud mask using MSK_CLDPRB
function maskClouds(image) {
  var cloudMask = image.select('MSK_CLDPRB').lte(10); // Keeps pixels with less than 10% cloud probability
  return image.updateMask(cloudMask);
}

// Function to calculate NDVI
function calculateNDVI(image) {
  return image.normalizedDifference(['B8', 'B4']).rename('NDVI');
}

// Function to filter and process images with lower cloud coverage
function filterBestImage(month) {
  var start = ee.Date(month + '-01');
  var end = start.advance(1, 'month');

  // Filter the Sentinel-2 collection
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')  // Using Surface Reflectance collection
 // Using Surface Reflectance collection
    .filterBounds(aoiGeometry)
    .filterDate(start, end)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 50)) // Filters images with up to 50% clouds
    .map(maskClouds); // Applies the cloud mask

  // Create mosaic if needed (use the median of available images)
  var bestImage = collection.median();

  // Calculate NDVI
  var ndvi = calculateNDVI(bestImage).clip(aoiGeometry);

  return ndvi.set('month', month); // Adds the month attribute
}

// Loop to process each month
var ndviResults = months.map(function(month) {
  return filterBestImage(month);
});

// Visualize NDVI on the map
ndviResults.forEach(function(ndvi, index) {
  var ndviImage = ee.Image(ndvi);
  var layerName = 'NDVI ' + months[index];
  Map.addLayer(ndviImage, {min: 0, max: 1, palette: ['blue', 'white', 'green']}, layerName);
});

// Export each NDVI as GeoTIFF
ndviResults.forEach(function(ndvi, index) {
  var fileName = 'NDVI_' + months[index].replace('-', '_');
  Export.image.toDrive({
    image: ee.Image(ndvi),
    description: fileName,
    folder: 'GEE_Exports',
    fileNamePrefix: fileName,
    region: aoiGeometry,
    scale: 10,
    crs: 'EPSG:4326', // WGS84 projection
    maxPixels: 1e13
  });
});

print('Processing completed. Check the map to preview NDVI layers and use the Tasks panel for export.');