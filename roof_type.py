import geopandas as gpd
import numpy as np

def calculate_attributes(gdf):
    """
    Calculate necessary geometrical attributes: area, perimeter, vertices, and number of parts.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame containing the roof geometries.
        
    Returns:
        GeoDataFrame: Updated GeoDataFrame with new attributes.
    """
    gdf['area'] = gdf.geometry.area
    gdf['perimeter'] = gdf.geometry.length
    gdf['num_parts'] = gdf.geometry.apply(lambda geom: len(geom.geoms) if geom.geom_type == 'MultiPolygon' else 1)
    gdf['vertices'] = gdf.geometry.apply(lambda geom: len(geom.exterior.coords) if geom.geom_type == 'Polygon' else sum(len(part.exterior.coords) for part in geom.geoms))
    return gdf

def load_boundary_vertices(boundary_shapefile):
    """
    Load the boundary shapefile and calculate the number of vertices for each boundary.
    
    Args:
        boundary_shapefile (str): Path to the shapefile containing roof boundaries.
        
    Returns:
        dict: Dictionary with boundary vertices, keyed by a unique identifier.
    """
    boundary_gdf = gpd.read_file(boundary_shapefile)
    boundary_gdf['vertices'] = boundary_gdf.geometry.apply(lambda geom: len(geom.exterior.coords) if geom.geom_type == 'Polygon' else sum(len(part.exterior.coords) for part in geom.geoms))
    boundary_vertices = boundary_gdf['vertices'].to_dict()
    return boundary_vertices

def classify_roofs(gdf, boundary_vertices, tolerance=10):
    """
    Classify roofs into flat or sloped based on geometrical properties and comparison with boundary vertices.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame containing the roof polygons with derived attributes.
        boundary_vertices (dict): Dictionary containing the vertices from the boundary shapefile.
        tolerance (int): Tolerance for comparing vertices between roof and boundary.
        
    Returns:
        GeoDataFrame: GeoDataFrame with an additional 'roof_type' column indicating 'flat' or 'sloped'.
    """
    
    def roof_type(row):
        area = row['area']
        perimeter = row['perimeter']
        vertices = row['vertices']
        num_parts = row['num_parts']
        
        # Calculate perimeter-to-area ratio
        ratio = perimeter / area
        
        if area < 11000:
            if perimeter < 200:
                return 0
            elif area < 1000:
                return 0
            elif num_parts > 2:
                # If there are multiple parts, assess the complexity
                if perimeter > 300 and vertices < 150:
                    for boundary_id, boundary_vert in boundary_vertices.items():
                        if abs(vertices - boundary_vert) <= tolerance:
                            return 1
                    return 0
                else:
                    return 0
            elif num_parts == 2:
                # Handle the case with exactly two parts
                parts = list(row.geometry.geoms)
                areas = [part.area for part in parts]
                
                # Compare the areas of the two parts
                if min(areas) / max(areas) < 0.3:
                    return 1  # One part is much smaller than the other
                else:
                    return 0
            elif num_parts == 1:
                return 1
            else:
                return 0
        else:
            return 1
    
    gdf['roof_type'] = gdf.apply(roof_type, axis=1)
    return gdf

# Load your shapefile
shapefile_path = 'roofs.shp'
gdf = gpd.read_file(shapefile_path)

# Calculate the required attributes from geometry
gdf = calculate_attributes(gdf)

# Load the boundary shapefile and calculate boundary vertices
boundary_shapefile_path = 'roof_footprints.shp'
boundary_vertices = load_boundary_vertices(boundary_shapefile_path)

# Classify the roofs
gdf_classified = classify_roofs(gdf, boundary_vertices)

# Save the classified shapefile
gdf_classified.to_file('classified_roofs_1.shp')

# Optionally, display the first few rows
print(gdf_classified.head())
