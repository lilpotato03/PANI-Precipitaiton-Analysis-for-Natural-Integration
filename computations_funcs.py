def fill_depressions(dem):
    filled = dem.focal_min(2).unmask(dem)
    return filled

