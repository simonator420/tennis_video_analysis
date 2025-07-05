
def convert_pixel_distance_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels

def convert_meters_to_pixel_distance(meters, reference_height_in_meters, reference_height_in_pixels):
    return (meters * reference_height_in_pixels) / reference_height_in_meters
def calculate_pixels(frame, width, height):
    frame.x = int(width*frame.x)
    frame.y = int(height*frame.y)
    if frame.xcenter:
        frame.xoffset = int((width-frame.x)/2)
    else:
        frame.xoffset = int(width*frame.xoffset)
    if frame.ycenter:
        frame.yoffset = int((height-frame.y)/2)
    else:
        frame.yoffset = int(height*frame.yoffset)
    return frame