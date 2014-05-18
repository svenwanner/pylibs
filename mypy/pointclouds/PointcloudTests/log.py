import bpy

with open('/home/swanner/PycharmProjects/PointCloudHandling/PointcloudTests/pos4.log', mode='w', encoding='utf-8') as log:
    cam = bpy.data.objects['Camera']
    camObj = bpy.data.cameras['Camera']
    
    log.write('focal_length_fov_x : ' + str(camObj.angle_x)+'\n')
    log.write('focal_length_fov_y : ' + str(camObj.angle_y)+'\n')
    log.write('focal_length_mm : ' + str(camObj.lens)+'\n')
    
    f_px = (camObj.lens/camObj.sensor_width) * bpy.data.scenes['Scene'].render.resolution_x * (bpy.data.scenes['Scene'].render.resolution_percentage/100.0)
    log.write('focal_length_px : ' + str(f_px)+'\n')
    
    log.write('sensor_height_mm : ' + str(camObj.sensor_height)+'\n')
    log.write('sensor_width_mm : ' + str(camObj.sensor_width)+'\n')
    
    log.write('matrix_world : ' + str(cam.matrix_world)+'\n')
    log.write('location : ' + str(cam.location)+'\n')
    log.write('rotation : ' + str(cam.rotation_euler)+'\n')
    
    log.close()