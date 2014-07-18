import inspect, os
context = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

from mypy.streaming.processor import StructureTensorProcessor, DenseLightFieldEngine

parameter = {"filepath": context+"/rendered/fullRes",
         "resultpath": context+"/results2_FR/cloud",
         "num_of_cams": 11,
         "total_frames": 231,
         "focus": [3, 4],
         "cam_rotation": [180.0, 0.0, 0.0],
         "cam_initial_pos": [0.0, 0.0, 2.0],
         "world_accuracy_m": 0.0,
         "resolution": [960, 540],
         "sensor_size": 32,
         "colorspace": "rgb",
         "inner_scale": 0.6,
         "outer_scale": 1.1,
         "min_coherence": 0.95,
         "focal_length_mm": 16,
         "baseline": 0.01004347826086956522,
         "min_depth": 1.40,
         "max_depth": 2.1}


processor = StructureTensorProcessor(parameter)

engine = DenseLightFieldEngine(parameter, processor)
engine.run()