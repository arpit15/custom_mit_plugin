import os
os.environ['DRJIT_LIBLLVM_PATH'] = "/opt/homebrew/opt/llvm@16/lib/libLLVM.dylib"

import mitsuba as mi
# mi.set_variant("scalar_rgb")
mi.set_variant("llvm_ad_rgb")

# from emitters.spot import Spot
from emitters.ies import IES

# mi.register_emitter("myspot", Spot)
mi.register_emitter("ies", IES)

# load scene
scene = mi.load_file("scenes/ies_rendering/ies_modeling.xml")
# calculate image
image = mi.render(scene)
# write image to computer
mi.util.write_bitmap("scenes/ies_rendering/ies.png", image)