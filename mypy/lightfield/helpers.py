import numpy as np
import logging,sys
from scipy.ndimage import shift

# ============================================================================================================
#======================              Activate Deugging modus with input argument          ====================
#=============================================================================================================
def checkDebug(argv):
    if len(argv) > 1:
        if argv[1] == 'debug':
            logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    #logging.debug('Example')
    #logging.info('Example')

    return 0




def enum(**enums):
    return type('Enum', (), enums)


# ============================================================================================================
#======================              Shift Light fied to horoptor depth                   ====================
#=============================================================================================================

def refocus_3d(lf, focus, lf_type='h'):
    """
    refocus a 3D light field by an integer pixel shift
    
    :param lf: numpy array of structure [num_of_cams,height,width,channels]
    :param focus: integer pixel value to refocus
    :param lf_type: char 'h' or 'v' to decide between horizontal and vertical light field
    :return lf: numpy array of structure [num_of_cams,height,width,channels]
    """
    assert isinstance(lf, np.ndarray)
    assert isinstance(focus, int) or isinstance(focus, float)
    assert isinstance(lf_type, type(''))

    tmp = np.copy(lf)
    if lf_type == 'h':
        for h in range(lf.shape[0]):
            for c in range(lf.shape[3]):
                if isinstance(focus, float):
                    tmp[h, :, :, c] = shift(lf[h, :, :, c], shift=[0, (h - lf.shape[0] / 2) * focus] )
                elif isinstance(focus, int):
                    tmp[h, :, :, c] = np.roll(lf[h, :, :, c], shift=(h - lf.shape[0] / 2) * focus, axis=1)
    elif lf_type == 'v':
        for v in range(lf.shape[0]):
            for c in range(lf.shape[3]):
                if isinstance(focus, float):
                    tmp[v, :, :, c] = shift(lf[v, :, :, c], shift=[(v - lf.shape[0] / 2) * focus, 0])
                elif isinstance(focus, int):
                    tmp[v, :, :, c] = np.roll(lf[v, :, :, c], shift=(v - lf.shape[0] / 2) * focus, axis=0)
    else:
        print "refocus undefined"

    return tmp