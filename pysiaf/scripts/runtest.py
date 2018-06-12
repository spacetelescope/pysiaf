import os
import sys
try:
    import pysiaf
except (ImportError, FileNotFoundError):
    if __name__ == '__main__' and __package__ is None:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pysiaf

from pysiaf import utils
from pysiaf.utils import tools2

apName1 = 'NRCA1_FULL'
apName2 = 'NRCA5_FULL'
tools2.matchV2V3(apName1, apName2)
