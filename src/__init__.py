
import sys
import pkg_resources as pkgres
import logging

### Set up version information####
__version__ = pkgres.require("pmfcalculator")[0].version
#################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt='[%(name)s] %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.propagate = False

from readerNd import ReaderNd
#import StatsUtils
