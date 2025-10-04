import petram
import os

base = os.getenv('PetraM')
if base is None: base = ''
serial = os.path.join(base, 'bin', 'petrams')
parallel = os.path.join(base, 'bin', 'petramp')

