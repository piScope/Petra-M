import petram
import os

base = os.getenv('PetraM')
serial = os.path.join(base, 'bin', 'petrams')
parallel = os.path.join(base, 'bin', 'petramp')

