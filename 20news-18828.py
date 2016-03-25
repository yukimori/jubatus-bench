import sys
import os

for path in sys.argv[1:]:
    label = os.path.basename(os.path.dirname(path))
    try:
        data = open(path, 'r', encoding='utf-8').read()
    except:
        data = open(path, 'r', encoding='latin-1').read()
    data = data.replace('"', '""')
    sys.stdout.write(label)
    sys.stdout.write(',"')
    sys.stdout.write(data)
    sys.stdout.write('"\n')
