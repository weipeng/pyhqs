import csv
import ujson as json
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def read_csv(fname, delimiter=',', quotechar='"'):
    f_txt = open(f, 'rb')
    try:
        yield csv.reader(f_txt, delimiter=delimiter, quotechar=quotechar)
    finally:
        f_txt.close()

@contextmanager
def write_csv(f, delimiter=',', quotechar='"'):
    create_path(f)
    f_txt = open(f, 'wb')
    try:
        yield csv.writer(f_txt, delimiter=delimiter, quotechar=quotechar)
    finally:
        f_txt.close()

def create_path(fname):
    Path(fname).parent.mkdir(parents=True, exist_ok=True)

def save_pickle(obj, fname):
    create_path(fname)
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def save_json(obj, fname):
    create_path(fname)
    with open(fname, 'w') as f:
        json.dump(obj, f)

def load_json(fname):
    with open(fname, 'rb') as f:
        return json.load(f)
