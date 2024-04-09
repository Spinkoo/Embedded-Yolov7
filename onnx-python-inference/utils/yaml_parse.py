import yaml
def load_file(path):
    return yaml.safe_load(open(path, 'r'))

# returns number of classes, anchors
def fetch_params(path):
    params = load_file(path)

    nc = params['nc']
    ftmapsz = params['featuremapsz']

    anchors = params['anchors']
    anchors = [[anch[i:i + 2] for i in range(0, len(anch), 2)] for anch in anchors]
    return nc, anchors, ftmapsz
