
RC_PARAMS = {}
# Configure rcParams
size  = {'major': 6, 'minor': 3}
width = {'major': 1, 'minor': 1}
for xy in ['xtick', 'ytick']:
    for t in ['major', 'minor']:
        RC_PARAMS[f'{xy}.{t}.size'] = size[t]
        RC_PARAMS[f'{xy}.{t}.width'] = width[t]
    RC_PARAMS[f'{xy}.direction'] = 'in'
    RC_PARAMS[f'{xy}.color'] = 'white'
RC_PARAMS['text.usetex'] = True
RC_PARAMS['axes.edgecolor'] = 'white'