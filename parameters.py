"""Parameter module"""
import numpy as np, yaml, os

def _generate_matrix(params):
    if not params['C_is_eye']:
        c_mat = np.zeros((params['dimo'], params['dimx']))
        col_h = 0
        col_u = params['dim2']
        col_v = 2 * params['dim2']
        for i in range(params['dimo']):
            if i < params['no_h']:
                c_mat[(i, col_h)] = 1
                col_h += params['h_freq']
            elif params['no_h'] <= i < params['no_u']:
                c_mat[(i, col_u)] = 1
                col_u += params['v_freq']
            else:
                c_mat[(i, col_v)] = 1
                col_v += params['v_freq']

    return c_mat


def get_params(input_file):
    """Get simulation parameters"""
    print('Loading parameters from %s...' % input_file)
    with open(input_file, 'r') as (fin):
        params = yaml.safe_load(fin)
    params['N'] = int(params['N'])
    params['M'] = int(params['M'])
    params['dim'] = params['d'] 
    params['dim2'] = params['dim'] ** 2
    params['dimx'] = 3 * params['dim2']
    params['no_h'] = (params['dim'] // params['h_freq']) ** 2
    params['no_u'] = (params['dim'] // params['v_freq']) ** 2
    params['no_v'] = params['no_u']
    params['dimo'] = params['no_h'] + params['no_u'] + params['no_v']
    params['dx'] = 2 / (params['d'] - 1)
    params['ESS_threshold'] = params['N'] * params['ESS_t']
    params['half_g'] = 0.5 * params['g']
    params['sig'] = params['sig'] ** 2 / params['dimx']
    params['C'] = _generate_matrix(params)
    params['Idx'] = np.eye(params['dimx'])
    params['Ido'] = np.eye(params['dimo'])
    if params['X_CoefMatrix_is_eye']:
        params['ldet_R1'] = 2 * params['dimx'] * np.log(params['sig_x'])
    else:
        raise NotImplementedError('Only X_CoefMatrix_is_eye is supported')
    if params['Y_CoefMatrix_is_eye']:
        params['ldet_R2'] = 2 * params['dimo'] * np.log(params['sig_y'])
    else:
        raise NotImplementedError('Only Y_CoefMatrix_is_eye is supported')
    params['log_det_R2'] = 2 * params['dimo'] * np.log(params['sig_y'])
    params['R1_sqrt'] = params['sig_x'] * params['Idx']
    params['R1'] = params['sig_x'] ** 2 * params['Idx']
    params['log_det_R1'] = 2 * params['dimx'] * np.log(params['sig_x'])




    #create a restart dir
    os.makedirs(params["restart_dir"], exist_ok=True)

    #create an MCMC dir
    os.makedirs(params["mcmc_dir"], exist_ok=True)

    #create an lpf dir
    os.makedirs(params["lpf_dir"], exist_ok=True)

    #create an enkf dir
    os.makedirs(params["enkf_dir"], exist_ok=True)

    #create an etkf dir
    os.makedirs(params["etkf_dir"], exist_ok=True)

    #create an etkf_sqrt dir
    os.makedirs(params["etkf_sqrt_dir"], exist_ok=True)

    return params
