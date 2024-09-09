import numpy as np
from .osd import Osd
from .tools import DEM_matrices, DEM_to_matrices
import torch.nn as nn

def naive_bposd(circuit, syndrome, max_iterations):
    
    matrices = DEM_to_matrices(circuit.detector_error_model(decompose_errors=False))
    priors = matrices.priors
    check_matrix = matrices.check_matrix.toarray()
    
    llr = np.log( (1-priors) / priors )
    syndrome = np.reshape(syndrome, (len(syndrome), 1))

    # Messages initialisations
    qc_msg = check_matrix*llr

    # Iterative message passing
    for _ in range(max_iterations):

        # Check to qubit messages
        divide = np.tanh(qc_msg/2)
        divide[divide==0] = 1
        cq_msg = 2*np.arctanh( np.prod(divide, axis=1, keepdims=True) / divide )
        cq_msg = cq_msg*check_matrix
        cq_msg = cq_msg*( ((-1)**syndrome) )


        # Qubit to check messages
        qc_msg = check_matrix*llr
        qc_msg += np.sum(cq_msg, axis=0, keepdims=True)
        qc_msg = qc_msg*check_matrix
        qc_msg -= cq_msg


    # Beliefs
    beliefs = np.sum(cq_msg, axis=0) + llr
    
    # Correction
    correction = np.zeros(len(check_matrix[0]))
    for i in range(len(beliefs)):
        if beliefs[i] < 0:
            correction[i] = 1
    
    correction = np.array(correction)
    correction = np.reshape(correction, (len(correction), 1))
    
    if not np.all(((check_matrix@correction) % 2) - syndrome):
        return correction
    else:
        decoder = Osd(model=circuit,
                       bp_method='product_sum',
                       max_bp_iters=max_iterations,
                       ms_scaling_factor=0.0,
                       osd_method='OSD_CS',
                       osd_order=20)
        sigmoid = nn.Sigmoid()
        correction = decoder.decode(sigmoid(-beliefs))
        return correction