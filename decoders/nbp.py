import torch
import torch.nn as nn
import os
import stim
import numpy as np
from .tensor_tools import DEM_Matrices, DEM_to_matrices
from .osd import Osd

class Nbp(nn.Module):
    
    def __init__(self,
                 circuit: stim.Circuit=None,
                 layers: int = 20,
                 batch_size: int = 1,
                 loss_function: str = 'binary_cross_entropy',
                 weights: str = None,
                 osd_method : str = 'OSD_CS',
                 osd_order : int = 20):
        
        super().__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layers = layers
        self.batch_size = batch_size
        self.loss_function = loss_function
        
        self.circuit = circuit
        self.dem = self.circuit.detector_error_model(decompose_errors=False)
        
        self.matrices = DEM_to_matrices( self.dem )
        
        self.H = self.matrices.check_matrix.to(self.device)
        self.L = self.matrices.logical_matrix.to(self.device)
        self.llrs = self.matrices.llrs.to(self.device)
    
        self.m, self.n = self.H.size()
        
        if not weights:
            self.ini_weights_as_one()
        else:
            self.load_weights(weights)
            
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus(beta=1.0, threshold=50)
        self.softmax = nn.Softmax(dim=0)
        
        self.osd = Osd(model=circuit,
                       bp_method='product_sum',
                       max_bp_iters=self.layers,
                       ms_scaling_factor=0.0,
                       osd_method=osd_method,
                       osd_order=osd_order)
            
    def ini_weights_as_one(self):
        
        self.weights_llr = []
        self.weights_de = []
        
        self.marg_weights_llr = []
        self.marg_weights_de = []
        
        self.rhos = []
        self.residual_weights = []
        
        for _ in range(self.layers):
            
            self.weights_de.append(torch.ones_like(self.H, dtype=float, requires_grad=True, device=self.device))
            self.weights_llr.append(torch.ones_like(self.llrs, dtype=float, requires_grad=True, device=self.device))
            
            self.marg_weights_de.append(torch.ones_like(self.H, dtype=float, requires_grad=True, device=self.device))
            self.marg_weights_llr.append(torch.ones_like(self.llrs, dtype=float, requires_grad=True, device=self.device))
        
        self.residual_weights.append(torch.zeros(self.layers, dtype=float, requires_grad=True, device=self.device))
        self.rhos.append(torch.ones(self.layers, dtype=float, requires_grad=True, device=self.device))
                
    def save_weights(self, path: str):
        
        if not os.path.exists(path):
            os.makedirs(path)
    
        file_de = 'weights_de.pt'
        file_llr = 'weights_llr.pt'
        
        file_marg_de = 'marg_weights_de.pt'
        file_marg_llr = 'marg_weights_llr.pt'
        
        file_residuals = 'residual_weights.pt'
        file_rhos = 'rhos.pt'
        
        torch.save(self.weights_de, os.path.join(path, file_de))
        torch.save(self.weights_llr, os.path.join(path, file_llr))
        
        torch.save(self.marg_weights_de, os.path.join(path, file_marg_de))
        torch.save(self.marg_weights_llr, os.path.join(path, file_marg_llr))
        
        torch.save(self.residual_weights, os.path.join(path, file_residuals))
        torch.save(self.rhos, os.path.join(path, file_rhos))
            
    def load_weights(self, path: str):
        
        file_de = 'weights_de.pt'
        file_llr = 'weights_llr.pt'
        
        file_marg_de = 'marg_weights_de.pt'
        file_marg_llr = 'marg_weights_llr.pt'
        
        file_residuals = 'residual_weights.pt'
        file_rhos = 'rhos.pt'
        
        self.weights_de = torch.load(os.path.join(path, file_de), map_location=torch.device(self.device))
        self.weights_llr = torch.load(os.path.join(path, file_llr), map_location=torch.device(self.device))
        
        self.marg_weights_de = torch.load(os.path.join(path, file_marg_de), map_location=torch.device(self.device))
        self.marg_weights_llr = torch.load(os.path.join(path, file_marg_llr), map_location=torch.device(self.device))
        
        self.residual_weights = torch.load(os.path.join(path, file_residuals), map_location=torch.device(self.device))
        self.rhos = torch.load(os.path.join(path, file_rhos), map_location=torch.device(self.device))
        
    def update_error_nodes(self, incoming_messages, weights_llr, weights_de):
        
        outgoing_messages = torch.zeros_like(incoming_messages, dtype=float, device=self.device)
        
        weighted_messages = incoming_messages * weights_de
        
        outgoing_messages += self.H * self.llrs * weights_llr
        outgoing_messages += torch.sum(weighted_messages, dim=1, keepdim=True)
        outgoing_messages *= self.H
        outgoing_messages -= incoming_messages
        
        return outgoing_messages
    
    def update_detector_nodes(self, incoming_messages, syndromes):
        
        divider = torch.tanh(incoming_messages / 2)
        divider = torch.where(divider == 0, torch.tensor(1.0), divider)
        divider = torch.clamp(divider, -1 + 1e-6, 1 - 1e-6)

        multiplicator = torch.pow(-1, syndromes).to(self.device) * self.H
        
        outgoing_messages = 2*torch.atanh(torch.prod(divider, dim=2, keepdim=True) / divider)
        outgoing_messages *= multiplicator
        
        return outgoing_messages
    
    def compute_beliefs(self, detector_to_error_messages, marg_weights_llr, marg_weights_de):
        
        weighted_messages = detector_to_error_messages * marg_weights_de
        beliefs = torch.sum(weighted_messages, dim=1)
        beliefs += self.llrs*marg_weights_llr
        
        return beliefs
    
    def infer_predictions(self, beliefs):
        
        predictions = torch.zeros_like(beliefs, dtype=float, device=self.device)
        predictions[beliefs < 0] = 1
        predictions = predictions.int()
        
        return predictions
    
    def soft_vectors(self, beliefs):
        return self.sigmoid(-beliefs)
    
    def loss(self, beliefs, errors):
        
        if self.loss_function == 'binary_cross_entropy':
            loss = self.softplus(beliefs)
            loss -= (1 - errors) * beliefs
            loss = torch.sum(loss, dim=1)
            
        if self.loss_function == 'He=s':
            e = errors + self.sigmoid(-beliefs)
            loss = self.H.double() @ e.T
            loss = torch.abs(torch.sin(np.pi * loss / 2))
            loss = torch.sum(loss, dim=0)
            
        if self.loss_function == '[HL]e=s':
            e = errors + self.sigmoid(-beliefs)
            H_L = torch.cat((self.H, self.L), dim=0)
            loss = H_L.double() @ e.T
            loss = torch.abs(torch.sin(np.pi * loss / 2))
            loss = torch.sum(loss, dim=0)
            
        return loss
    
    def forward(self, syndromes, errors):
        
        rhos_normalised = self.softmax(torch.cat(self.rhos)).squeeze()
        loss_array = torch.zeros((self.batch_size, self.layers), dtype=float, device=self.device)
        messages_en_to_dn = torch.zeros((self.batch_size, self.m, self.n), dtype=float, device=self.device)
        messages_dn_to_en = torch.zeros((self.batch_size, self.m, self.n), dtype=float, device=self.device)
        
        for i in range(self.layers):
            
            assert not torch.isnan(messages_dn_to_en).any()
            assert not torch.isnan(messages_en_to_dn).any()
            
            messages_en_to_dn = self.update_error_nodes(messages_dn_to_en, self.weights_llr[i], self.weights_de[i])
            residual_messages = self.residual_weights[0][i] * messages_dn_to_en
            messages_dn_to_en = self.update_detector_nodes(messages_en_to_dn, syndromes) + residual_messages
            beliefs = self.compute_beliefs(messages_dn_to_en, self.marg_weights_llr[i], self.marg_weights_de[i])
            loss_array[:, i] = self.loss(beliefs, errors) * rhos_normalised[i]
        
        loss_array = loss_array
        loss = torch.sum(loss_array, dim=1)
        loss = torch.sum(loss, dim=0) / self.batch_size
        
        return loss
    
    def decode(self, syndrome):
        
        messages_en_to_dn = torch.zeros((1, self.m, self.n), dtype=float, device=self.device)
        messages_dn_to_en = torch.zeros((1, self.m, self.n), dtype=float, device=self.device)
        
        for i in range(self.layers):
            
            messages_en_to_dn = self.update_error_nodes(messages_dn_to_en, self.weights_llr[i], self.weights_de[i])
            residual_messages = self.residual_weights[0][i] * messages_dn_to_en
            messages_dn_to_en = self.update_detector_nodes(messages_en_to_dn, syndrome) + residual_messages 
            beliefs = self.compute_beliefs(messages_dn_to_en, self.marg_weights_llr[i], self.marg_weights_de[i])
        
        error_guess = self.infer_predictions(beliefs)
        prediction = ( self.L @ error_guess.T ) % 2
        
        return prediction
    
    def decode_osd(self, syndrome):
        
        messages_en_to_dn = torch.zeros((1, self.m, self.n), dtype=float, device=self.device)
        messages_dn_to_en = torch.zeros((1, self.m, self.n), dtype=float, device=self.device)
        
        for i in range(self.layers):
            
            messages_en_to_dn = self.update_error_nodes(messages_dn_to_en, self.weights_llr[i], self.weights_de[i])
            residual_messages = self.residual_weights[0][i] * messages_dn_to_en
            messages_dn_to_en = self.update_detector_nodes(messages_en_to_dn, syndrome) + residual_messages 
            beliefs = self.compute_beliefs(messages_dn_to_en, self.marg_weights_llr[i], self.marg_weights_de[i])
            
        error_guess = self.infer_predictions(beliefs)
        syndrome_guess = (self.H @ error_guess.T) % 2
        syndrome_guess = syndrome_guess.reshape_as(syndrome)
        
        if torch.equal(syndrome_guess, syndrome):
            prediction = ( self.L @ error_guess.T ) % 2
        else:
            soft_vector = self.soft_vectors(beliefs)
            prediction = self.osd.decode(soft_vector.detach().numpy())
            prediction = torch.tensor(prediction)
        return prediction