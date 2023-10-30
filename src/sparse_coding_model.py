import numpy as np

# Class for doing sparse coding
class OlshausenFieldModel:
    def __init__(self, num_inputs, num_units, batch_size, Phi=None,
                 lr_r=1e-2, lr_Phi=1e-2, lmda=5e-3):
        self.lr_r = lr_r # learning rate of r
        self.lr_Phi = lr_Phi # learning rate of Phi
        self.lmda = lmda # regularization parameter
        
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.batch_size = batch_size
        
        # Weights
        if Phi is None:
            Phi = np.random.randn(self.num_inputs, self.num_units).astype(np.float32)
            self.Phi = Phi * np.sqrt(1/self.num_units)
        else:
            self.Phi = Phi

        # activity of neurons
        self.r = np.zeros((self.batch_size, self.num_units))

    def load_dict(self, d):
        self.lr_r = d['lr_r']
        self.lr_Phi = d['lr_Phi']
        self.lmda = d['lmda']
        self.num_inputs = d['num_inputs']
        self.num_units = d['num_units']
        self.batch_size = d['batch_size']
        self.Phi = d['Phi']
        self.r = d['r']

    def to_dict(self):
        return {
            'lr_r': self.lr_r,
            'lr_Phi': self.lr_Phi,
            'lmda': self.lmda,
            'num_inputs': self.num_inputs,
            'num_units': self.num_units,
            'batch_size': self.batch_size,
            'Phi': self.Phi,
            'r': self.r
        }
    
    def set_lr(self, lr):
        self.lr_Phi = lr
        self.lr_r = lr
    
    def initialize_states(self, batch_size=None):
        bs = self.batch_size if batch_size is None else batch_size
        self.r = np.zeros((bs, self.num_units))
        
    def normalize_rows(self):
        self.Phi = self.Phi / np.maximum(np.linalg.norm(self.Phi, 
            ord=2, axis=0, keepdims=True), 1e-8)

    # thresholding function of S(x)=|x|
    def soft_thresholding_func(self, x, lmda):
        return np.maximum(x - lmda, 0) - np.maximum(-x - lmda, 0)

    def calculate_total_error(self, error, include_sparsity=True):
        #recon_error = np.mean(error**2)
        recon_error = np.mean(error**2, axis=1)
        sparsity_r = 0
        if include_sparsity:
            sparsity_r = self.lmda*np.mean(np.abs(self.r)) 
        return recon_error + sparsity_r
        
    def __call__(self, inputs, training=True):
        # Updates                
        error = inputs - self.r @ self.Phi.T
        
        r = self.r + self.lr_r * error @ self.Phi
        self.r = self.soft_thresholding_func(r, self.lmda)

        if training:  
            error = inputs - self.r @ self.Phi.T
            dPhi = error.T @ self.r
            self.Phi += self.lr_Phi * dPhi
            
        return error, self.r

    def get_prediction(self):
        return self.r @ self.Phi.T

    def get_Phi(self):
        return self.Phi

    def get_r(self):
        return self.r

    def clone(self, batch_size):
        cloned_model = OlshausenFieldModel(self.num_inputs, self.num_units, 
            batch_size, self.Phi, self.lr_r, self.lr_Phi, self.lmda)
        return cloned_model

    def clone_with_different_lambda(self, batch_size, lmda):
        cloned_model = OlshausenFieldModel(self.num_inputs, self.num_units, 
            batch_size, self.Phi, self.lr_r, self.lr_Phi, lmda)
        return cloned_model
    
def _update_latents(
        model: OlshausenFieldModel, batch_inputs, nt_max: int, 
        mode: str, eps: float, verbose: bool = False):
    # Input a new batch until latent variables are converged 
    r_tm1 = model.r # set previous r (t minus 1)
    error = None

    for t in range(nt_max):
        if verbose:
            print(f'\t{t + 1} of {nt_max}')

        # Update r without update weights 
        error, r = model(batch_inputs, training=False)
        dr = r - r_tm1 

        # Compute norm of r
        dr_norm = np.linalg.norm(dr, ord=2) / (eps + np.linalg.norm(r_tm1, ord=2))
        r_tm1 = r # update r_tm1

        # Check convergence of r, then update weights
        if t > 20 and dr_norm < eps:
            if mode == 'train':
                error, r = model(batch_inputs, training=True)
            break

        # If failure to convergence, break and print error
        if t >= nt_max-2: 
            print("Failed to converge")
            break
    
    return error

def evaluate(model: OlshausenFieldModel, inputs, batch_size: int, nt_max: int, eps: float):
    num_batches = int(np.ceil(inputs.shape[0] / batch_size))

    prediction = np.zeros(inputs.shape)
    error_list = []
    batch_error_list = []

    for i in range(num_batches):
        print(f'{i+1} of {num_batches}')
        i0 = i * batch_size
        i1 = min(inputs.shape[0], i0 + batch_size)
        batch = inputs[i0:i1, :] - np.mean(inputs[i0:i1, :])

        model.initialize_states(batch_size=(i1 - i0))
        error = _update_latents(model, batch, nt_max, 'valid', eps)

        errors = model.calculate_total_error(error, include_sparsity=False)
        prediction[i0:i1, :] = model.get_prediction()
        error_list.extend(errors) # Append errors
        batch_error_list.append(np.mean(errors))

    return error_list, batch_error_list, prediction
    
def train(
        model: OlshausenFieldModel, inputs, num_iter=1000, nt_max=1000, 
        batch_size=250, eps=1e-2, verbose=False):

    assert(len(inputs.shape) == 2)

    error_list = []
    batch_error_list = []
    num_samples = len(inputs)

    for iter_ in range(num_iter):
        index = np.random.choice(num_samples, batch_size)
        batch_inputs = inputs[index] - np.mean(inputs[index])
        
        model.initialize_states() # reset states
        model.normalize_rows() # normalize weights

        error = _update_latents(model, batch_inputs, nt_max, 'train', eps, verbose)
        errors = model.calculate_total_error(error, include_sparsity=False)

        error_list.extend(errors) # Append errors
        batch_error_list.append(np.mean(errors))

        last_err = np.mean(
            batch_error_list[max(0, len(batch_error_list) - 100):]
        )

        print('{} of {} ({:.3f}, {:.3f})'.format(iter_+1, num_iter, np.mean(errors), last_err))    
    
    return error_list, batch_error_list