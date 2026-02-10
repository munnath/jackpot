import torch
import torch.optim as optim
from tqdm import tqdm

def reorder_list(i, n):
    """
    Reorder the list of indices of the profile likelihood values 

    Parameters
    ----------
    i : int
        start the list from i.
    n : int
        lenght of the list.

    Returns
    -------
    The list [i, i-1, i-2, ..., 1, 0, i+1, i+2, ..., n-2, n-1]

    """
    return list(range(i, n)) + list(range(i))[::-1]

def profile_likelihood(model, x0, param_index, param_values,
                       num_steps=100, method = "LBFGS",
                       lr = None, tol_change = 1e-5):
    
    assert method in ["Adam", "LBFGS"]
    
    i_begin = int(torch.argmin((param_values - x0[param_index])**2))
    n_vals = len(param_values)
    
    profile_likelihoods = [0. for _ in range(n_vals)]
    y0 = model(x0)
    
    # Create a mask to exclude the fixed parameter from optimization
    mask = torch.ones_like(x0, dtype=torch.bool)
    mask[param_index] = False

    indices = reorder_list(i_begin, n_vals)
    x_prev = x0.detach().clone()
    
    ### MAIN LOOP FOR EACH POINT OF THE PROFILE
    for i_val in (t:=tqdm(indices)):
        
        # INITIALIZE THE PARAMETER VALUE
        if i_val == i_begin+1:
            params = x0.detach().clone()
        else:
            params = x_prev.detach().clone()
        params[param_index] = param_values[i_val]
        
        # Parameters to optimize
        params_to_optimize = params[mask].clone()
        params_to_optimize.requires_grad_(True)
        
        if method == "Adam":
            if lr == None:
                lr = 1e-3
            optimizer = optim.Adam([params_to_optimize], lr=lr)  
            
            for _ in range(num_steps):
                optimizer.zero_grad()
                
                # Reconstruct the full parameter vector
                current_params = params.clone()
                current_params[mask] = params_to_optimize
                
                # Forward pass
                output = model(current_params) 
                loss = torch.sum((output - y0)**2)
                loss.backward(retain_graph = False)
                t.set_description(f"grad_norm:{params_to_optimize.grad.norm():.3e}, loss:{torch.sqrt(loss.detach()):.3e}")
                optimizer.step()
        elif method == "LBFGS":
            if lr == None:
                lr = 1.
            optimizer = optim.LBFGS([params_to_optimize], lr=lr, 
                                    line_search_fn="strong_wolfe",
                                    history_size=10, 
                                    tolerance_grad=tol_change,
                                    tolerance_change=tol_change
                                    )  
            
            def closure():
                # Reconstruct the full parameter vector
                current_params = params.clone()
                current_params[mask] = params_to_optimize
                
                output = model(current_params) 
                loss = torch.sum((output - y0)**2)
                loss.backward(retain_graph = False)
                t.set_description(f"iter:{i_step}, grad_norm:{params_to_optimize.grad.norm():.3e}, loss:{torch.sqrt(loss.detach()):.3e}")
                return loss
            
            for i_step in range(num_steps):
                optimizer.zero_grad()
                # Forward pass
                loss = optimizer.step(closure)

                state = optimizer.state[optimizer._params[0]]
                prev_loss = state.get('prev_loss', 1e10)  
                tol_chg = optimizer.param_groups[0]['tolerance_change']
                
                if torch.abs(loss - prev_loss) < tol_chg:
                    break
            
            # Save the optimizers values
            with torch.no_grad():
                current_params = params.clone()
                current_params[mask] = params_to_optimize.clone()
                output = model(current_params) 
                loss = torch.sqrt(torch.sum((output - y0)**2) + 1e-64)
                snr =  10 * (torch.log10(torch.sum((y0)**2)) - 2 * torch.log10(loss))
                x_discr = torch.linalg.norm((x0 - current_params))
                #print(f"x discrepancy: {x_discr:.3e}, y snr: {snr:.3f}")
        
        # Compute the profile likelihood (negative of the loss)
        profile_likelihoods[i_val] = loss.item()
        
        #Empty the memory
        optimizer.zero_grad()
    
    return profile_likelihoods