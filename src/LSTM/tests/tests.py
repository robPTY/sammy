import torch
import sys
sys.path.insert(0, '..')

from cell_block import Cell


def test_manual_grad():
    torch.manual_seed(42)
    
    input_dims = 1
    hidden_dims = 5
    output_dims = 1
    
    cell = Cell(input_dims, hidden_dims, output_dims)
    
    x = torch.randn(1, input_dims)
    prev_h = torch.randn(1, hidden_dims)
    prev_ct = torch.randn(1, hidden_dims)
    target = torch.randn(1, output_dims)

    # Perform forward pass 
    cell.zero_grad()
    Xt, ft, it, ct, ot, tanh_cell_state, cell_state, ht, zt = cell.forward(prev_ct, prev_h, x)
    loss_manual = 0.5 * torch.sum((zt - target) ** 2)
    dZ = zt - target
    states_cache = {"Xt": Xt, "ft": ft, "it": it, "ot": ot,
        "candidate": ct, "prev_cst": prev_ct, "cell_state": cell_state,
        "tanh_cell_state": tanh_cell_state, "prev_hidden": prev_h, "ht": ht}
    
    # Run backward pass 
    dht_next = torch.zeros(1, hidden_dims)
    dct_next = torch.zeros(1, hidden_dims)
    dx_manual, dprev_h_manual, dprev_ct_manual = cell.backward(states_cache, dZ, dht_next, dct_next)
    manual_grads = {"dWy": cell.dWy.clone(), "dby": cell.dby.clone(), "dWo": cell.dWo.clone(),
        "dbo": cell.dbo.clone(),"dWf": cell.dWf.clone(),"dbf": cell.dbf.clone(), 
        "dWi": cell.dWi.clone(), "dbi": cell.dbi.clone(),"dWc": cell.dWc.clone(),
        "dbc": cell.dbc.clone()}
    
    # Pytorch Autograd
    x_ag = x.clone().requires_grad_(True)
    prev_h_ag = prev_h.clone().requires_grad_(True)
    prev_ct_ag = prev_ct.clone().requires_grad_(True)
    
    Wf_ag = cell.Wf.clone().requires_grad_(True)
    bf_ag = cell.bf.clone().requires_grad_(True)
    Wi_ag = cell.Wi.clone().requires_grad_(True)
    bi_ag = cell.bi.clone().requires_grad_(True)
    Wc_ag = cell.Wc.clone().requires_grad_(True)
    bc_ag = cell.bc.clone().requires_grad_(True)
    Wo_ag = cell.Wo.clone().requires_grad_(True)
    bo_ag = cell.bo.clone().requires_grad_(True)
    Wy_ag = cell.Wy.clone().requires_grad_(True)
    by_ag = cell.by.clone().requires_grad_(True)
    
    # Autograd forward pass 
    Xt_ag = torch.cat((x_ag, prev_h_ag), dim=1)
    ft_ag = torch.sigmoid(Xt_ag @ Wf_ag + bf_ag)
    cell_state_ag = prev_ct_ag * ft_ag
    it_ag = torch.sigmoid(Xt_ag @ Wi_ag + bi_ag)
    ct_ag = torch.tanh(Xt_ag @ Wc_ag + bc_ag)
    cell_state_ag = cell_state_ag + ct_ag * it_ag
    ot_ag = torch.sigmoid(Xt_ag @ Wo_ag + bo_ag)
    tanh_cell_state_ag = torch.tanh(cell_state_ag)
    ht_ag = ot_ag * tanh_cell_state_ag
    zt_ag = ht_ag @ Wy_ag + by_ag
    loss_ag = 0.5 * torch.sum((zt_ag - target) ** 2)

    loss_ag.backward()
    autograd_grads = {"dWy": Wy_ag.grad, "dby": by_ag.grad, "dWo": Wo_ag.grad, "dbo": bo_ag.grad,
        "dWf": Wf_ag.grad, "dbf": bf_ag.grad, "dWi": Wi_ag.grad, "dbi": bi_ag.grad,
        "dWc": Wc_ag.grad,"dbc": bc_ag.grad}
    
    print("=" * 60)
    print("Gradient Comparison: Manual vs PyTorch Autograd")
    print("=" * 60)
    
    all_passed = True
    tolerance = 1e-5
    
    for name in manual_grads:
        manual = manual_grads[name]
        autograd = autograd_grads[name]
        
        max_diff = torch.max(torch.abs(manual - autograd)).item()
        is_close = torch.allclose(manual, autograd, atol=tolerance)
        
        status = "✓ PASS" if is_close else "✗ FAIL"
        print(f"{name:6s}: max_diff = {max_diff:.2e} {status}")
        
        if not is_close:
            all_passed = False
            print(f"       Manual:   {manual.flatten()[:5].tolist()}...")
            print(f"       Autograd: {autograd.flatten()[:5].tolist()}...")
    
    # Also compare input gradients
    print("-" * 60)
    print("Input Gradients:")
    
    dx_autograd = x_ag.grad
    dprev_h_autograd = prev_h_ag.grad
    dprev_ct_autograd = prev_ct_ag.grad
    
    dx_close = torch.allclose(dx_manual, dx_autograd, atol=tolerance)
    dprev_h_close = torch.allclose(dprev_h_manual, dprev_h_autograd, atol=tolerance)
    dprev_ct_close = torch.allclose(dprev_ct_manual, dprev_ct_autograd, atol=tolerance)
    
    print(f"dx:      max_diff = {torch.max(torch.abs(dx_manual - dx_autograd)).item():.2e} {'✓ PASS' if dx_close else '✗ FAIL'}")
    print(f"dprev_h: max_diff = {torch.max(torch.abs(dprev_h_manual - dprev_h_autograd)).item():.2e} {'✓ PASS' if dprev_h_close else '✗ FAIL'}")
    print(f"dprev_ct: max_diff = {torch.max(torch.abs(dprev_ct_manual - dprev_ct_autograd)).item():.2e} {'✓ PASS' if dprev_ct_close else '✗ FAIL'}")
    
    if not (dx_close and dprev_h_close and dprev_ct_close):
        all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("ALL TESTS PASSED! Manual gradients match PyTorch autograd.")
    else:
        print("SOME TESTS FAILED! Check gradient implementations.")
    
    print("=" * 60)
    
    assert all_passed, "Gradient mismatch detected!"
    return all_passed

def main():
    print("Running gradient verification tests...\n")
    
    try:
        test_manual_grad()
        print("\n✓ All tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
