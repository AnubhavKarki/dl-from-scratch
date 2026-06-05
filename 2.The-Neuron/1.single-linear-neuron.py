# Writing Neural Network from scratch (no activation functions)

import numpy as np

# Forward Pass:
def forward_pass(x, w1, w2):
    h = x * w1
    y_hat = h * w2

    return h, y_hat

# Loss Function:
def loss_function(y_hat, y):
    return 0.5 * (y_hat - y) ** 2

# Gradient Calculation:
def compute_gradients(x, w1, w2, y):
    h, y_hat = forward_pass(x, w1, w2)
    
    dL_dy_hat = y_hat - y
    dL_dw2 = dL_dy_hat * h
    dL_dw1 = dL_dy_hat * (x * w2)
    
    return dL_dw1, dL_dw2
    

# Training Loop:
def train_network(x, y, w1_init, w2_init, alpha=0.01, epochs=10):
    w1, w2 = w1_init, w2_init
    
    for epoch in range(epochs):
        h, y_hat = forward_pass(x, w1, w2)
        loss = loss_function(y_hat, y)
        dL_dw1, dL_dw2 = compute_gradients(x, w1, w2, y)
        
        print(f"""
                Iteration {epoch+1:2d}
                Loss: {loss:.4f}
                w1: {w1:.4f}
                w2: {w2:.4f}
                y_hat: {y_hat:.4f}
            """)
    
        # Update Weights:
        w1 -= alpha * dL_dw1
        w2 -= alpha * dL_dw2
    
    return w1, w2, loss

# Main Function:
if __name__ == "__main__":
    x = 2.0
    y = 20.0
    
    w1_init = 2.0
    w2_init = 0.5
    
    alpha = 0.01
    epochs = 10
    
    w1_final, w2_final, loss = train_network(x, y, w1_init, w2_init, alpha, epochs)
    
    print(f"Final Loss: {loss:.4f}")
    print(f"Final w1: {w1_final:.4f}")
    print(f"Final w2: {w2_final:.4f}")