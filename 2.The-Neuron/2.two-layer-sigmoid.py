# Writing a complex Neural Network from scratch (with activation functions)

import numpy as np

# Sigmoid Activatiom Function:
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid Derivative:
def sigmoid_derivative(x):
    return x * (1 - x)

# Foward Pass:
def forward_pass(x1, x2, w1, w2, w3):
    z1 = (x1 * w1 + x2 * w2)
    h1 = sigmoid(z1)
    
    z2 = (h1 * w3)
    y_hat = sigmoid(z2)
    
    return z1, h1, z2, y_hat

# Loss Function:
def loss_function(y_hat, y):
    return 0.5 * (y_hat - y) ** 2

# Gradient Calculation:
def compute_gradients(x1, x2, w1, w2, w3, y):
    z1, h1, z2, y_hat = forward_pass(x1, x2, w1, w2, w3)
    
    dL_dy_hat = y_hat - y
    dy_hat_dz2 = y_hat * (1 - y_hat)
    dL_dz2 = dL_dy_hat * dy_hat_dz2
    dL_dw3 = dL_dz2 * h1
    dh1_dz1 = sigmoid_derivative(h1)
    dL_dz1 = dL_dz2 * w3 * dh1_dz1
    dL_dw1 = dL_dz1 * x1
    dL_dw2 = dL_dz1 * x2
    
    return dL_dw1, dL_dw2, dL_dw3
    

# Training Loop:
def train_network(x1, x2, y, w1_init, w2_init, w3_init, alpha=0.01, epochs=10):
    w1, w2, w3 = w1_init, w2_init, w3_init
    
    for epoch in range(epochs):
        z1, h1, z2, y_hat = forward_pass(x1, x2, w1, w2, w3)
        loss = loss_function(y_hat, y)
        dL_dw1, dL_dw2, dL_dw3 = compute_gradients(x1, x2, w1, w2, w3, y)
        
        print(f"""
            Iteration {epoch+1:2d}
            Loss: {loss:.4f}
            w1: {w1:.4f}
            w2: {w2:.4f}
            w3: {w3:.4f}
            y_hat: {y_hat:.4f}
        """)

        # Update Weights:
        w1 -= alpha * dL_dw1
        w2 -= alpha * dL_dw2
        w3 -= alpha * dL_dw3
    
    return w1, w2, w3, loss

# Main Function:
if __name__ == "__main__":
    x1 = 1.0
    x2 = 2.0
    y = 1.0
    w1_init = 0.10
    w2_init = -0.30
    w3_init = 0.50
    alpha = 0.01
    epochs = 1000
    
    w1, w2, w3, loss = train_network(x1, x2, y, w1_init, w2_init, w3_init, alpha, epochs)
    
    print(f"Final Loss: {loss:.4f}")
    print(f"Final w1: {w1:.4f}")
    print(f"Final w2: {w2:.4f}")
    print(f"Final w3: {w3:.4f}")
    print("Training complete!")