# C14220293 - NIKOLAS HENRIK OWEN CHAN
# C14220331 - YESTOYA L. MINGGUS
# C14220304 - NICHOLAS ANTHONY PRASETYO

import numpy as np

def metode_cramer(A, b):
    n = len(b)
    det_A = np.linalg.det(A)
    
    if abs(det_A) < 1e-10:
        raise ValueError("The matrix A is singular (non-invertible)")
    
    solutions = []
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        det_A_i = np.linalg.det(A_i)
        xi = det_A_i / det_A
        solutions.append(xi)
    
    return solutions


def metode_naive_gauss(A, b):
    n = len(b)
    
    # Convert input arrays to float type
    A = A.astype(float)
    b = b.astype(float)
    
    # Augmented matrix [A | b]
    Ab = np.hstack((A, b.reshape(-1, 1)))
    
    # Forward elimination
    for i in range(n):
        # Divide the current row by the pivot element
        pivot = Ab[i, i]
        Ab[i] /= pivot
        
        # Eliminate below the current row
        for j in range(i + 1, n):
            factor = Ab[j, i]
            Ab[j] -= factor * Ab[i]
    
    # Back substitution
    solutions = np.zeros(n)
    for i in range(n - 1, -1, -1):
        solutions[i] = Ab[i, -1]
        for j in range(i + 1, n):
            solutions[i] -= Ab[i, j] * solutions[j]
    
    return solutions

def main():
    # Contoh Soal:
    # 2x + 3y - z = 7
    # 4x - y + 3z = 2
    # x - 2y + z = 5
    
    A = np.array([[2, 3, -1],
                  [4, -1, 3],
                  [1, -2, 1]])
    b = np.array([7, 2, 5])
    
    try:
        # Menggunakan metode Cramer's
        cramer_solutions = metode_cramer(A, b)
        print("Metode Cramer's Rule:", cramer_solutions)
        
        # Menggunakan metode naive gauss
        gaussian_solutions = metode_naive_gauss(A, b)
        print("Metode Naive Gaussian :", gaussian_solutions)
        
    except ValueError as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
