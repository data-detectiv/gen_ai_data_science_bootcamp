import numpy as np
import pandas as pd

def gradient_descent(x, y, lr=0.01, epochs=3000):
    m = 0.0
    b = 0.0

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_scaled = (x - x_min)/ (x_max - x_min)
    y_scaled = (y - y_min)/ (y_max - y_min)

    for epoch in range(epochs):
        y_pred = m * x_scaled + b
        error = y_scaled - y_pred
        cost = np.mean(error ** 2)

        dm = -2 * np.mean(x_scaled * error)
        db = -2 * np.mean(error)

        b -= db * lr
        m -= dm * lr

        if epoch % 100 == 0:
            print(f" m= {m}, b = {b}, Epoch {epoch} : Cost = {cost}")

    b_org = b * (y_max - y_min) + y_min - m * (y_max - y_min) * x_min / (x_max - x_min)
    m_org = m * (y_max - y_min) / (x_max - x_min)

    print(f"b = {b_org}, m = {m_org}")

    return b_org, m_org

if __name__ == "__main__":
    df = pd.read_csv("home_prices.csv")

    x = df["area_sqr_ft"].to_numpy()
    y = df['price_lakhs'].to_numpy()
    
    print(gradient_descent(x, y))