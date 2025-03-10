import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class OptionGreeks:
    def __init__(self, S, T, r):
        """
        S: Spot price of the underlying asset
        K: Strike price of the option
        T: Time to expiration (in years)
        r: Risk-free interest rate (annual)
        """
        self.S = S  
        self.T = T  
        self.r = r  

    def compute_delta_call(self, sigma, S, K, T, r):
        """
        Calculate the delta of a call option using the Black-Scholes model.
        
        sigma: Implied volatility (annual)
        S: Spot price of the underlying asset
        K: Strike price of the option
        T: Time to expiration (in years)
        r: Risk-free interest rate (annual)
        
        Returns: Delta of the call option
        """

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return stats.norm.cdf(d1)
    
    def compute_gamma_call(self, sigma, S, K, T, r):
        """
        Calculate the gamma of a call option using the Black-Scholes model.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

    def compute_vega_call(self, sigma, S, K, T, r):
        """
        Calculate the vega of a call option using the Black-Scholes model.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * stats.norm.pdf(d1) * np.sqrt(T)
        return vega

    def compute_theta_call(self, sigma, S, K, T, r):
        """
        Calculate the theta of a call option using the Black-Scholes model.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        theta = - (S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        return theta

    def plot_delta_call(self):
        """
        Plot the effect of implied volatility on the delta of a call option.
        """
        # Moneyness range (K/S ratio from 0.5 to 1.5)
        moneyness = np.linspace(0.5, 1.5, 100)

        # Implied volatilities (from 10% to 100%)
        IV = np.linspace(0.1, 1.0, 500)

        # Plot for each volatility level
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        cmap = plt.get_cmap("coolwarm") 
        norm = mcolors.Normalize(vmin=0.1, vmax=1.0)

        # Plot for each volatility level
        for i, sigma in enumerate(IV):
            deltas = [self.compute_delta_call(sigma, self.S, K, self.T, self.r) for K in moneyness * self.S]
            gammas = [self.compute_gamma_call(sigma, self.S, K, self.T, self.r) for K in moneyness * self.S]
            vegas = [self.compute_vega_call(sigma, self.S, K, self.T, self.r) for K in moneyness * self.S]
            thetas = [self.compute_theta_call(sigma, self.S, K, self.T, self.r) for K in moneyness * self.S]
            
            color = cmap(norm(sigma))  # Map the sigma value to a color

            # Plot Delta
            axs[0, 0].plot(moneyness, deltas, label=f"Volatility = {sigma*100:.1f}%", color=color)
            axs[0, 0].set_title("Delta")
            axs[0, 0].set_xlabel("K/S")
            axs[0, 0].set_ylabel("Δ")
            axs[0, 0].grid(True)

            # Plot Gamma
            axs[0, 1].plot(moneyness, gammas, label=f"Volatility = {sigma*100:.1f}%", color=color)
            axs[0, 1].set_title("Gamma")
            axs[0, 1].set_xlabel("K/S")
            axs[0, 1].set_ylabel("Γ")
            axs[0, 1].grid(True)

            # Plot Vega
            axs[1, 0].plot(moneyness, vegas, label=f"Volatility = {sigma*100:.1f}%", color=color)
            axs[1, 0].set_title("Vega")
            axs[1, 0].set_xlabel("K/S")
            axs[1, 0].set_ylabel("ν")
            axs[1, 0].grid(True)

            # Plot Theta
            axs[1, 1].plot(moneyness, thetas, label=f"Volatility = {sigma*100:.1f}%", color=color)
            axs[1, 1].set_title("Theta")
            axs[1, 1].set_xlabel("K/S")
            axs[1, 1].set_ylabel("Θ")
            axs[1, 1].grid(True)

        # Add colorbar to show the relationship between color and volatility
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Adjust the position and size of the colorbar
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Implied Volatility (%)")

        plt.tight_layout(rect=[0, 0, 0.9, 1]) 
        plt.show()

if __name__ == "__main__":
    S = 100             
    T = 30 / 365   
    r = 0.05       

    option = OptionGreeks(S, T, r)
    option.plot_delta_call()

    print('done')


