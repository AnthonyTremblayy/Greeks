import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class OptionGreeks:
    def __init__(self, S, T, r, sigma, option_type='call'):
        """
        Parameters used if input is fixed

        S: Spot price of the underlying asset
        T: Time to expiration (in years)
        r: Risk-free interest rate (annual)
        option_type: 'call' or 'put' (option type for Greeks calculations)
        """
        self.S = S  
        self.T = T  
        self.r = r  
        self.sigma = sigma
        self.option_type = option_type

    def compute_delta(self, sigma, S, K, T, r):
        """
        Calculate the delta of an option using the Black-Scholes model.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        if self.option_type == 'call':
            return stats.norm.cdf(d1)  
        elif self.option_type == 'put':
            return stats.norm.cdf(d1) - 1 
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    def compute_gamma(self, sigma, S, K, T, r):
        """
        Calculate the gamma of an option using the Black-Scholes model.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))

        return gamma

    def compute_vega(self, sigma, S, K, T, r):
        """
        Calculate the vega of an option using the Black-Scholes model.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * stats.norm.pdf(d1) * np.sqrt(T)

        return vega 

    def compute_theta(self, sigma, S, K, T, r):
        """
        Calculate the theta of an option using the Black-Scholes model.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.option_type == 'call':
            theta = - (S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        elif self.option_type == 'put':
            theta = - (S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        return theta
    
    # def volatility_smile(S, K, T, base_vol=0.2, skew=0.1):
    #     """
    #     Generate implied volatility using a simple volatility smile model.
    #     The model assumes higher volatility for out-of-the-money options.
        
    #     S: Spot price of the underlying asset
    #     K: Strike price of the option
    #     T: Time to expiration (in years)
    #     base_vol: Base implied volatility (for ATM options)
    #     skew: A factor to adjust for how volatility changes with moneyness.
    #     """
    #     moneyness = K / S
    #     volatility = base_vol + skew * np.abs(moneyness - 1)  # Skew increases as moneyness deviates from 1 (ATM)
        
    #     return volatility

    def plot_greeks(self, variable = 'vol'):
            """
            Plot the effect of volatility or time to expiry on the delta, gamma, vega, and theta of an option.
            """
            moneyness = np.linspace(0.5, 1.5, 100)  # Moneyness range (K/S ratio from 0.5 to 1.5)
            
            if variable == 'vol':
                # Implied volatilities (from 10% to 100%)
                x_values = np.linspace(0.1, 1.0, 500)
                x_label = "IV (%)"
                x_values_name = "Volatility"
                norm = mcolors.Normalize(vmin=0.1, vmax=1.0)
            elif variable == 'ttm':
                # Time to expiration from 1 day to 1 year
                x_values = np.linspace(0.01, 1.0, 750)
                x_label = "Time to Maturity (Years)"
                x_values_name = "Time to Maturity"
                norm = mcolors.Normalize(vmin=0.01, vmax=1.0)
            else:
                raise ValueError("Invalid variable. Use 'vol' or 'ttm'.")

            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            cmap = plt.get_cmap("coolwarm")

            for value in x_values:
                deltas = []
                gammas = []
                vegas = []
                thetas = []
                
                for K in moneyness * self.S:
                    if variable == 'vol':
                        deltas.append(self.compute_delta(value, self.S, K, self.T, self.r))
                        gammas.append(self.compute_gamma(value, self.S, K, self.T, self.r))
                        vegas.append(self.compute_vega(value, self.S, K, self.T, self.r))
                        thetas.append(self.compute_theta(value, self.S, K, self.T, self.r))
                    elif variable == 'ttm':
                        deltas.append(self.compute_delta(self.sigma, self.S, K, value, self.r))
                        gammas.append(self.compute_gamma(self.sigma, self.S, K, value, self.r))
                        vegas.append(self.compute_vega(self.sigma, self.S, K, value, self.r))
                        thetas.append(self.compute_theta(self.sigma, self.S, K, value, self.r))

                color = cmap(norm(value))  # Map the volatility or time to expiry to a color

                # Plot Delta
                axs[0, 0].plot(moneyness, deltas, label=f"{x_values_name} = {value:.2f}", color=color)
                axs[0, 0].set_title(f"Delta ({self.option_type.capitalize()})")
                axs[0, 0].set_xlabel("K/S")
                axs[0, 0].set_ylabel("Δ")
                axs[0, 0].grid(True)

                # Plot Gamma
                axs[0, 1].plot(moneyness, gammas, label=f"{x_values_name} = {value:.2f}", color=color)
                axs[0, 1].set_title(f"Gamma ({self.option_type.capitalize()})")
                axs[0, 1].set_xlabel("K/S")
                axs[0, 1].set_ylabel("Γ")
                axs[0, 1].grid(True)

                # Plot Vega
                axs[1, 0].plot(moneyness, vegas, label=f"{x_values_name} = {value:.2f}", color=color)
                axs[1, 0].set_title(f"Vega ({self.option_type.capitalize()})")
                axs[1, 0].set_xlabel("K/S")
                axs[1, 0].set_ylabel("ν")
                axs[1, 0].grid(True)

                # Plot Theta
                axs[1, 1].plot(moneyness, thetas, label=f"{x_values_name} = {value:.2f}", color=color)
                axs[1, 1].set_title(f"Theta ({self.option_type.capitalize()})")
                axs[1, 1].set_xlabel("K/S")
                axs[1, 1].set_ylabel("Θ")
                axs[1, 1].grid(True)

            # Add colorbar to show the relationship between color and chosen variable (Volatility or Time to Expiry)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Adjust the position and size of the colorbar
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(x_label)

            plt.tight_layout(rect=[0, 0, 0.9, 1]) 
            plt.show()

if __name__ == "__main__":
    S = 100             
    T = 30 / 365   
    r = 0.05   
    sigma = 0.1   

    # Initialize for a call option with volatility varying
    option_call_vol = OptionGreeks(S, T, r, sigma, option_type='call')
    option_call_vol.plot_greeks(variable='vol')

    # Initialize for a call option with time to expiry varying
    option_call_time = OptionGreeks(S, T, r, sigma, option_type='call')
    option_call_time.plot_greeks(variable='ttm')

    # Initialize for a put option with volatility varying
    option_put_vol = OptionGreeks(S, T, r, sigma, option_type='put')
    option_put_vol.plot_greeks(variable='vol')

    # Initialize for a put option with time to expiry varying
    option_put_time = OptionGreeks(S, T, r, sigma, option_type='put')
    option_put_time.plot_greeks(variable='ttm')

    print('done')

