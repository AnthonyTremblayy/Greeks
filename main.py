import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

class OptionGreeks:
    def __init__(self, S, r, T, option_type='call', vol_surface=None):
        
        """
        Parameters used if input is fixed
        S: Spot price of the underlying asset
        r: Risk-free rate (either a constant (int/float) or a yield curve (dictionary with time to maturity (months) as keys and rates as values))
        T: Time to expiration (in months)
        option_type: 'call' or 'put'
        vol_surface: Optional volatility surface. If None, a volatility smile will be used as a proxy.
        """
        self.S = S  
        self.T = T  
        self.option_type = option_type
        self.vol_surface = vol_surface
        
        if isinstance(r, (int, float)):
            self.r = r  # Constant rate
        elif isinstance(r, dict):
            self.r = self._interpolate_yield_curve(T, r) / 100 # Rate based on yield curve
        else:
            raise ValueError("r must be either a constant or a yield curve (dict)")
        
        if vol_surface is not None:
            self.vol_array = self._interpolate_vol_surface_along_ttm(self.T, vol_surface) / 100
        
    def _interpolate_yield_curve(self, time, yield_curve):
        """
        Interpolates the yield curve to get the yield for any given time to maturity.
        Assumes the input `yield_curve` is a dictionary with time to maturity as keys and rates as values.
        """
        # Extract tenors and rates from the dictionary
        tenors = np.array(list(yield_curve.keys()))
        rates = np.array(list(yield_curve.values()))
        
        # Interpolate to create a function for yield at any time to maturity
        self.yield_curve = interp1d(tenors, rates, kind='cubic')
        
        # Return an interpolated yield value for the input T
        return self.yield_curve(time)
    
    def _interpolate_vol_surface_along_ttm(self, T, vol_surface):
        """
        Interpolates the volatility surface for the given time to maturity T.
        vol_surface: 2D array where each row corresponds to a different time to maturity.
        """
        
        ttm = vol_surface["ttm"]
        vol = vol_surface["vol"]
        vol_interp = interp1d(ttm, vol, kind='cubic', axis=0)

        return vol_interp(T)
    
    def _interpolate_vol_surface_along_moneyness(self, K, vol_surface):
        """
        Interpolates the volatility surface for a given moneyness.
        """
        moneyness = vol_surface["moneyness"]
        vol_interp = interp1d(moneyness, self.vol_array, kind='cubic')

        return vol_interp(K / self.S * 100)
    
    def plot_volatility_smile(self, atm_vol=0.25, curvature=1.5, skew=0.35):
        """
        Plot the volatility smile based on moneyness (K/S).
        """
        K_values = np.linspace(0.8 * self.S, 1.2 * self.S, 100)
        implied_vols = [OptionGreeks.volatility_smile(self.S, K, atm_vol, curvature, skew) for K in K_values]
        plt.figure(figsize=(8, 5))
        plt.plot(K_values / self.S, implied_vols, color='b', linewidth=2)
        plt.axvline(x=1, linestyle="--", color="gray")
        plt.xlabel("Moneyness (K/S)")
        plt.ylabel("Implied Volatility")
        plt.title("Volatility Smile")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_greeks(self, variable = 'vol'):
            """
            Plot the effect of volatility or time to expiry on the delta, gamma, vega, and theta of an option.
            """
            moneyness = np.linspace(0.8, 1.2, 100)  # Moneyness range (K/S ratio from 0.5 to 1.5)
            
            if variable == 'vol':
                # Implied volatilities (from 10% to 100%)
                x_values = np.linspace(0.1, 1.0, 100)
                x_label = "IV (%)"
                x_values_name = "Volatility"
                norm = mcolors.Normalize(vmin=0.1, vmax=1.0)
            elif variable == 'ttm':
                # Time to expiration from 1 day to 1 year
                x_values = np.linspace(0.01, 1.0, 200)
                x_label = "TTM (Years)"
                x_values_name = "Time to Maturity"
                norm = mcolors.Normalize(vmin=0.01, vmax=1.0)
            else:
                raise ValueError("Invalid variable. Use 'vol' or 'ttm'.")
            
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            cmap = plt.get_cmap("coolwarm")

            min_linewidth = 0.3  # Thinnest line
            max_linewidth = 2  # Thickest line

            for value in x_values:
                deltas = []
                gammas = []
                vegas = []
                thetas = []
                
                for K in moneyness * self.S:
                    if variable == 'vol':
                        deltas.append(OptionGreeks.compute_delta(value, self.S, K, self.T, self.r, self.option_type))
                        gammas.append(OptionGreeks.compute_gamma(value, self.S, K, self.T, self.r))
                        vegas.append(OptionGreeks.compute_vega(value, self.S, K, self.T, self.r))
                        thetas.append(OptionGreeks.compute_theta(value, self.S, K, self.T, self.r, self.option_type))

                    elif variable == 'ttm':
                        if self.vol_surface is not None:
                            sigma = self._interpolate_vol_surface_along_moneyness(K, vol_surface=self.vol_surface)
                        else:
                            sigma = OptionGreeks.volatility_smile(self.S, K) #Proxy
                            
                        deltas.append(OptionGreeks.compute_delta(sigma, self.S, K, value, self.r, self.option_type))
                        gammas.append(OptionGreeks.compute_gamma(sigma, self.S, K, value, self.r))
                        vegas.append(OptionGreeks.compute_vega(sigma, self.S, K, value, self.r))
                        thetas.append(OptionGreeks.compute_theta(sigma, self.S, K, value, self.r, self.option_type))
                        
                normalized_value = (value - min(x_values)) / (max(x_values) - min(x_values)) # Normalize value to determine line thickness
                linewidth = max_linewidth - (normalized_value * (max_linewidth - min_linewidth))
                shade = np.sqrt(normalized_value) # More shading for thicker lines
                color = cmap(norm(value))  # Map the volatility or time to expiry to a color

                # Plot Delta
                axs[0, 0].plot(moneyness, deltas, label=f"{x_values_name} = {value:.2f}", color=color, linewidth=linewidth, alpha = shade)
                axs[0, 0].set_title(f"Delta ({self.option_type.capitalize()})")
                axs[0, 0].set_xlabel("K/S")
                axs[0, 0].set_ylabel("Δ")
                axs[0, 0].grid(True)

                # Plot Gamma
                axs[0, 1].plot(moneyness, gammas, label=f"{x_values_name} = {value:.2f}", color=color, linewidth=linewidth, alpha = shade)
                axs[0, 1].set_title(f"Gamma ({self.option_type.capitalize()})")
                axs[0, 1].set_xlabel("K/S")
                axs[0, 1].set_ylabel("Γ")
                axs[0, 1].grid(True)

                # Plot Vega
                axs[1, 0].plot(moneyness, vegas, label=f"{x_values_name} = {value:.2f}", color=color, linewidth=linewidth, alpha = shade)
                axs[1, 0].set_title(f"Vega ({self.option_type.capitalize()})")
                axs[1, 0].set_xlabel("K/S")
                axs[1, 0].set_ylabel("ν")
                axs[1, 0].grid(True)

                # Plot Theta
                axs[1, 1].plot(moneyness, thetas, label=f"{x_values_name} = {value:.2f}", color=color, linewidth=linewidth, alpha = shade)
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
    
    @staticmethod
    def compute_delta(sigma, S, K, T, r, option_type='call'):
        """
        Calculate the delta of an option using the Black-Scholes model.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return stats.norm.cdf(d1)  
        elif option_type == 'put':
            return stats.norm.cdf(d1) - 1 
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
    @staticmethod
    def compute_gamma(sigma, S, K, T, r):
        """
        Calculate the gamma of an option using the Black-Scholes model.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    
    staticmethod
    def compute_vega(sigma, S, K, T, r):
        """
        Calculate the vega of an option using the Black-Scholes model.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * stats.norm.pdf(d1) * np.sqrt(T)
        return vega 
    
    @staticmethod
    def compute_theta(sigma, S, K, T, r, option_type='call'):
        """
        Calculate the theta of an option using the Black-Scholes model.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            theta = - (S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        elif option_type == 'put':
            theta = - (S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        return theta
    
    @staticmethod
    def volatility_smile(S, K, atm_vol=0.25, curvature=1.5, skew=0.35):
        """
        Generate a volatility smile 
        K: Strike price
        atm_vol: Base volatility for ATM options
        curvature: Controls how strong the smile effect is (higher makes the U deeper)
        skew: Controls the tilt (smirk), making OTM puts more volatile than calls.
        """
        moneyness = K / S
        smile = atm_vol + curvature * (moneyness - 1)**2  # U-shape
        smirk = skew * (1- moneyness)  # Tilt effect (positive skews toward higher IV for puts)
        return smile + smirk
    
if __name__ == "__main__":
    
    T = 1/12 # Years
    
    # Market Data -- SPX March 11 2025
    S = 5604.83             
    
    # Volatility Surface
    vol_surface = {
        "moneyness": np.array([80.0, 90.0, 95.0, 97.5, 100.0, 102.5, 105.0, 110.0, 120.0]),  # Moneyness values (%)
        "ttm": [1/12, 2/12, 3/12, 6/12, 9/12, 1],  # Expiry periods in years
        "vol": np.array([  # Volatility surface for each expiry
            [43.82, 32.93, 28.46, 26.51, 24.27, 21.92, 19.80, 16.95, 23.11],  # 1M
            [37.31, 28.89, 25.29, 23.56, 21.83, 20.05, 18.34, 15.64, 17.03],  # 2M
            [33.90, 27.00, 23.98, 22.52, 21.01, 19.47, 17.94, 15.45, 14.48],  # 3M
            [29.54, 24.65, 22.44, 21.32, 20.18, 18.99, 17.77, 15.54, 13.13],  # 6M
            [27.39, 23.45, 21.65, 20.75, 19.83, 18.88, 17.89, 15.92, 13.19],  # 9M
            [26.02, 22.72, 21.16, 20.38, 19.58, 18.76, 17.95, 16.24, 13.57]   # 1Y
        ])
    }
    
    # Yield Curve (in years)
    yield_curve = {
        0.25 / 12: 4.332, 
        0.5 / 12: 4.329, 
        0.75 / 12: 4.329, 
        1 / 12: 4.332, 
        2 / 12: 4.330, 
        3 / 12: 4.303,
        4 / 12: 4.254, 
        5 / 12: 4.207, 
        6 / 12: 4.163, 
        9 / 12: 4.030, 
        12 / 12: 3.931
    }

    # Call option with volatility varying
    option_call_vol = OptionGreeks(S, yield_curve, T, option_type='call')
    option_call_vol.plot_greeks(variable='vol')
    
    # Call option with time to expiry varying
    option_call_time = OptionGreeks(S, yield_curve, T, option_type='call', vol_surface=vol_surface)
    option_call_time.plot_greeks(variable='ttm')
    
    # Put option with volatility varying
    option_put_vol = OptionGreeks(S, yield_curve, T, option_type='put')
    option_put_vol.plot_greeks(variable='vol')

    # Put option with time to expiry varying
    option_put_time = OptionGreeks(S, yield_curve, T, option_type='put', vol_surface=vol_surface)
    option_put_time.plot_greeks(variable='ttm')

    print('done')