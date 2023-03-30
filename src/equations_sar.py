import cv2
import numpy as np
import xarray as xr
import xrft
from scipy import ndimage

# -- custom
import src.cmod5n as cmod5n


def MAD(data):
    # -- Median Absolute Deviation
    median = np.nanmedian(data)
    MAD = np.nanmedian(abs(data - median))
    return MAD


def contour_stats(da_polar, coord, coord2, freq_min, freq_max, percentile):
    # -- compute psd for spectrum with higher frquency than minimum frequency
    spectrum = da_polar.sel({coord:slice(freq_min, np.inf)}) * da_polar[coord] # 
    
    # -- compute reference psd between frequencies of interest
    spectrum_reference = da_polar.sel({coord:slice(freq_min, freq_max)}) * da_polar[coord]
    
    # -- calculate what relative percentage of energy as a function of reference energy
    cumsum_scaled = (spectrum.cumsum(dim=coord) / spectrum_reference.sum() * len(spectrum[coord2]))

    # -- calculate contours in pixels from the centre
    contour = (cumsum_scaled > percentile).argmax(dim=coord)
    
    # -- calculate statistics for contours
    contour_val = contour.values
    mean = np.mean(contour_val)
    median = np.median(contour_val)
    std = np.std(contour_val)
    mad = MAD(contour_val)
    return mean, median, std, mad


def hamming_window(image):
    
    """
    Apply Hamming window on input NRCS data
    """
    #create 1D Hamming windows
    windowx = np.hamming(image.shape[1]).T
    windowy = np.hamming(image.shape[0])
    
    #meshgrid to combine both 1-D filters into 2D filter
    windowX, windowY = np.meshgrid(windowx, windowy)
    window = windowX*windowY
    
    return window


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and weighted standard deviation.
    """
    average = np.average(values, weights = weights)
    variance = np.average((values - average)**2, weights = weights)
    return (average, np.sqrt(variance))


def da_averaging(list_of_dataarrays, list_of_dims):
    """
    Input:
        list_of_dataarrays = list of xr.dataarays, input a list of datarrays which are to be stacked along a new dimension
        list_of_dims = list of str, list of dimensions names perpendicular to which the data ought to be satcked and averaged
    
    Output:
        average of datarraays
    """
    
    # -- create dummy dimensions
    dummy_dims = [dim +'_dummy' for dim in list_of_dims]
    # -- create dictionary from lsit of original and dummy dimensions
    dims_dict = dict(zip(list_of_dims, dummy_dims))
    # -- rename main dimensions in spectra (f and theta) such that adding a dimension to the existing main coordinates will not crash
    da_renamed = [arr.swap_dims(dims_dict) for arr in list_of_dataarrays]
    # -- stack separate polar spectra using a new arbitrary dimension
    da_stack = xr.concat(da_renamed, dim="tile", coords= list_of_dims , compat="override")
    # -- take the average over this arbitrary dimension
    da_mean = da_stack.mean(dim = 'tile')
    # -- add back previous coordinates belonging to main dimensions
    # -- NOTE!: since each tile (data array in the stack) should be clipped/interpolated to the same size, all their coordinates are identical
    coordinates_to_add = {dummy_dim: (dummy_dim, list_of_dataarrays[0][dim].values) for dim, dummy_dim in zip(list_of_dims, dummy_dims)}
    da_mean = da_mean.assign_coords(coordinates_to_add)
    # -- rename back to old names
    da_mean = da_mean.rename(dict(zip(dummy_dims, list_of_dims)))
    # -- add old attributes
    for dim in list_of_dims: # for each dimension in the list of dimensions
        try:
            # add every attribute (add in try loop in case no attributes are contained)
            for attribute in list(list_of_dataarrays[0][dim].attrs.keys()):
                da_mean[dim].attrs[attribute] = list_of_dataarrays[0][dim].attrs[attribute]
            pass
        except Exception:
            pass
        
    return da_mean


def da_PSD(da_polar_spec, idx_inertial_max = 0, idx_inertial_min = -1):
    """
    calculates 1D spectrum from 2D density spectrum and returns frequency normalised and averaged PSD
    
    Input: 
        da_polar_spec: dataArray, density spectrum with coordinates theta (anugular) and f (frequency) with 
        idx_inertial_max: int, index belonging to maximum inertial subrange spectral amplitude (inertial subrange peak)
        idx_inertial_min: int, index belonging to minimum inertial subrange spectral amplitude (inertial subrange trough)
        
    Output: 
        PSD: dataArray, 1D PSD    
        S: float, frequency normalised and averaged spectral amplitude within inertial subrange
        S_std: float, standard deviation of spectrum to compute S
    """
    
    # -- calculate density parameters
    d_theta = da_polar_spec.theta.spacing * np.pi / 180   # angular resolution from degrees to radians
    PSD_f = da_polar_spec.f
    
    # -- multiplied times frequency yields density spectrum, sum over theta to get 1D, next multiply times spacings to go to PSD in A^3/f^2
    PSD = (da_polar_spec*PSD_f).sum(dim = 'theta') * np.prod([PSD_f.spacing, d_theta]) / PSD_f.spacing
    
    # -- multiply PSD with 5/3 such that inertial subrange becomes flat (only over indexes within the inertial subrange)
    S_scaled = (PSD*PSD_f**(5/3))[idx_inertial_max:idx_inertial_min]
    
    # calculate frequency weighted average and std for frequency-multiplied PSD
    wavelengths_inertial = 1/PSD_f.values[idx_inertial_max:idx_inertial_min]
    S, S_std = weighted_avg_and_std(S_scaled, wavelengths_inertial)
    S_std_norm = S_std / np.median(S_scaled)
    
    return PSD, S, S_std_norm


def ds_prepare(sar_ds):
    """
    This function takes a sar dataset (with coordinates in pixels), fills the NaN's (where possible), detrends and adds coordinates in metres 
    
    Input: 
        sar_ds: sar dataset containing a 'sigma0' field with coordinates 'atrack' and 'xtrack'  
        
    Output:
        sar_ds: dataset updated to contain nanfilled 'sigma0_nanfill' and 'sigma0_detrend'
    """
    
    # slice such that nans are excluded from raw data
    sar_ds = sar_ds[{'atrack':slice(5,-5), 'xtrack':slice(5,-5)}]
    
    # remove Nans by interpolation
    interp_01 = sar_ds.sigma0.interpolate_na(dim = 'atrack', method= 'linear', fill_value= 'extrapolate')
    sar_ds['sigma0_nanfill'] = interp_01.interpolate_na(dim = 'xtrack', method= 'linear', fill_value= 'extrapolate')
    
    # detrending using kernels
    pixel_size = np.mean([sar_ds.pixel_atrack_m, sar_ds.pixel_xtrack_m]) # find average pixel size in metres
    filter_size = 10000 # median filter has size of 10000 metres
    pixels = int(filter_size // pixel_size) # find number of pixels in median filter size
    if pixels%2!= 1: # has to be odd
        pixels+=1
    
    # gaussian kernel (ok time and performance)
    sigma0_trend = cv2.GaussianBlur(sar_ds.sigma0_nanfill.values, (pixels , pixels), 0)
    
    # remove trend, divide by and window
    sar_ds['sigma0_detrend_plotting'] = (sar_ds.sigma0_nanfill - sigma0_trend) / sigma0_trend
    sar_ds['sigma0_detrend'] = sar_ds.sigma0_detrend_plotting * hamming_window(sar_ds.sigma0)
    
    # Convert to units in meter and calculate 2D spectra 
    sar_ds = sar_ds.assign_coords({'atrack_m': ('atrack', np.arange(0, len(sar_ds.atrack))*sar_ds.pixel_atrack_m), \
                                   'xtrack_m': ('xtrack', np.arange(0, len(sar_ds.xtrack))*sar_ds.pixel_xtrack_m)})
    sar_ds = sar_ds.swap_dims({'atrack':'atrack_m','xtrack':'xtrack_m'})
    sar_ds = sar_ds.drop(['atrack', 'xtrack'])
    
    return sar_ds

def ds_windfield(sar_ds, PolarSpecSigma0_smooth, wdir_ambiguity, label, wdir_ref = None, freq_max = 1 / 600, freq_min = 1 / 3000):
    """
    This function derives the orientation of greatest energy within a bandpass of the smoothed polar spectrum
    Next the wind field is calculated using the found orientation with respect to the radar sensor (i.e. range direction) using CMOD5.N
    
    Input:
        sar_ds: sar dataset from 'ds_prepare'
        PolarSpecSigma0_smooth: smoothed polar sigma0 dataset from 'ds_to_polar'
        label: wether to consider the structure to be rolls or cells (adds 90 deg offset)
        wdir_ambiguity: a priory wind direction with which to resolve 180 degree ambiguity
        wdir_ref: wind direction used to compute the wind field, used if you want to use a-priori knowledge not just for resolving 180 deg ambiguity
                  e.g. use wdir_ref = wdir_era5 if you want to use the era5 wind direction as the wind direction for wind-field computations
        freq_max: frequency of bandpass shorter wavelength in 1/m, high frequencies can contain swell
        freq_min: frequency of bandpass upper wavelength in 1/m, low frequencies can contian mesoscale activity
        
    Output:
        sar_ds: updated with a find field with coordinates in metre
        sum_theta: sum of energy per theta within bandpass
        angle_pre_conversion: angle in polar spectrum with greatest energy 
    """

    bandpass_subset = PolarSpecSigma0_smooth.sel(f = slice(freq_min, freq_max))

    # sum of all energy per theta 
    sum_theta = (bandpass_subset*bandpass_subset.f).sum(dim = 'f')

    # find peak orientation in smoothed spectrum
    angle_pre_conversion = bandpass_subset['theta'].isel({'theta':  sum_theta.argmax()}).values
    
    # convert peak to azimuth direction
    angle = ( -(angle_pre_conversion - 90) + 360) % 360

    # convert angle from w.r.t. azimuth to w.r.t. range
    energy_dir_range =  (angle  - 90 ) % 360 

    # convert angle from w.r.t. azimuth to w.r.t. North
    energy_dir =  ((sar_ds.ground_heading.mean().values  + 360 ) % 360 + angle + 360 ) % 360

    # energy direction is perpendicular to wind direction for Wind Streaks
    if label == 'Wind Streaks':
        offset = 90 # approximate
    if label == 'Micro Convective Cell':
        offset = 0 # approximate

    wdir = (energy_dir + offset + 360) % 360

    # use ERA5 wdir as reference to resolve 180 degree ambiguity        
    diff = abs(wdir_ambiguity - wdir)
    if (diff >= 90) & (diff <= 270):
        wdir = (wdir + 180) % 360
        angle = (angle + 180) % 360
        energy_dir_range = (energy_dir_range + 180) % 360        

    # add correction for convection type and convert from azimuth to range by + 90 (radar viowing direction)
    phi = (angle + offset - 90 ) % 360 
    
    # if you want to use a hardcoded value to calculate the wind field
    # converts reference wind dir w.r.t. North to wind direction w.r.t. range
    if wdir_ref is not None:
        phi = 360 - ((sar_ds.ground_heading.mean().values  + 360 ) % 360 - wdir_ref + 90 ) % 360

    # calculate wind field
    windfield = cmod5n.cmod5n_inverse(sar_ds.sigma0.values, phi , sar_ds.incidence.values, CMOD5 = False, iterations = 10)
        
    sar_ds['windfield'] = (('atrack_m', 'xtrack_m'), windfield)
    
    return sar_ds, sum_theta, angle_pre_conversion, energy_dir, energy_dir_range, wdir



def ds_cartesian_spectrum(sar_ds, smoothing = False, parameter = 'windfield', scaling  = 'density', detrend='constant', window = 'hann', window_correction = 'True'):
    """
    Function to calculate spectral characteristics of the windfield contained in 'sar_ds'
    
    Input:
        sar_ds: xarray dataset
        parameter: parameter in dataset for which spectrum has to be calculated
        scaling: scaling of power spectrum (e.g. density or energy)
        detrend: type of detrending in power-spectrum calculation (constant removes mean a.k.a. 0th frequency)
        window: type of window to apply in power-spectrum calculation 
        window_correction: whether to correct amplitude of spectrum for hypothetical energy loss

    Output:
        CartSpec: cartesian spectrum
        var_cartesian: variance (energy) of cartesian image. To be used to determine effect of wind correction from spectral calculation
        var_beyond_nyquist: variance in the corners of cartesian grid that falls outisde of poalr spectrum
    """
    # -- calculate power spectral density of windfield using (and correcting for) a Hanning window
    CartSpec = xrft.power_spectrum(sar_ds[parameter] , scaling = scaling, detrend = detrend, window = window, window_correction = window_correction)  

    # -- optionally smooth spectrum with gaussian filter
    if smoothing:
        sigma = [2,2] # arbitrarily selected
        spectrum_smoothed = ndimage.gaussian_filter(CartSpec, sigma, mode='constant')
        ds_CartSpec = xr.Dataset({})
        ds_CartSpec['spectrum'] = CartSpec
        ds_CartSpec['spectrum_smoothed'] = (('freq_atrack_m', 'freq_xtrack_m'), spectrum_smoothed)
        CartSpec = ds_CartSpec['spectrum_smoothed']

    # -- add and swap dimensions
    CartSpec = CartSpec.assign_coords({'f_range':CartSpec.freq_xtrack_m, 'f_azimuth': CartSpec.freq_atrack_m})
    CartSpec = CartSpec.swap_dims({'freq_xtrack_m':'f_range','freq_atrack_m':'f_azimuth'})
    CartSpec.f_range.attrs.update({'spacing':CartSpec.freq_xtrack_m.spacing})
    CartSpec.f_azimuth.attrs.update({'spacing':CartSpec.freq_atrack_m.spacing})

    # -- calculate total energy inside cartesian spectrum, dividing density spcetrum by spacing which is equal to the variance 
    var_cartesian = CartSpec.sum().values * np.prod([CartSpec.f_range.spacing, CartSpec.f_azimuth.spacing])

    # -- calculate energy that falls outside polar spectrum but within Cartesian
    x, y = np.meshgrid(CartSpec.f_range, CartSpec.f_azimuth)
    indexes_beyond_nyquist = np.where(np.sqrt(x**2 + y**2) > CartSpec.f_range.max().values, 1, 0)
    var_beyond_nyquist = (CartSpec * indexes_beyond_nyquist).sum().values * np.prod([CartSpec.f_range.spacing, CartSpec.f_azimuth.spacing])
    
    return CartSpec, var_cartesian, var_beyond_nyquist

def polar_interpolation(cartesian_spectrum, max_f, x_coord = 'f_range', y_coord = 'f_azimuth', Nt = 720, Nf = 300, interpolation = 'linear'):
        """
        Calculates polar spectrum from cartesian spectrum
        
        input:
            cartesian_spectrum: input dataraay with 
            max_freq: maximum frequency to interpolate to
            Nt: number of thetas (angles) to interpolate to
            Nf: number of frequencies to interpolate to
            
        output: 
            PolarSpec: polar spectrum with new coordinates
        """
        # create theta grid (from 0 to 360 degrees)
        theta_spacing = 360 / Nt
        theta = np.linspace(0, 360 - theta_spacing, Nt)
        theta = xr.DataArray(theta, dims='theta', coords={'theta':theta})
    
        # create frequency grid
        fspacing = float(max_f / Nf)
        f = np.linspace(0, max_f, Nf)   # use linspace here or np.arange()
        f = xr.DataArray(f, dims='f', coords={'f':f})
    
        # calculate equivalent coordinates of polar system in cartesian system
        fx_plot = f*np.cos(np.deg2rad(theta))
        fy_plot = f*np.sin(np.deg2rad(theta))
    
        # interpolate from cartesian spectrum to polar spectrum
        PolarSpec = cartesian_spectrum.interp(coords = {x_coord: fx_plot, y_coord: fy_plot}, assume_sorted = True, kwargs = {'fill_value':None}, method = interpolation)
        PolarSpec.f.attrs.update({'spacing':fspacing})
        PolarSpec.theta.attrs.update({'spacing':theta_spacing})
        
        return PolarSpec

def spectral_calculations(polar_spectrum, theta_spacing, frequency_spacing, var_windfield, var_cartesian, var_beyond_nyquist, angle_pre_conversion = 0, freq_max = 1 / 600, freq_min = 1 / 3000):
        """
        Calculates information from a polar interpolated spectrum
        
        input:
            polar_spectrum: spectrum with coordinates f (frequency) and theta (angle, in degrees) from which calculations are made
            theta_spacing: spacing between theta (angular resolution of polar spectrum) in radian
            frequency_spacing: spacing between frequencies 
            var_windfield: energy in raw wind field
            var_cartesian: energy in cartesian spectrum of wind field (after windowig and correction)
            var_beyond_nyquist: energy cut from polar spectrum during interpolaation from cartesian to polar
            angle_pre_conversion: angle in bandpassed polar spectrum with greatest energy (used to calculate densities in spectrum)
            freq_max: frequency of bandpass shorter wavelength in 1/m, high frequencies can contain swell
            freq_min: frequency of bandpass upper wavelength in 1/m, low frequencies can contian mesoscale activity
            
        output: 
            beam1: datarray with polar spectrum of beam 1
            beam2: datarray with polar spectrum of beam 2
            var_windfield: np.var(windfield). Variance unadultarated by polar conversion or windows
            beams: dataset with polar information for beams centred around 'angle_pre_conversion'
            var_bandpass: variance of polar spectrum contained within bandpass 
            var_highpass: variance of polar spectrum contained within frequencies greater than maximum specified (i.e. swell) including frequencies beyon nyquist but within cartesian 
            var_lowpass: variance of polar spectrum contained within frequencies smaller than minimum specified (i.e. mesoscale)
            var_bandpass_beam: variance of polar spectrum contained within bandpass section of the beam
            var_polar: variance contained in polar spectrum. Expected to be slightly less than var_windfield due to windowing and polar interpolation
            var_beam: variance contained in beams of polar spectrum
            density_beam: relative amount of energy in beam compared to average of spectrum (e.g. 1= average 2 = twice average)
            density_bandpass: relative amount of energy in bandpass compared to average of spectrum
            density_beam_bandpass: relative amount of energy in bandpass within beam compared to average of bandpass
        """
        
        # calculate total energy within the polar spectrum, depending on the winddowing effect polar_nrj < cartesian_nrj
        polar_nrj = (polar_spectrum*polar_spectrum.f).sum().values * np.prod([frequency_spacing, theta_spacing])
   
        # select range of angles with conditioning incase it passes the 0 or 360 degree boundary (for both sides of the spectrum)
        beam_size = 20
        slice_min = (angle_pre_conversion - beam_size + 360 ) % 360
        slice_max = (angle_pre_conversion + beam_size + 360 ) % 360
        
        angle_pre_conversion_mirror = (angle_pre_conversion + 180 ) % 360
        slice_min_mirror = (slice_min + 180 ) % 360
        slice_max_mirror = (slice_max + 180 ) % 360
   
        # add conditions in case the beam crosses the 0 - 360 boundary 
        # calculated for two seperate beams instead of for one multiplied times 2 in case polar spectrum is not composed of two equal halves
        # beam 1
        # if maximum crosses 360 line
        if (slice_max < angle_pre_conversion) & (slice_min < angle_pre_conversion):
            idx_beam1 = [i[0] for i in np.argwhere( (polar_spectrum.theta.values >= slice_min) | (polar_spectrum.theta.values <= slice_max))]
        # if minimum crosses 360 line
        elif (slice_max > angle_pre_conversion) & (slice_min > angle_pre_conversion):
            idx_beam1 = [i[0] for i in np.argwhere( (polar_spectrum.theta.values >= slice_min) | (polar_spectrum.theta.values <= slice_max))]
        # if neither crosses 360 line   
        elif (slice_max > angle_pre_conversion) & (slice_min < angle_pre_conversion):
            idx_beam1 = [i[0] for i in np.argwhere( (polar_spectrum.theta.values >= slice_min) & (polar_spectrum.theta.values <= slice_max))]
          
        # beam 2 (e.g. beam 1 but shifted 180 degree)
        # if maximum crosses 360 line   
        if (slice_max_mirror < angle_pre_conversion_mirror) & (slice_min_mirror < angle_pre_conversion_mirror):
            idx_beam2 = [i[0] for i in np.argwhere( (polar_spectrum.theta.values >= slice_min_mirror) | (polar_spectrum.theta.values <= slice_max_mirror))]
        # if minimum crosses 360 line
        elif (slice_max_mirror > angle_pre_conversion_mirror) & (slice_min_mirror > angle_pre_conversion_mirror):
            idx_beam2 = [i[0] for i in np.argwhere( (polar_spectrum.theta.values >= slice_min_mirror) | (polar_spectrum.theta.values <= slice_max_mirror))]
        # if neither crosses 360 line     
        elif (slice_max_mirror > angle_pre_conversion_mirror) & (slice_min_mirror < angle_pre_conversion_mirror):
            idx_beam2 = [i[0] for i in np.argwhere( (polar_spectrum.theta.values >= slice_min_mirror) & (polar_spectrum.theta.values <= slice_max_mirror))]
   
        # select beam subset (i.e. points within few degrees of angle of greatest variation)
        beam1 = polar_spectrum[:, idx_beam1]
        beam2 = polar_spectrum[:, idx_beam2]
        beams = xr.concat([beam1, beam2], "theta") # add both beams into a single dataraay
   
        polar_nrj_beams = (beams*beams.f).sum().values * np.prod([frequency_spacing, theta_spacing])
   
        # calculate energy within different parts of the polar spectrum
        spectrum_bandpass = polar_spectrum.sel(f = slice(freq_min, freq_max))
        spectrum_highpass = polar_spectrum.sel(f = slice(freq_max, 1))  # all energy in wavelengths shorter than minimum wavelength, including that which falls outside polar but still within cartesian
        spectrum_lowpass = polar_spectrum.sel(f = slice(0, freq_min)) # all energy in wavelengths longer than the maximum, should be energy in mesoscale
        var_bandpass = (spectrum_bandpass*spectrum_bandpass.f).sum().values * np.prod([frequency_spacing, theta_spacing])
        var_highpass = (spectrum_highpass*spectrum_highpass.f).sum().values * np.prod([frequency_spacing, theta_spacing]) + var_beyond_nyquist
        var_lowpass = (spectrum_lowpass*spectrum_lowpass.f).sum().values * np.prod([frequency_spacing, theta_spacing])
   
        spectrum_bandpass_beam = beams.sel(f = slice(freq_min, freq_max))
        var_bandpass_beam = (spectrum_bandpass_beam*spectrum_bandpass_beam.f).sum().values * np.prod([frequency_spacing, theta_spacing])
   
        var_polar = polar_nrj
        var_beam = polar_nrj_beams
        
        polar_effect = var_polar / var_cartesian
        window_effect = var_cartesian / var_windfield
        low_pass_frac = var_lowpass / var_polar
        high_pass_frac = var_highpass / var_polar
        bandpass_frac = var_bandpass / var_polar
                
        frac_beam = var_beam / var_polar
        density_beam = frac_beam / (beam_size * 4 / 360)
        density_bandpass = var_bandpass / var_polar
        density_beam_bandpass = var_bandpass_beam / var_bandpass / (beam_size * 4 / 360)

        return beam1, beam2, var_windfield, beams, var_bandpass, var_highpass, var_lowpass, var_bandpass_beam, var_polar, var_beam, \
            polar_effect, window_effect, low_pass_frac, high_pass_frac, bandpass_frac, frac_beam, density_beam, density_bandpass, density_beam_bandpass \

def loop1(U_n, z = 10):
    """
    First loop of Young's approach. Calculates surface stress Tau , friction velocity u* and roughness length z_0
    based on neutral wind speed input'
    
    Input:
        U_n: Neutral wind speed at z-meter elevation m/s
        z: elevation of wind speed, 10m for CMOD5.N
        
    Output:
        u*: friction velocity in m/s
        z_0: friction length in m
        C_dn: neutral drag coefficient
        
    """
    karman = 0.40                           # Karman constant
    Charnock = 0.011                        # Charnock constant
    g = 9.8                                 # Gravitational acceleration, m/s**2
    z = z                                   # measurements height, 10 metres for CMOD5.N 
    rho_air = 1.2                           # kg/m**3, 1.2 for 20 degrees, 1.25 for 10 degreess                           
    T = 20                                  # temperature in Celcius
    
    # kinematic viscosity of air
    nu = 1.326 * 10**(-5) *(1 + (6.542 * 10**(-3))* T + (8.301 * 10**(-6)) * T**2 - (4.840 * 10**(-9)) * T**3) # m**2/s
    
    # prepare loop of 15 iterations
    iterations = 15
    A_u_star = np.ones(iterations)    # m/s
    A_surface_stress = np.ones(iterations)       # kg/ m / s**2  [Pa]
    A_Cdn = np.ones(iterations)                  # 
    A_z_0 = np.ones(iterations)                  # m

    # Initialise loop with windspeed and iterate with refined estimates of neutral drag coefficient
    for i in range(iterations):
        if i > 0:
            A_u_star[i] = np.sqrt(A_Cdn[i-1] * U_n**2)
            A_surface_stress[i] = rho_air * A_u_star[i]**2
            A_z_0[i] = (Charnock * A_u_star[i]**2) / g + 0.11 * nu / A_u_star[i]
            A_Cdn[i] = (karman / np.log( z / A_z_0[i]) )**2
    
    # calculate stress field based on retrieved constants and windspeed estimates
    surface_stress = rho_air * A_Cdn[-1] *  U_n**2 

    # save friction velocity and friction length based on mean stress field and neutral drag coefficient
    u_star = np.sqrt(surface_stress / rho_air)
    z_0 = (Charnock * u_star**2) / g + 0.11 * nu / u_star
    Cdn = A_Cdn[-1]
    
    return u_star, z_0, Cdn

def loop2B(U_n, u_star, z_0, Zi, Cdn, PolarSpec, z = 10, dissip_rate = 1, freq_max = 1 / 600, freq_min = 1 / 3000, freq_lower_lim = 1/300):
    """
    Second loop of Young's approach. Requires output of loop 1. Recalculates wind field using stability correction.
    Similar to loop two but instead of using wind variance of entire field only uses inertial subrange
    Outputs recalculated parameters, obukhov Length L and kinematic heat flux B
    
    Input:
        U_n: Neutral wind speed at z-meter elevation m/s
        u_star: u_star from loop 1
        z_0: z_0 from loop 1
        Zi: a-priori value for Boundary layer height
        Cdn: neutral drag coefficient from loop 1
        PolarSpec: dataset containing high res. interpolated 'sigma0_detrend' density spectrum on polar grid. To be used for spectral calculations
        label: expected convection form, cells is standard, rolls result in slight modification
        z: wind field measurement height
        dissip_rate: approximately between 0.5 and 2.5 (kaimal et al,  1976)
        freq_max: frequency of bandpass shorter wavelength in 1/m, high frequencies can contain swell
        freq_min: frequency of bandpass upper wavelength in 1/m, low frequencies can contain mesoscale activity
        freq_lower_lim = frequency beyond which no local minima are found to match with ERA5

    Output:
        sigma_u: estimated wind-field variance
        L: Obukhov length in meters
        B: Kinematic heat flux in metres
        w_star: convective velocity scale in m/s
        w_star_normalised_deviation: std of w_star values in inertial subrange
        corr_fact: stability correction factor
        H: heat flux (from Ocean into atmosphere)
        spectral_peak: wavelength of the spectral peak. In literature this is multiplied times a factor to get Zi
        spectral_valley: wavelength of the spectral valley, i.e. assumed to be lowest point in inertial subrange
        idx_inertial_min: index corresponding to the inertial subrange minimum in the 1D spectrum derived from the 2D (smoothed) spectrum
        idx_inertial_max: index corresponding to the inertial subrange maximum in the 1D spectrum derived from the 2D (smoothed) spectrum

    ###### Zi: lowest inversion height following Kaimal et al 1976, Sikora et al 1997
    
    """

    
    # find indexes in smoothed spectrum belonging to peak and trough of inertial subrange.
    # idx_start_min = np.argmin(abs(PolarSpec_smooth.f.values - freq_max)) # 1/freq_max metres = maximum value for lower limit
    idx_start_max = np.argmin(abs(PolarSpec.f.values - freq_min)) # no peaks considered with wavelengths greater than the bandpass
    
    lower_limit = 1/freq_lower_lim 
    # --------------- NOT USING SMOOTHED SPECTRUM FOR PEAKS ----------------#
    # find highest point in smoothed spectrum, i.e. intertial subrange peak
    idx_inertial_max = ((PolarSpec*PolarSpec.f**(5/3)).sel(f = slice(freq_min, freq_max)).sum(dim = 'theta')).argmax(dim=['f'])['f'].values * 1 + idx_start_max

    # find lowest point in smoothed spectrum, i.e. intertial subrange trough
    x_axis = 1 / PolarSpec.f.values 
    idx_start_min = np.argmin(abs(PolarSpec.f.values - 1/x_axis[idx_inertial_max]))
    idx_inertial_min = ((PolarSpec*PolarSpec.f**(1)).sel(f = slice(1/x_axis[idx_inertial_max], 1/lower_limit)).sum(dim = 'theta')).argmin(dim=['f'])['f'].values * 1 + idx_start_min

    # retry finding index of lowest point if it failed before.
    # by setting the power of the frequency lower the higher frequencies are "lifted" less therefore it becomes easier (but perhaps less reliable) to find local minima
    if idx_inertial_min == idx_inertial_max:
        idx_inertial_min = ((PolarSpec*PolarSpec.f**(0.75)).sel(f = slice(freq_min, 1/lower_limit)).sum(dim = 'theta')).argmin(dim=['f'])['f'].values * 1 + idx_start_min
    if idx_inertial_min == idx_inertial_max:
        idx_inertial_min = ((PolarSpec*PolarSpec.f**(0.5)).sel(f = slice(freq_min, 1/lower_limit)).sum(dim = 'theta')).argmin(dim=['f'])['f'].values * 1 + idx_start_min
    if idx_inertial_min == idx_inertial_max:
        idx_inertial_min = ((PolarSpec*PolarSpec.f**(0.25)).sel(f = slice(freq_min, 1/lower_limit)).sum(dim = 'theta')).argmin(dim=['f'])['f'].values * 1 + idx_start_min

    pi = 3.1415926535
    z = z                                  # measurements height, 10 metres for CMOD5.N 
    karman = 0.40                          # Karman constant
    T_v = 293                              # virtual potential temperature in Kelvin
    g = 9.8                                # gravitational acceleration [m/s2]
    rho_air = 1.2                          # air density [kg/m3]
    Cp = 1005                              # heat capacity air
    iterations = 10
    kolmogorov = 0.5
    dissip_rate = dissip_rate              # 0.6 is low and 2 about average according to fig.4 in Kaimal et al. (1976)

    # takes entire polar spectrum, averages along all theta angles, multiples density spectrum by area and divides by
    # frequency spacing to arrive at the Power Spectral Density (PSD) needed for further calculations
    d_theta = PolarSpec.theta.spacing * np.pi / 180   # angular resolution from degrees to radians
    PSD = (PolarSpec*PolarSpec.f).sum(dim = 'theta').values * np.prod([PolarSpec.f.spacing, d_theta]) / PolarSpec.f.spacing

    # Invoke Taylor's hypothesis to convert spatial frequency to temporal
    PSD /= U_n
    
    x_axis = 1 / PolarSpec.f.values    # spatial wavelengths in metre, x_axis[idx_inertial_max] should be 1.5 * Zi 
    
    spectral_peak = x_axis[idx_inertial_max]        # peak wavelength in spectrum
    spectral_valley = x_axis[idx_inertial_min]    # valley wavelength in spectrum
    
    # create arrays to store loop results
    w_star_normalised_deviation = np.ones(iterations)
    w_star = np.ones(iterations)
    B = np.ones(iterations)
    L = np.ones(iterations)
    x = np.ones(iterations)
    Psi_m = np.ones(iterations)
    corr_fact = np.ones(iterations)
    
    for i in range(iterations):
        if i > 0:
            # spatial wavelengths within selected part of inertial subrange
            Lambda = x_axis[idx_inertial_max:idx_inertial_min]

            # select PSD within inertial subrange and apply correction factor
            S = PSD[idx_inertial_max:idx_inertial_min] * corr_fact[i-1]**1
            # correction factor not squared! FFT requires squared correction but afterwards PSD is also divided by
            # wind speed, thus dividing again by C_corr_fact[i-1]
            
            # calculate corrected wind speed (corrected for non-neutrality)
            U_corr = U_n * corr_fact[i-1]
            
            # calculate cyclic frequency in per second
            n = 1 / Lambda * U_corr
            
            # calculate dimensionless frequency
            fi = n * Zi / U_corr

            # calculate convective velocity (w_star) for all frequencies in S
            pre_w_star = np.sqrt((2 * pi)**(2/3) * fi**(2/3) * n * S / (kolmogorov * dissip_rate**(2/3)))

            # determine weights and calculate weighted mean and std of convective velocity scale

            weights = x_axis[idx_inertial_max:idx_inertial_min] / np.min(x_axis[idx_inertial_max:idx_inertial_min])
            w_star[i] = weighted_avg_and_std(pre_w_star, weights)[0]
            w_star_normalised_deviation[i] = weighted_avg_and_std(pre_w_star, weights)[1] / np.median(pre_w_star)

            # calculate kinematic heat flux
            B[i] =  (w_star[i]**3 * T_v) / (g * Zi)
            
            # Monin Obukhov similarity theory
            L[i] = - (u_star**3 * T_v) / (B[i] * karman * g)

            # structure function and empirical constant from Young et. al. 2000 (typo corrected)
            x[i] = (1 + 16 * abs(z / L[i]))**0.25
            Psi_m[i] = np.log(((1 + x[i]**2) / 2)**2) - 2 * np.arctan(x[i]) + pi / 2
        
            # stability correction factor from Young et. al. 2000
            corr_fact[i] = 1 - (Psi_m[i] * np.sqrt(Cdn)) / karman

    # calculate final outputs to return at the end of function
    sigma_u = u_star * np.sqrt(4 + 0.6 * (-Zi / L[-1])**(2/3))
    H = B[-1] * Cp * rho_air       # heat flux

    return sigma_u, L[-1], B[-1], w_star[-1], w_star_normalised_deviation[-1], corr_fact[-1], H, spectral_peak, spectral_valley, idx_inertial_min, idx_inertial_max, PSD


def tiled_spectra(ds, parameter, tiles = 2, list_of_dims = ['f', 'theta'], interpolation = 'linear'):
    """
    Function to calculate tiled spectra, e.g. for tiles = 2 the input dataray will be split into 2x2 
    
    Input: 
        ds: dataset, input datasat containing with coordinates atrack_m and xtrack_m
        parameter: str,name of parameter in dataarray for which to calculate tiled spectra
        tiles: int, number of tiles in x and y direction
        list_of_dims: list of str, list of dimensions names perpendicular to which the data ought to be stacked and averaged
        
    Output:
        ds_polar_mean: dataArray, averaged spectra of all tiled spectra
        PolarSpectras_plot: dataArray, averaged spectra of all tiled spectra. interpolated to lower resolution for plotting purposes
    """

    # -- find shape of dataset
    shapes = np.array(np.shape(ds[parameter]))
    # -- select fine tiles
    grid_size = min(shapes)//tiles # grid size in pixels
    x_sub, y_sub = shapes // grid_size
    
    # -- storage for in loop 
    PolarSpectras = []
    PolarSpectras_plot = []
    
    for k in range(x_sub):
        for l in range(y_sub):

            # split data into sub tiles
            ds_sub = ds[dict(atrack_m=slice((k)*grid_size, (k+1)*grid_size), xtrack_m=slice((l)*grid_size, (l+1)*grid_size))]
            ds_sub = ds_sub.assign_coords({"atrack_m": ds_sub.atrack_m - np.min(ds_sub.atrack_m)})
            ds_sub = ds_sub.assign_coords({"xtrack_m": ds_sub.xtrack_m - np.min(ds_sub.xtrack_m)})
    
            # -- compute cartesian windfield spectrum for the sub tiles
            cartesian_spectrum_sub, _, _ = ds_cartesian_spectrum(ds_sub, smoothing = False, parameter = parameter, scaling  = 'density', detrend='constant', window = 'hann', window_correction = 'True')
                
            # -- interpolate cartesian spectrum to polar spectrum in coarse and high resolution (the former for calculations, the latter for plotting)
            max_f = cartesian_spectrum_sub.f_range.max().values*1
            PolarSpec_sub = polar_interpolation(cartesian_spectrum_sub, max_f, x_coord = 'f_range', y_coord = 'f_azimuth', Nt = 3600, Nf = 600, interpolation = interpolation)
            PolarSpec_sub_plot = polar_interpolation(cartesian_spectrum_sub, max_f, x_coord = 'f_range', y_coord = 'f_azimuth', Nt = 720, Nf = 300, interpolation = interpolation)
            
            # -- store data for later
            PolarSpectras.append(PolarSpec_sub)
            PolarSpectras_plot.append(PolarSpec_sub_plot)
            
    # -- average and interpolate polar spectra of windfield tiles to a single polar spectra representative of the entire field
    ds_polar_mean = da_averaging(PolarSpectras, list_of_dims=list_of_dims)  
    ds_polar_plot_mean = da_averaging(PolarSpectras_plot, list_of_dims=list_of_dims)  
            
    return ds_polar_mean, ds_polar_plot_mean













