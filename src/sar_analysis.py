import numpy as np
import xarray as xr
import src.equations_sar as eq


def sar_analysis(sar_ds, zi_ref, wdir_ambiguity, wdir_ref = None):
    """
    Function calculate various statistics from input sar dataset. 
    """

    label = 'Wind Streaks'          # 'Wind Streaks' or 'Micro Convective Cell', for latter adds 90 deg wind-dir. offset
    freq_max = 1 / 600              # maximum frequency in bandpass  [m-1]
    freq_min = 1 / 3000             # minimum frequency in bandpass  [m-1]
    freq_lower_lim = 1 / 300        # maximum frequency for spectral troughs in inertial subrange [m-1]
    tiles = 2                       # number of tiles along x- and y-axis over which to average
    interpolation = 'linear'        # polar to cartesian interpolation method
    zi_ref                          # reference Boundary layer depth value from ERA5
    wdir_ref                        # a-priory wind direction, e.g. from ERA5. If no value given its ignored
    wdir_ambiguity                  # wind direction used to resolve 180 degree ambiguity
    dissip_rate = 1                 # dimensionless energy dissip. rate, approx. between [0.5-2.5] (Kaimal et al,  1976)
    z = 10                          # measurement height

    # prepare dataset by detrending, filling nans and converting units/coordinates
    sar_ds = eq.ds_prepare(sar_ds)

    # -- compute cartesian spectrum of sigma0
    cartesian_spectrum_sigma0, var_cartesian_sigma0, var_beyond_nyquist_sigma0 = eq.ds_cartesian_spectrum(sar_ds,
                                                                                                          smoothing=True,
                                                                                                          parameter='sigma0_detrend',
                                                                                                          scaling='density',
                                                                                                          detrend=None,
                                                                                                          window=None,
                                                                                                          window_correction=None)

    # -- interpolate cartesian sigma0 spectrum to polar spectrum
    max_f = cartesian_spectrum_sigma0.f_range.max().values * 1
    PolarSpec_sigma_0 = eq.polar_interpolation(cartesian_spectrum_sigma0, max_f, x_coord='f_range', y_coord='f_azimuth',
                                               Nt=720, Nf=600, interpolation=interpolation)

    # -- use polar sigma0 spectrum to compute wind direction and corresponding wind field from sigma0
    sar_ds, sum_theta, angle_pre_conversion, energy_dir, energy_dir_range, wdir = eq.ds_windfield(sar_ds,
                                                                                                  PolarSpec_sigma_0,
                                                                                                  wdir_ambiguity=wdir_ambiguity,
                                                                                                  wdir_ref=wdir_ref,
                                                                                                  label=label,
                                                                                                  freq_max=freq_max,
                                                                                                  freq_min=freq_min)

    # -- compute cartesian windfield spectrum
    cartesian_spectrum, var_cartesian, var_beyond_nyquist = eq.ds_cartesian_spectrum(sar_ds,
                                                                                     smoothing=False,
                                                                                     parameter='windfield',
                                                                                     scaling='density',
                                                                                     detrend='constant',
                                                                                     window='hann',
                                                                                     window_correction='True')

    # -- calculate averaged spectra for images subdivided into N x N tiles
    da_polar_mean, da_polar_plot_mean = eq.tiled_spectra(ds=sar_ds, parameter='windfield', tiles=tiles, interpolation=interpolation)
    da_polar_sigma_mean, _ = eq.tiled_spectra(ds=sar_ds, parameter='sigma0', tiles=tiles, interpolation=interpolation)

    # -- calculate parameters from averaged spectra
    var_windfield = sar_ds['windfield'].var().values * 1
    theta_spacing = da_polar_mean.theta.spacing * np.pi / 180  # in radian
    frequency_spacing = da_polar_mean.f.spacing

    # -- compute statistics from averaged spectra
    beam1, beam2, var_windfield, beams, var_bandpass, var_highpass, var_lowpass, var_bandpass_beam, var_polar, var_beam, \
        polar_effect, window_effect, frac_lowpass, frac_highpass, frac_bandpass, frac_beam, density_beam, density_bandpass,\
        density_beam_bandpass \
        = eq.spectral_calculations(da_polar_mean, theta_spacing, frequency_spacing, var_windfield, var_cartesian,
                                   var_beyond_nyquist, angle_pre_conversion=angle_pre_conversion, freq_max=freq_max,
                                   freq_min=freq_min)

    # -- compute friction velocity of wind field in loop 1
    U_n = sar_ds.windfield.median().values * 1
    u_star, z_0, Cdn = eq.loop1(U_n=U_n, z=z)

    # -- compute atmospheric parameters from wind field in loop 2
    sigma_u, L, B, w_star, w_star_normalised_deviation, corr_fact, H, spectral_peak, spectral_valley, idx_inertial_min, idx_inertial_max, PSD = \
        eq.loop2B(U_n=U_n, u_star=u_star, z_0=z_0, Zi=zi_ref, Cdn=Cdn, PolarSpec=da_polar_mean, z=z,
                  dissip_rate=dissip_rate, freq_max=freq_max, freq_min=freq_min, freq_lower_lim=freq_lower_lim)

    # -- compute frequency normalised and average spectral information of wind field and sigma0
    _, S_windfield_xi_mean, S_windfield_xi_std_norm = eq.da_PSD(da_polar_mean, idx_inertial_max=idx_inertial_max,
                                                                idx_inertial_min=idx_inertial_min)
    _, S_sigma0_xi_mean, S_sigma0_xi_std_norm = eq.da_PSD(da_polar_sigma_mean, idx_inertial_max=idx_inertial_max,
                                                          idx_inertial_min=idx_inertial_min)

    # -- compute contour information of polar spectral PSD
    mean_25th, median_25th, std_25th, mad_25th = eq.contour_stats(da_polar=da_polar_plot_mean, coord='f', coord2='theta', 
                                                                   freq_min=freq_min, freq_max =freq_max, percentile=0.25)
    mean_50th, median_50th, std_50th, mad_50th = eq.contour_stats(da_polar=da_polar_plot_mean, coord='f', coord2='theta', 
                                                                   freq_min=freq_min, freq_max =freq_max, percentile=0.50)
    mean_75th, median_75th, std_75th, mad_75th = eq.contour_stats(da_polar=da_polar_plot_mean, coord='f', coord2='theta', 
                                                                   freq_min=freq_min, freq_max =freq_max, percentile=0.75)
    
    # -- calculate misc. info of sar
    time_imagette = sar_ds.start_date
    incidence_avg = sar_ds.incidence.mean().values * 1
    mean_ground_heading = sar_ds.ground_heading.mean().values * 1
    lon_sar = sar_ds.longitude.mean().values * 1
    lat_sar = sar_ds.latitude.mean().values * 1

    row = [time_imagette, lat_sar, lon_sar, U_n, wdir, incidence_avg, mean_ground_heading, energy_dir_range,
        energy_dir, window_effect, polar_effect, var_cartesian, var_windfield, var_polar, var_bandpass, var_highpass,
        var_lowpass, var_beam, var_bandpass_beam, var_beyond_nyquist, frac_beam, frac_bandpass, frac_lowpass,
        frac_highpass, density_beam, density_bandpass, density_beam_bandpass,u_star, z_0, Cdn, sigma_u, L, B, w_star, 
        w_star_normalised_deviation, corr_fact, H, spectral_peak, spectral_valley, mean_25th, median_25th, std_25th, 
        mad_25th, mean_50th, median_50th, std_50th, mad_50th, mean_75th, median_75th, std_75th, mad_75th,
        S_windfield_xi_mean, S_windfield_xi_std_norm, S_sigma0_xi_mean, S_sigma0_xi_std_norm,
            ]
    
    return row


if __name__ == '__main__':
    file_name = '/export/home/owen/Documents/scripts/SAR_paper/data/roll.nc'
    wdir_era5 = 110.224240
    zi_era5 =  904.41970
    sar_ds = xr.open_dataset(file_name)
    output = sar_analysis(sar_ds, zi_ref = zi_era5, wdir_ambiguity = wdir_era5, wdir_ref = None)
    print(output)
