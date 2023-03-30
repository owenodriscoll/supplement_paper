from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from matplotlib import pyplot as plt 
import pandas as pd 
import numpy as np

def outlier_detector(df, column_key_start, column_key_end, pca_comp = 0.80, neighbours = 100, plot_PCA = False):
    """
    Function using SKlearn's 'LocalOutlierFactor' to detec outliers within a specififed range of the input df's columns
    
    Input:
        df: dataframe from which to select sub-dataframe
        column_key_start: name of column to start 
        column_key_end: nume of column end including column itself
        pca_comp: number of Principal components to include (int for number, float for fraction explained variance)
        neighbours: number of neighbours to conisder in localOutlier
        plot_PCA: if true will plot first two principal componenets and colour outliers
    """
    

    # -- Select specified columns
    idx_measurements = list(df.keys()).index(column_key_start)
    idx_measurements_end = list(df.keys()).index(column_key_end) + 1  # plus 1 because [5:10] does not include idx 10
    data = df.iloc[:, idx_measurements : idx_measurements_end]
    
    # -- change processing depending on whether PCA should be invoked or not
    if (type(pca_comp) == float) | (type(pca_comp) == int) :

        # -- apply standard scaler (outliers will remain)
        x = StandardScaler().fit_transform(data)
        
        # -- select fraction or number of Principal componenets and create PCA
        pca = PCA(n_components = pca_comp)
        
        # -- apply PCA
        X = pca.fit_transform(x)
        
    else:
        X = data
        
    # -- create outlier detector
    outlier_detector = LocalOutlierFactor(n_neighbors=neighbours)
    
    # -- apply detector on data
    inliers = outlier_detector.fit_predict(X)
    
    # -- create df with inliers only
    df_outliers_removed = df[inliers==1] # inliers = 1
    
    if plot_PCA == True:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c = inliers)
        plt.xlabel(r"Principal Component 1"); plt.ylabel(r"Principal Component 2")
        plt.title("Outliers")
        plt.show()
        
    return df_outliers_removed
    
    
def wdir_errors(wdir_estimate, wdir_validation):
    Error_wdir = wdir_estimate - wdir_validation
    Error_wdir = np.where(Error_wdir>=270, Error_wdir - 360, Error_wdir)
    Error_wdir = np.where(Error_wdir>=90, Error_wdir - 180, Error_wdir)
    Error_wdir = np.where(Error_wdir<=-270, Error_wdir + 360, Error_wdir)
    Error_wdir = np.where(Error_wdir<=-90, Error_wdir + 180, Error_wdir)
    return Error_wdir
    
    
def sample_data(df, class_col, num_samples, random_state=42):
    # Count number of instances of each class
    class_counts = df[class_col].value_counts()
    
    # Remove classes with less instances than num_samples
    valid_classes = class_counts[class_counts >= num_samples].index.tolist()
    
    # Filter dataframe to only include rows with valid classes
    filtered_df = df[df[class_col].isin(valid_classes)]
    
    # Sample num_samples instances from each valid class
    samples = []
    for class_name in valid_classes:
        class_df = filtered_df[filtered_df[class_col] == class_name]
        sampled_class_df = class_df.sample(n=num_samples, random_state=random_state)
        samples.append(sampled_class_df)
    
    return pd.concat(samples)
    
    
def envelope(df, param_x, param_y, begin, end, steps =25, log = True):
    """
    function to derive the median and quantiles for a pointcloud from a df with two specified parameters
    """
    
    placeholder = df.copy()
    
    if log == True:
        bins = np.logspace(begin, end, steps)
    else:
        bins=np.linspace(begin, end, steps)
        
    placeholder['bins_x'], bins = pd.cut(abs(placeholder[param_x]), bins=bins, include_lowest=True, retbins=True)
    placeholder['bins_y'], bins = pd.cut(abs(placeholder[param_y]), bins=bins, include_lowest=True, retbins=True)
        
    bin_center = (bins[:-1] + bins[1:]) /2
    bin_median = abs(placeholder.groupby('bins_x')[param_y].agg(np.nanmedian))#.nanmedian())
    bin_count_x = abs(placeholder.groupby('bins_x')[param_x].count())
    bin_count_y = abs(placeholder.groupby('bins_y')[param_y].count())
    bin_std = abs(placeholder.groupby('bins_x')[param_y].agg(np.nanstd)) #.nanstd())
    bin_quantile_a = abs(placeholder.groupby('bins_x')[param_y].agg(lambda x: np.nanpercentile(x, q = 2.5)))
    bin_quantile_b = abs(placeholder.groupby('bins_x')[param_y].agg(lambda x: np.nanpercentile(x, q = 16)))
    bin_quantile_c = abs(placeholder.groupby('bins_x')[param_y].agg(lambda x: np.nanpercentile(x, q = 84)))
    bin_quantile_d = abs(placeholder.groupby('bins_x')[param_y].agg(lambda x: np.nanpercentile(x, q = 97.5)))
    return bin_center, bin_median, bin_count_x, bin_count_y, bin_std, bin_quantile_a, bin_quantile_b, bin_quantile_c, bin_quantile_d


def plot_envelope(df_plot, hist_steps, title, x_axis_title, alpha = 1):
    
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,figsize=(10,8))
    ################# first figure ##################################
    
    ax1 = axes[0,0]
    im = ax1.scatter( df_plot.y_test, df_plot.y_pred, alpha = alpha, s = 1, c = 'k')
    #                  c = df_plot.y_ML, cmap = 'jet', norm=colors.LogNorm(vmin=10, vmax=1000))
    # cbar = fig.colorbar(im, ax = ax1, location='right', pad = -0.0)
    # cbar.set_label('ratio  validation / corrected', rotation=270, labelpad = 20.3)
    ax1.plot([1, 3000], [1, 3000], '--k')
    ax1.set_ylabel('|Obukhov length| estimate')
    ax1.set_title('Obukhov length prediction (Test)')
    
    #######################  second figure  #############################
    
    ax2 = axes[0,1]
    ax2_2 = ax2.twinx()
    bin_center, bin_median, bin_count_test, bin_count_pred, bin_std, bin_quantile_a, bin_quantile_b, bin_quantile_c, bin_quantile_d = envelope(df_plot, 'y_test', 'y_pred', \
                                                                                                                             -1, 4, steps =hist_steps, log = True)
    ax2.plot(bin_center, bin_median, 'k', label = r'Median$_{estimate}$')
    ax2.fill_between(bin_center, bin_quantile_b, bin_quantile_c, color = 'gray', alpha = 0.8, label = r'$\sigma$')
    ax2.fill_between(bin_center, bin_quantile_a, bin_quantile_d, color = 'gray', alpha = 0.6, label = r'2$\sigma$')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.plot([1, 3000], [1, 3000], '--k')
    ax2.set_ylim(0.5,10000)
    ax2.set_xlim(0.5,10000)
    ax2.legend()
    ax2.set_title('Obukhov length prediction (Test)')
    
    ax2_2.bar(bin_center[1:], bin_count_pred.values[1:], width= np.diff(bin_center), color = 'b')
    ax2_2.tick_params(axis='y', colors='b')
    ax2_2.set_xscale('log')
    ylim = int(bin_count_pred.max()*3)
    ax2_2.set_ylim(0,ylim)
    ax2_2.set_ylabel('occurence', color='b')
    
    
    ###################### third figure ###################
    
    ax3 = axes[1,0]
    im = ax3.scatter( df_plot.y_test, df_plot.y_ML, alpha = alpha, c = 'k', s =1)
    ax3.plot([1, 3000], [1, 3000], '--k')
    ax3.set_ylabel('|Obukhov length| estimate')
    ax3.set_xlabel(x_axis_title)
    ax3.set_title('Obukhov length corrected (Test)')
    
    ###################### fourth figure ###################
    ax4 = axes[1,1]
    ax4_2 = ax4.twinx()
    bin_center, bin_median, bin_count_test, bin_count_pred, bin_std, bin_quantile_a, bin_quantile_b, bin_quantile_c, bin_quantile_d = envelope(df_plot, 'y_test', 'y_ML', \
                                                                                                                             -1, 4, steps =hist_steps, log = True)
    ax4.plot(bin_center, bin_median, 'k', label = r'Median$_{corrected}$')
    ax4.fill_between(bin_center, bin_quantile_b, bin_quantile_c, color = 'gray', alpha = 0.8, label = r'$\sigma$')
    ax4.fill_between(bin_center, bin_quantile_a, bin_quantile_d, color = 'gray', alpha = 0.6, label = r'2$\sigma$')
    
    
    ax4.set_xlabel(x_axis_title)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.plot([1, 3000], [1, 3000], '--k')
    ax4.set_ylim(0.5,10000)
    ax4.set_xlim(0.5,10000)
    ax4.set_title('Obukhov length corrected (Test)')
    ax4.legend()
    
    ax4_2.bar(bin_center[1:], bin_count_pred.values[1:], width= np.diff(bin_center), color = 'b')
    ax4_2.tick_params(axis='y', colors='b')
    ax4_2.set_xscale('log')
    ax4_2.set_ylim(0,ylim )
    ax4_2.set_ylabel('Occurence', color='b')
    
    fig.tight_layout()
    fig.suptitle(title, fontsize = 15)
    fig.subplots_adjust(top=0.88)
    # plt.subplots_adjust(wspace=0.3)
    plt.show()

def plot_envelope_single(df_plot, param_test, param_predict, hist_steps, title, x_axis_title, y_axis_title, alpha = 1, legend = True, axis_scale = 'log', ax_min = 0.5, ax_max = 10000 ):

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,figsize=(7,6))
    
    fontsize = 15
    ax2 = axes
    ax2_2 = ax2.twinx()
    bin_center, bin_median, bin_count_test, bin_count_pred, bin_std, bin_quantile_a, bin_quantile_b, bin_quantile_c, bin_quantile_d = envelope(df_plot, param_test, param_predict, \
                                                                                                                             -1, 4, steps =hist_steps, log = True)
    ax2.plot(bin_center, bin_median, 'k', label = r'Median')
    ax2.fill_between(bin_center, bin_quantile_b, bin_quantile_c, color = 'gray', alpha = 0.8, label = r'$68\%%$')
    ax2.fill_between(bin_center, bin_quantile_a, bin_quantile_d, color = 'gray', alpha = 0.6, label = r'$95\%%$')
    
    ax2.set_xscale(axis_scale)
    ax2.set_yscale(axis_scale)
    ax2.plot([ax_min*2, ax_max/2], [ax_min*2, ax_max/2], '--k')
    ax2.set_ylim(ax_min,ax_max)
    ax2.set_xlim(ax_min,ax_max)
    # ax2.legend(fontsize = fontsize)
    # ax2.set_title('Obukhov length prediction (Test)', fontsize=fontsize)
    ax2.set_ylabel(y_axis_title, fontsize=fontsize)
    ax2.set_xlabel(x_axis_title, fontsize=fontsize)
    
    ax2.tick_params(which='major', width=3, length=6)
    ax2.tick_params(which='minor', width=1.5, length=3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    
    # ax2_2.bar(bin_center[1:], bin_count_pred.values[1:], width= np.diff(bin_center), hatch="X", edgecolor = 'k', color = 'gray', alpha = 0.5)
    ax2_2.step(bin_center[1:], bin_count_pred.values[1:], where = 'mid', color = 'r', alpha = 0.5, linewidth = 3)
    ax2_2.step(bin_center[1:], bin_count_test.values[1:], where = 'mid', color = 'b', alpha = 0.5, linewidth = 3)
    # ax2_2.bar(bin_center[1:], bin_count_test.values[1:], width= np.diff(bin_center), color = 'none', edgecolor = 'k')
    ax2_2.tick_params(axis='y', colors='b')
    # ax2_2.set_yticks([0, 0.05, 0.10, 0.15, 0.2])
    # ax2_2.set_xscale(axis_scale)
    # ylim = 0.20# int(bin_count_test.max()*4)
    # ax2_2.set_ylim(0,ylim)
    # ax2_2.set_ylabel('Relative freq.', color='b', fontsize=fontsize)
    ax2_2.set_xscale('log')
    ylim = int(bin_count_test.max()*4)
    ax2_2.set_ylim(0,ylim)
    ax2_2.set_ylabel('Hist. count', color='b', fontsize=fontsize)
    
    # -- plot custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    if legend == True:
        legend_elements = [Line2D([0], [0], color='k', lw=4, label = 'Median'),
                           Patch(facecolor='gray', alpha = 0.8, edgecolor='none', label = '68%'),
                           Patch(facecolor='gray', alpha = 0.6, edgecolor='none', label = '95%'),
                           # Patch(hatch="X", edgecolor = 'k', facecolor = 'none', label='Val.'),
                           Line2D([0], [0], color='r', alpha = 0.5, linewidth = 3, label = 'Est.'),
                           Line2D([0], [0], color='b', alpha = 0.5, linewidth = 3, label = 'Val.')]
        plt.legend(handles = legend_elements, framealpha =0.99, edgecolor = 'black', borderpad = 0.2, 
                   loc = 'upper left', ncol = 2, fontsize = fontsize)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    # plt.subplots_adjust(wspace=0.3)
    # plt.show()
    
    return bin_center, bin_median, bin_count_test, bin_count_pred, bin_std, fig
    
    

def world_maps_single(df_input, variables, statistics, norms, cmaps, rows, columns, title = None, cbar_title = None, shrink = 0.85,  resolution = 2, pad =0.1, fontsize = 15,
                      labelsize = 10, labelpad = 15, cbar_ticks = [-1, 0, 1], cbar_labels = ['-1', '0', '1'], extent = [-180, 180, -75, 75], 
                      xticks = [-180, -120, -60, 0, 60, 120, 180], yticks = [60, 30, 0, -30, -60],  figsize = None):
    
    """
    df_input: dataframe containing all variables selected and 'lon' and 'lat' parameters
    variables: list containing variable names as strings
    statistics: list containing the statistics as string for each variable (e.g. ['mean', 'median'])
    norms: list of colorbar norms per variable (e.g. [matplotlib.colors.Normalize(vmin=0, vmax=360), matplotlib.colors.LogNorm(vmin=-15, vmax=15)])
    cmaps: list of colormap names to be used (e.g. ['Reds_r', 'jet'])
    rows: number of rows in the figure
    colums: number of columns in the figure
    shrink: shrinkage factor of the colorbars
    resolution: resolution of gridded 'lon' and 'lat'
    """

    import matplotlib.colors
    import cartopy as cart
    import cartopy.crs as ccrs
    from scipy.stats import binned_statistic_2d
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    
    if figsize == None: figsize=(columns*10,rows*5)
    
    # -- create figure
    fig, axes= plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    
    
    # -- set original facecolour to white such that axes can be made gray without making the rest transparant
    fig.set_facecolor('white')
    
    # --------- add coastline to each plot ----------
    # -- for multiple rows and columns
    if len(np.shape(axes)) ==2:
        [(axes[x,y].add_feature(cart.feature.LAND,zorder=100, edgecolor='k'), 
          axes[x,y].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--'), 
          axes[x,y].set_facecolor('silver')
         ) for x in np.arange(0,np.shape(axes)[0],1) for y in np.arange(0,np.shape(axes)[1],1)]
        
    # -- for single row or column
    elif len(np.shape(axes)) ==1: 
        [(axes[x].add_feature(cart.feature.LAND,zorder=100, edgecolor='k'), 
          axes[x].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--'), 
          axes[x].set_facecolor('silver'),
         ) for x in np.arange(0,np.shape(axes)[0],1)]
        
    # -- for single cell
    elif ~(len(np.shape(axes)) >=1): 
          axes.add_feature(cart.feature.LAND,zorder=100, edgecolor='k') 
          gl = axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--') 
          axes.set_facecolor('silver')
          
          gl.top_labels = False
          gl.right_labels = False
          gl.xlabel_style = {'size': labelsize}
          gl.ylabel_style = {'size': labelsize}
          gl.ylocator = mticker.FixedLocator(yticks)
          gl.xlocator = mticker.FixedLocator(xticks)
          gl.xformatter = LONGITUDE_FORMATTER
          gl.yformatter = LATITUDE_FORMATTER
          axes.set_extent(extent, ccrs.PlateCarree())
          axes.tick_params(axis='both', which='major', labelsize=labelsize)
    # -----------------------------------------------
    
    # -- create grid for 2D histogram
    lat = np.arange(-90,90,resolution)
    lon = np.arange(-180,180,resolution)
    lons,lats = np.meshgrid(lon,lat)
    
    # -- create empty list to store plotted data
    datas = []

    #######################################
    ####### 2D binned histogram ###########
    #######################################
    for idx, ax in enumerate(np.ravel(axes)):
        if idx <= len(np.ravel(variables))-1:
            data, _, _, _ = binned_statistic_2d(df_input.lon,df_input.lat,values=df_input[variables[idx]], statistic= statistics[idx], bins=[lon, lat], expand_binnumbers=True)
            data = data.T; datas.append(data)
            im = ax.imshow(data, origin="lower", extent = [-180, 180,-90, 90], cmap = cmaps[idx], norm =norms[idx])
            if title == None:
                ax.set_title(r'$\mathbf{%s}(%s)$' %(str(statistics[idx]).replace('_', '\ '), variables[idx].replace('_', '\ ')), fontsize = fontsize)
            else: 
                ax.set_title(title, fontsize = fontsize)
            shrink = shrink
            cbar = fig.colorbar(im, ax=ax, shrink=shrink, pad=pad, ticks = cbar_ticks) #cbar3.set_ticks([-1, -0.5, 0,  0.5, 1])
            if cbar_title == None:
                'ok, no title for you then'
            else:
                cbar.set_label(cbar_title, rotation=270, labelpad=labelpad, fontsize = fontsize)
                cbar.ax.set_yticklabels(cbar_labels)
                cbar.ax.tick_params(labelsize=labelsize) 
                
    fig.tight_layout()
    
    return fig, datas
    
    
