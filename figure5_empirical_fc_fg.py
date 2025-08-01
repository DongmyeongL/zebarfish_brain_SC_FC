import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
from random import uniform
from colorsys import hsv_to_rgb
from scipy.stats import linregress
from scipy.stats import lognorm, gaussian_kde, norm, kurtosis, skew, expon
import seaborn as sns
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import powerlaw 
from scipy.stats import lognorm, gamma
from scipy.optimize import curve_fit
import generate_sc_optimization as gso
from brainspace.gradient import GradientMaps


def analyze_distributions_with_plots(data, title ,xlim,figrue_axis):
    """
    주어진 데이터의 Probability Density Function (PDF)을 log-log 스케일로 그리고,
    log-normal, exponential, stretched exponential, gamma 분포를 피팅하여 시각화합니다.
    CDF 플롯은 포함되지 않습니다.

    Parameters:
    - data (array-like): 입력 데이터 (양수만 포함)
    - title (str): 플롯의 제목
    """
    data = np.array(data)
    data = data[data > 0]  # Remove non-positive values
    data.sort()  # Sort data (still good practice, even if not directly for CDF plot)

    # --- Prepare for plotting ---
    # Create a smooth range of x-values for plotting fitted PDFs
    # Ensure x_vals cover the range of data for better visualization
    # x_min_plot needs to be strictly positive for log-log plot
    x_min_plot = np.min(data) * 0.9 if np.min(data) > 0 else 0.01
    x_max_plot = np.max(data) * 1.1
    x_vals = np.linspace(x_min_plot, x_max_plot, 1000)
    # Ensure x_vals are positive for log scales
    x_vals = x_vals[x_vals > 0]

    # --- Initial Fits (needed for PDF parameters) ---
    # These fits are based on the full data, or potentially for CCDF first
    # However, for simplicity and consistency with previous requests,
    # we'll perform fits here, assuming they are robust enough for PDF visualization.

    # Log-Normal Distribution Fit
    try:
        shape_ln, loc_ln, scale_ln = lognorm.fit(data, floc=0)
    except Exception as e:
        print(f"Log-Normal fit failed: {e}")
        shape_ln, loc_ln, scale_ln = None, None, None # Set to None if fit fails

    # Exponential Distribution Fit
    scale_exp = np.mean(data)

    # Stretched Exponential Fit (requires curve_fit on CCDF or specialized PDF fit)
    # Since stretched exponential PDF fit is more complex and less direct than CDF fit,
    # we'll fit its CCDF to get parameters, then use those for PDF.
    def stretched_exp_sf(x, tau, beta):
        x_processed = np.array(x, dtype=float)
        x_processed[x_processed <= 0] = np.finfo(float).eps
        return np.exp(-(x_processed / tau)**beta)

    initial_guess_se = [np.mean(data), 0.5]
    tau_se, beta_se = None, None
    try:
        # Empirical CCDF for fitting stretched exponential
        ccdf_empirical_for_se_fit = 1 - (np.arange(1, len(data) + 1) / len(data))
        x_for_fit_se = data[ccdf_empirical_for_se_fit > 0]
        y_for_fit_se = ccdf_empirical_for_se_fit[ccdf_empirical_for_se_fit > 0]
        valid_indices_se = (x_for_fit_se > 0) & (y_for_fit_se > 0)
        x_for_fit_se = x_for_fit_se[valid_indices_se]
        y_for_fit_se = y_for_fit_se[valid_indices_se]

        params_se, covariance_se = curve_fit(stretched_exp_sf, x_for_fit_se, y_for_fit_se, p0=initial_guess_se,
                                             bounds=([np.finfo(float).eps, 0.01], [np.inf, 1.0]),
                                             maxfev=5000)
        tau_se, beta_se = params_se
    except Exception as e:
        print(f"Stretched Exponential fit (for PDF params) failed: {e}")

    # Gamma Distribution Fit
    try:
        a_gamma, loc_gamma, scale_gamma = gamma.fit(data, floc=0)
    except Exception as e:
        print(f"Gamma fit failed: {e}")
        a_gamma, loc_gamma, scale_gamma = None, None, None # Set to None if fit fails


    # --- Figure: PDF Plot (Log-Log Scale) ---
    #plt.figure(figsize=(10, 7))

    # Empirical PDF using KDE (Kernel Density Estimation)
    # seaborn's histplot with kde=True can generate a KDE plot,
    # but for log-log plotting, we often need the KDE density values directly.
    # We'll compute KDE and plot it with loglog.
    #kde = sns.kdeplot(data, log_scale=False, cut=0) # Get KDE values without plotting directly
    #if kde is not None and len(kde.get_lines()) > 0: # Check if KDE lines exist
    #    kde_x, kde_y = kde.get_lines()[0].get_data()
    #    # Filter for positive values for log-log plot
    #    valid_kde_indices = (kde_x > 0) & (kde_y > 0)
    #    plt.loglog(kde_x[valid_kde_indices], kde_y[valid_kde_indices], color='skyblue', linewidth=2, label='Empirical PDF (KDE)')
    #else: # Fallback for very small datasets where KDE might not produce lines
    print("KDE plot generation might have failed or produced no lines. Plotting histogram fallback.")
    hist_counts, hist_bins = np.histogram(data, bins='auto', density=True)
    hist_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    valid_hist_indices = (hist_centers > 0) & (hist_counts > 0)
    figrue_axis.loglog(hist_centers[valid_hist_indices], hist_counts[valid_hist_indices], 'o', markersize=1, color='black', label='Empirical PDF (Histogram)')


    # --- Plotting Fitted PDFs on Log-Log Scale ---

    # Log-Normal Distribution
    x_vals=hist_centers[valid_hist_indices];
    exp_y_vals=hist_counts[valid_hist_indices];
    
    if shape_ln is not None:
        pdf_fitted_ln = lognorm.pdf(x_vals, shape_ln, loc=loc_ln, scale=scale_ln)
        
        log_norm_error=0;
        for k in range(len(exp_y_vals)):
            log_norm_error+=(np.log10(exp_y_vals[k])- np.log10(pdf_fitted_ln[k]))**2;
            
        
        valid_plot_indices_ln = (x_vals > 0) & (pdf_fitted_ln > 0)
        figrue_axis.loglog(x_vals[valid_plot_indices_ln], pdf_fitted_ln[valid_plot_indices_ln], '-', linewidth=1, label='Log-Normal Fit')



    # Exponential Distribution
    pdf_fitted_exp = expon.pdf(x_vals, loc=0, scale=scale_exp)
    valid_plot_indices_exp = (x_vals > 0) & (pdf_fitted_exp > 0)
    figrue_axis.loglog(x_vals[valid_plot_indices_exp], pdf_fitted_exp[valid_plot_indices_exp], '-', linewidth=1, label='Exponential Fit')

    exp_error=0;
    for k in range(len(exp_y_vals)):
            exp_error+=(np.log10(exp_y_vals[k])-np.log10(pdf_fitted_exp[k]))**2;
    
    ll=len(exp_y_vals)
    print(f'{title}_log_norm_{(log_norm_error)/ll}_exp_error_{(exp_error)/ll}');
    
    
    figrue_axis.tick_params(axis='x', pad=1, length=1)
    figrue_axis.tick_params(axis='y', pad=1, length=1)
    figrue_axis.set_xlim(xlim);
    '''
    # Stretched Exponential Distribution PDF
    def stretched_exp_pdf(x, tau, beta):
        x_processed = np.array(x, dtype=float)
        x_processed[x_processed <= 0] = np.finfo(float).eps
        # Ensure the exponent result is not negative for fractional powers
        val = (x_processed / tau)
        # Handle case where val could be 0 and beta-1 is negative
        if np.any(val == 0) and (beta - 1 < 0):
             val[val == 0] = np.finfo(float).eps
        
        return (beta / tau) * (val**(beta - 1)) * np.exp(-(val)**beta)

    if tau_se is not None and beta_se is not None:
        pdf_fitted_se = stretched_exp_pdf(x_vals, tau_se, beta_se)
        valid_plot_indices_se = (x_vals > 0) & (pdf_fitted_se > 0)
        #plt.loglog(x_vals[valid_plot_indices_se], pdf_fitted_se[valid_plot_indices_se], 'c-.', linewidth=2, label='Stretched Exp Fit')

    # Gamma Distribution
    if a_gamma is not None:
        pdf_fitted_gamma = gamma.pdf(x_vals, a_gamma, loc=loc_gamma, scale=scale_gamma)
        valid_plot_indices_gamma = (x_vals > 0) & (pdf_fitted_gamma > 0)
      #  plt.loglog(x_vals[valid_plot_indices_gamma], pdf_fitted_gamma[valid_plot_indices_gamma], 'm:', linewidth=2, label='Gamma Fit')
    '''
    #figrue_axis.title(f'{title}')
    #plt.xlabel("Value (log scale)")
    #plt.ylabel("Density (log scale)")
    #figrue_axis.legend()
    #figrue_axis.grid(True, which="both", ls="-", color='0.9')
    #figrue_axis.tight_layout()
    #plt.show()

    
    return [ (log_norm_error)/ll ,(exp_error)/ll   ];


mag=1.0;
plt.rcParams['font.size'] = 6*mag
plt.rcParams['font.family'] = 'Arial'



data = np.load('../raw_data/subject12_cellular_data.npz', allow_pickle=True)
sel_neuron = data['sel_neuron']
sc_mat = data['sc_mat']
t_CellXYZ_data = data['t_CellXYZ_data']
t_spot_data = data['t_spot_data']

selected_xyz = t_CellXYZ_data[sel_neuron]
spot_data = t_spot_data[sel_neuron]
#fc_mat = np.corrcoef(spot_data)


with open("../raw_data/subject12_cellular_data_sc_mat_indices.pkl", "rb") as f:
    indices_ii = pickle.load(f)

    
regionLabels = ['MON', 'Cb', 'MOS1', 'MOS2', 'MOS3', 'MOS4', 'MOS5', 'IPN', 'IO', 'Hc', 'Ra', 'T', 'aRF', 'imRF', 'pRF', 'GG', 'Hb', 'Hi', 'Hr', 'OG', 'OB', 'OE', 'P', 'Pi', 'PT', 'PO', 'PrT', 'R', 'SP', 'TeO', 'Th', 'TL', 'TS', 'TG', 'VR', 'NX','rMON', 'rCb', 'rMOS1', 'rMOS2', 'rMOS3', 'rMOS4', 'rMOS5', 'rIPN', 'rIO', 'rHc', 'rRa', 'rT', 'raRF', 'rimRF', 'rpRF', 'rGG', 'rHb', 'rHi', 'rHr', 'rOG', 'rOB', 'rOE', 'rP', 'rPi', 'rPT', 'rPO', 'rPrT', 'rR', 'rSP', 'rTeO', 'rTh', 'rTL', 'rTS', 'rTG', 'rVR', 'rNX'];

sort_id=np.load('../raw_data/subject_12_sort_id.npy');


plt.figure();
mean_data=[];
for i in range(len(indices_ii)):
    
    index=indices_ii[i];
    spot_data_i = spot_data[index];
    plt.plot(np.mean(spot_data_i, axis=0)-i);
    mean_data.append(np.mean(spot_data_i, axis=0));

np.save('../raw_data/subject12_cellular_data_mean_spot_data.npy', np.array(mean_data));

emp_data = np.array(mean_data);
fc_vectors,emp_fc_mat,emp_whole_fc_mat=gso.fun_fc_vector((emp_data), 15,5);


plt.figure();
plt.imshow(np.abs(emp_whole_fc_mat), cmap='jet', vmin=0, vmax=0.8);



gm = GradientMaps(n_components=2, approach='dm', kernel='cosine')
gm.fit(emp_whole_fc_mat);
gradients = gm.gradients_   
gradients1=gso.normalize_list(gradients[:,0].tolist());
gradients1=np.array(gradients1);
gmin=np.min(gradients1);
gmax=np.max(gradients1);
gradients1=(gradients1-gmin)/(gmax-gmin);
plt.figure();

mag=0.8;
plt.rcParams['font.size'] = 6;
plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(5.0*mag, 7*mag), dpi=300)


for i in range(len(gradients1)):
    index=indices_ii[i];
    ts= selected_xyz[index];
    ts=np.array(ts);
   
    c=np.ones((ts.shape[0], 1)) * gradients1[i];  # Default color
    tsx=ts[:,1];#[::-1];
    tsy=(ts[:,0]);#[::-1];
  
    plt.scatter(tsx, tsy,c=c,cmap='jet',alpha=0.3, s=0.5,vmin=0, vmax=1);
    #ymin, ymax = plt.ylim()

plt.axis('off')
plt.ylim(1000, 100)
plt.colorbar();
   
fig.savefig('./fig5_mean_emp_fg.png',format='png',dpi=300,transparent=True);



for qsel in [10, 15, 45]:
    fig = plt.figure(figsize=(5.0*mag, 7*mag), dpi=300)
    gm = GradientMaps(n_components=2, approach='dm', kernel='cosine')
    gm.fit(emp_fc_mat[qsel]);
    gradients = gm.gradients_   
    gradients1=gso.normalize_list(gradients[:,0].tolist());
    gradients1=np.array(gradients1);
    gmin=np.min(gradients1);
    gmax=np.max(gradients1);
    gradients1=(gradients1-gmin)/(gmax-gmin);
    for i in range(len(gradients1)):
        index=indices_ii[i];
        ts= selected_xyz[index];
        ts=np.array(ts);
        c=np.ones((ts.shape[0], 1)) * gradients1[i]; 
        plt.scatter(ts[:,1], ts[:,0],c=c,cmap='jet',alpha=0.3, s=0.5,vmin=0, vmax=1);
    plt.axis('off')
    plt.ylim(1000, 100)
    plt.colorbar();
    fig.savefig(f'./fig5_{qsel}_emp_fg.png',format='png',dpi=300,transparent=True);

    
    
plt.show();
qq=[];      




aaaa;






N = selected_xyz.shape[0]
M = 5000
if M > N:
    raise ValueError("선택할 수 있는 뉴런 수보다 M이 큽니다.")

pick_indices = np.random.choice(N, M, replace=False)


sub_fc = fc_mat[np.ix_(pick_indices, pick_indices)]
sub_sc = sc_mat[np.ix_(pick_indices, pick_indices)]
sub_xyz = selected_xyz[pick_indices] 

dist_matrix = squareform(pdist(sub_xyz, metric='euclidean'))

triu_mask = np.triu(np.ones_like(sub_fc, dtype=bool), k=1)

valid_mask = (sub_sc > 0) & triu_mask

fc_data = np.abs(sub_fc[valid_mask])
dist_data = dist_matrix[valid_mask]
fc_data=np.arctanh(fc_data)

axes[3].hexbin(dist_data, (fc_data), gridsize=80, cmap='jet', bins='log')


# 1. 거리 bin 설정
num_bins = 40
bins = np.linspace(np.min(dist_data), np.max(dist_data), num_bins + 1)
bin_indices = np.digitize(dist_data, bins)

# 2. 거리 bin별 통계 계산
stats = {
    'bin_center': [],
    'fc_mean': [],
    'fc_std': [],
    'fc_skew': [],
    'fc_kurtosis': [],
    'count': []
}

x=[];
y=[];
for b in range(1, num_bins + 1):

    fc_in_bin = fc_data[bin_indices == b]

    shape_ln, loc_ln, scale_ln = lognorm.fit(fc_in_bin, floc=0)
    
    
    print(b, (np.mean(np.log(fc_in_bin))) ,(np.std(np.log(fc_in_bin))))
    
    x.append([(bins[b-1] + bins[b]) / 2, np.mean(np.log(fc_in_bin))]);
    y.append([(bins[b-1] + bins[b]) / 2, np.std(np.log(fc_in_bin)) ]);
    
    
    if len(fc_in_bin) > 0:
        stats['bin_center'].append((bins[b-1] + bins[b]) / 2)
        stats['fc_mean'].append(np.mean(fc_in_bin))
        stats['fc_std'].append(np.std(fc_in_bin))
        stats['fc_skew'].append(skew(fc_in_bin))
        stats['fc_kurtosis'].append(kurtosis(fc_in_bin))
        stats['count'].append(len(fc_in_bin))
    else:
        # 빈 bin을 위해 NaN 추가
        stats['bin_center'].append((bins[b-1] + bins[b]) / 2)
        stats['fc_mean'].append(np.nan)
        stats['fc_std'].append(np.nan)
        stats['fc_skew'].append(np.nan)
        stats['fc_kurtosis'].append(np.nan)
        stats['count'].append(0)

plt.figure();
x=np.array(x);
y=np.array(y);

def exp_model(x, a,b,c):
    return a * np.exp(b * x)+c

from scipy.optimize import curve_fit

popt, pcov = curve_fit(exp_model, x[0:30,0], x[0:30,1], p0=(-2, -0.01,0));

a_est, b_est,c_est = popt
print(f"Estimated parameters: a = {a_est:.4f}, b = {b_est:.4f}, c = {c_est:.4f} ")

plt.plot(x[:,0],x[:,1]);
plt.plot(y[:,0],y[:,1]);


plt.show();


# 3. DataFrame으로 변환
import pandas as pd

df_stats = pd.DataFrame(stats)


#plt.figure(figsize=(10, 4))
axes[0].plot(df_stats['bin_center'], df_stats['fc_mean'], label='Mean FC', marker='o')
axes[0].fill_between(df_stats['bin_center'],
                 df_stats['fc_mean'] - df_stats['fc_std'],
                 df_stats['fc_mean'] + df_stats['fc_std'],
                 alpha=0.3, label='±1 Std')
#plt.xlabel('Distance')
#plt.ylabel('FC')
#plt.title('Distance vs FC (Mean ± Std)')
#axes[0].set_legend()
#plt.grid(True)
#plt.tight_layout()



#plt.plot(df_stats['bin_center'], df_stats['fc_skew'], marker='s', label='Skewness')
axes[4].plot(df_stats['bin_center'], df_stats['fc_kurtosis'], marker='^', label='Kurtosis')
#plt.xlabel('Distance')
#plt.ylabel('Value')
#plt.title('Skewness and Kurtosis of FC per Distance Bin')
#plt.legend()
#plt.grid(True)
#plt.tight_layout()



b=1;
fc_in_bin = fc_data[bin_indices == b];
analyze_distributions_with_plots(fc_in_bin, '' ,[0.01,2],axes[1]);

b=15;
fc_in_bin = fc_data[bin_indices == b];
analyze_distributions_with_plots(fc_in_bin, '' ,[0.01,2],axes[2]);

'''
b=25;
fc_in_bin = fc_data[bin_indices == b];
fig = plt.figure();
figrue_axis=fig.add_axes([0.1,0.3,0.5,0.5]);
analyze_distributions_with_plots(fc_in_bin, '' ,[0.01,2],figrue_axis);
'''


for i in range(5):
    axes[i].tick_params(axis='x', pad=1, length=1)
    axes[i].tick_params(axis='y', pad=1, length=1)
    
fig.savefig('./fig4.png',format='png',dpi=300,transparent=True);
    
plt.show()