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
    #print(np.min(data));
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
    #print("KDE plot generation might have failed or produced no lines. Plotting histogram fallback.")
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
            #log_norm_error+=((exp_y_vals[k])-(pdf_fitted_ln[k]))**2;
            
        
        valid_plot_indices_ln = (x_vals > 0) & (pdf_fitted_ln > 0)
        figrue_axis.loglog(x_vals[valid_plot_indices_ln], pdf_fitted_ln[valid_plot_indices_ln], '-', linewidth=1, label='Log-Normal Fit')



    # Exponential Distribution
    pdf_fitted_exp = expon.pdf(x_vals, loc=0, scale=scale_exp)
    valid_plot_indices_exp = (x_vals > 0) & (pdf_fitted_exp > 0)
    figrue_axis.loglog(x_vals[valid_plot_indices_exp], pdf_fitted_exp[valid_plot_indices_exp], '-', linewidth=1, label='Exponential Fit')

    exp_error=0;
    for k in range(len(exp_y_vals)):
            exp_error+=(np.log10(exp_y_vals[k])-np.log10(pdf_fitted_exp[k]))**2;
            #exp_error+=((exp_y_vals[k])-(pdf_fitted_exp[k]))**2;
    
    ll=len(exp_y_vals)
    #print(f'{title}_log_norm_{(log_norm_error)/ll}_exp_error_{(exp_error)/ll}');
    
    
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

# 🔍 1. 로그-노말 분포 분석 함수
def analyze_distributions_on_ccdf(data, title=''):
    """
    주어진 데이터의 Complementary Cumulative Distribution Function (1-CDF)을 그리고,
    log-normal, exponential, stretched exponential, gamma 분포를 피팅하여 시각화합니다.

    Parameters:
    - data (array-like): 입력 데이터 (양수만 포함)
    - title (str): 플롯의 제목
    """
    data = np.array(data)
    data = data[data > 0]  # 0 이하 값 제거
    data.sort()  # CDF 계산을 위해 데이터 정렬

    # 경험적 CCDF (1-ECDF) 계산
    # CCDF는 P(X > x) 이므로, 1 - ECDF
    ccdf_empirical = 1 - (np.arange(1, len(data) + 1) / len(data))

    plt.figure(figsize=(10, 7))

    # Log-log 스케일로 ECDF 플롯
    plt.loglog(data, ccdf_empirical, 'o', markersize=4, color='darkblue', label='Empirical CCDF')

    # --- Log-Normal 분포 피팅 및 플롯 ---
    try:
        shape_ln, loc_ln, scale_ln = lognorm.fit(data, floc=0)
        x_ln = np.linspace(min(data), max(data), 500)
        sf_fitted_ln = lognorm.sf(x_ln, shape_ln, loc=loc_ln, scale=scale_ln)
        plt.loglog(x_ln, sf_fitted_ln, 'r--', linewidth=2, label=f'Log-Normal Fit (s={shape_ln:.2f}, scale={scale_ln:.2f})')
    except Exception as e:
        print(f"Log-Normal fit failed: {e}")

    # --- Exponential 분포 피팅 및 플롯 ---
    scale_exp = np.mean(data) # Exponential distribution scale is the mean
    x_exp = np.linspace(min(data), max(data), 500)
    sf_fitted_exp = np.exp(-x_exp / scale_exp) # Survival function for exponential
    plt.loglog(x_exp, sf_fitted_exp, 'g:', linewidth=2, label=f'Exponential Fit (scale={scale_exp:.2f})')

    # --- Stretched Exponential 분포 피팅 및 플롯 ---
    # Stretched Exponential Survival Function: S(x) = exp(-(x/tau)^beta)
    def stretched_exp_sf(x, tau, beta):
        return np.exp(-(x / tau)**beta)

    # Initial guess for parameters (tau, beta)
    # tau could be around the mean, beta is usually between 0 and 1
    initial_guess_se = [np.mean(data), 0.5] 
    
    # Filter data for curve_fit to ensure positive values
    positive_data_indices = data > 0
    x_for_fit = data[positive_data_indices]
    y_for_fit = ccdf_empirical[positive_data_indices]

    # Ensure y_for_fit is also positive for log-log fit stability
    positive_y_indices = y_for_fit > 0
    x_for_fit = x_for_fit[positive_y_indices]
    y_for_fit = y_for_fit[positive_y_indices]
    
    try:
        # curve_fit requires valid x and y data for fitting
        params_se, covariance_se = curve_fit(stretched_exp_sf, x_for_fit, y_for_fit, p0=initial_guess_se,
                                             bounds=([0.01, 0.01], [np.inf, 1.0]), # tau > 0, 0 < beta <= 1
                                             maxfev=5000) # Increase max iterations if needed
        tau_se, beta_se = params_se
        x_se = np.linspace(min(data), max(data), 500)
        sf_fitted_se = stretched_exp_sf(x_se, tau_se, beta_se)
        plt.loglog(x_se, sf_fitted_se, 'c-.', linewidth=2, label=f'Stretched Exp Fit (tau={tau_se:.2f}, beta={beta_se:.2f})')
    except Exception as e:
        print(f"Stretched Exponential fit failed: {e}")

    # --- Gamma 분포 피팅 및 플롯 ---
    try:
        a_gamma, loc_gamma, scale_gamma = gamma.fit(data, floc=0)
        x_gamma = np.linspace(min(data), max(data), 500)
        sf_fitted_gamma = gamma.sf(x_gamma, a_gamma, loc=loc_gamma, scale=scale_gamma)
        plt.loglog(x_gamma, sf_fitted_gamma, 'm:', linewidth=2, label=f'Gamma Fit (a={a_gamma:.2f}, scale={scale_gamma:.2f})')
    except Exception as e:
        print(f"Gamma fit failed: {e}")

    # --- Powerlaw.Fit for Distribution Comparison (Informative Output) ---
    # powerlaw library is primarily for power-law and related distributions.
    # We use its distribution_compare for quantitative comparison of various fits.
    print(f"\n--- Distribution Comparison for '{title}' (Powerlaw.Fit) ---")
    try:
        fit = powerlaw.Fit(data, xmin=np.min(data), verbose=False)
        
        # Compare lognormal vs. exponential
        R_ln_exp, p_ln_exp = fit.distribution_compare('lognormal', 'exponential')
        print(f"Lognormal vs. Exponential (R, p): ({R_ln_exp:.4f}, {p_ln_exp:.4f})")
        
        # Compare lognormal vs. stretched exponential (Weibull is often used here)
        # Stretched exponential is also known as complementary cumulative Weibull.
        # powerlaw library's 'weibull' refers to the stretched exponential.
        R_ln_se, p_ln_se = fit.distribution_compare('lognormal', 'weibull')
        print(f"Lognormal vs. Stretched Exp (R, p): ({R_ln_se:.4f}, {p_ln_se:.4f})")

        # Compare lognormal vs. gamma
        R_ln_gamma, p_ln_gamma = fit.distribution_compare('lognormal', 'gamma')
        print(f"Lognormal vs. Gamma (R, p): ({R_ln_gamma:.4f}, {p_ln_gamma:.4f})")
        
        # Add more comparisons as desired (e.g., exponential vs. stretched exp)
        R_exp_se, p_exp_se = fit.distribution_compare('exponential', 'weibull')
        print(f"Exponential vs. Stretched Exp (R, p): ({R_exp_se:.4f}, {p_exp_se:.4f})")

        R_gamma_se, p_gamma_se = fit.distribution_compare('gamma', 'weibull')
        print(f"Gamma vs. Stretched Exp (R, p): ({R_gamma_se:.4f}, {p_gamma_se:.4f})")


        print(f"(Positive R favors the first distribution in the comparison)")
    except Exception as e:
        print(f"Could not perform powerlaw distribution comparison for '{title}': {e}")
    print(f"----------------------------------------------------------\n")


    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("1 - CDF (Survival Function)")
    plt.legend()
    plt.grid(True, which="both", ls="-", color='0.9')
    plt.tight_layout()
    plt.show()

    return
def generate_n_colors(nColors, saturation=65, value=90, randomness=0):
    """
    Generates a list of distinct RGB colors.

    Args:
        nColors (int): The number of colors to generate.
        saturation (int): Base saturation percentage (0-100).
        value (int): Base value percentage (0-100).
        randomness (float): Amount of randomness to add to saturation.

    Returns:
        list: A list of RGB color tuples.
    """
    h = np.linspace(0, 320, nColors)
    s = np.array([saturation + uniform(-randomness, randomness)] * nColors)
    v = np.array([value] * nColors)
    palette = []
    for i in range(nColors):
        palette.append(hsv_to_rgb(h[i] / 360, s[i] / 100, v[i] / 100))
    return palette


mag=1.0;
plt.rcParams['font.size'] = 6*mag
plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(7.0*mag, 6.0*mag), dpi=300)

dd=0.75/4;
hh=0.85/3-0.01
hh1=0.85/3-0.05
margin=0.05;
axes=[];
axes.append(fig.add_axes([margin,margin,(dd+margin)*1.81,hh1]));
#axes.append(fig.add_axes([margin+(margin+dd)*1,margin,dd,hh1]));

axes.append(fig.add_axes([margin+(margin+dd)*2,margin,(dd+margin)*1.81,hh1]));
#axes.append(fig.add_axes([margin+(margin+dd)*3,margin,dd,hh1]));

axes.append(fig.add_axes([margin,margin+hh+0.05,dd,hh1]));
axes.append(fig.add_axes([margin+(margin+dd)*1,margin+hh+0.05,dd,hh1]));
axes.append(fig.add_axes([margin+(margin+dd)*2,margin+hh+0.05,dd,hh1]));
axes.append(fig.add_axes([margin+(margin+dd)*3,margin+hh+0.05,dd,hh1]));

axes.append(fig.add_axes([margin,margin+2*hh+0.1,(dd*2+margin)*1.55,hh1]));
axes.append(fig.add_axes([margin+(margin+dd)*3,margin+2*hh+0.1,dd,hh1]));
#axes.append(fig.add_axes([margin+(margin+dd)*3,margin+2*hh+0.1,dd,hh1]));


check_region=np.zeros([7,72]);

jj=0;
for file_idx in [12,13,14,15,16,17,18]:

    file_name=f'../raw_data/subject_{file_idx}_data_synapse_sc_aux_data.pkl';
    with open(file_name, 'rb') as f:
            load_data = pickle.load(f)

    synapse_id=load_data['synapse_id'];
    
    for ii in synapse_id:
        check_region[jj,ii]=1;
    jj+=1;    

sum_check_region=np.sum(check_region,axis=0);
check_region_num=[];
for i in range(72):
        if(sum_check_region[i]==7):
            check_region_num.append(i);

subject_sum_degree_data=[];
for i in range(72):
    subject_sum_degree_data.append([]);


file_idx=12;
file_name=f'../raw_data/subject_{file_idx}_data_synapse_sc_aux_data.pkl';
with open(file_name, 'rb') as f:
    load_data = pickle.load(f)

synapse_id=load_data['synapse_id'];

new_set_synapse_id=[];

for tk in (synapse_id):
    if tk in check_region_num:
        new_set_synapse_id.append(tk); 

'''
for file_idx in [12,13,14,15,16,17,18]:

    sc_mat=np.load(f'../raw_data/subject_{file_idx}_data_synapse_sc_mat.py.npy', allow_pickle=True)
    file_name=f'../raw_data/subject_{file_idx}_data_synapse_sc_aux_data.pkl';
    with open(file_name, 'rb') as f:
            load_data = pickle.load(f)

    synapse_id=load_data['synapse_id'];
    indices_ii=load_data['indices_ii'];
    
    new_synapse_id=[];
    
    l=0;
    total_x_data=[];
    total_y_data=[];
    tx_data=[];
    ty_data=[];
    
    for tk in range(len(synapse_id)):
        
        k=synapse_id[tk];
        
        if k in check_region_num:
            pick_num=indices_ii[l];
            sub_sc = sc_mat[np.ix_(pick_num,pick_num)];
            
            sum_degree=np.sum(sub_sc,axis=0);
            
            x_data=[];
            y_data=np.sort(sum_degree);
            for ii in range(len(sum_degree)):
                x_data.append(ii);
            
            subject_sum_degree_data[k].extend(y_data);
            total_x_data.append(x_data/np.max(x_data));
            total_y_data.append(y_data);
            print(k,np.mean(y_data/np.max(y_data)),kurtosis(y_data/np.max(y_data)));
            tx_data.append(np.mean(y_data/np.max(y_data)));
            ty_data.append(kurtosis(y_data/np.max(y_data)));
            #plt.plot(x_data/np.max(x_data),y_data/np.max(y_data));
        
        l=l+1;

import pickle
with open('../raw_data/total_subject_intra_degree_datapk.pkl', 'wb') as f:
        pickle.dump(subject_sum_degree_data, f)
'''
import pickle
with open('../raw_data/total_subject_intra_degree_datapk.pkl', 'rb') as f:
            load_data = pickle.load(f)        

subject_sum_degree_data=load_data;


data=[];
regionLabels = ['MON', 'Cb', 'MOS1', 'MOS2', 'MOS3', 'MOS4', 'MOS5', 'IPN', 'IO', 'Hc', 'Ra', 'T', 'aRF', 'imRF', 'pRF', 'GG', 'Hb', 'Hi', 'Hr', 'OG', 'OB', 'OE', 'P', 'Pi', 'PT', 'PO', 'PrT', 'R', 'SP', 'TeO', 'Th', 'TL', 'TS', 'TG', 'VR', 'NX','rMON', 'rCb', 'rMOS1', 'rMOS2', 'rMOS3', 'rMOS4', 'rMOS5', 'rIPN', 'rIO', 'rHc', 'rRa', 'rT', 'raRF', 'rimRF', 'rpRF', 'rGG', 'rHb', 'rHi', 'rHr', 'rOG', 'rOB', 'rOE', 'rP', 'rPi', 'rPT', 'rPO', 'rPrT', 'rR', 'rSP', 'rTeO', 'rTh', 'rTL', 'rTS', 'rTG', 'rVR', 'rNX'];
error_data=[];
k=0;
mean_degree=[];
for ii in new_set_synapse_id[0:25]:
    
      
    data.append(kurtosis(subject_sum_degree_data[ii]));
    
    qqq=plt.figure()
    
    [log_error,exp_error]=analyze_distributions_with_plots(subject_sum_degree_data[ii],regionLabels[ii],[1,1000], qqq.add_axes([0.2,0.2,0.6,0.6]));
    print(k,regionLabels[ii],log_error,kurtosis(subject_sum_degree_data[ii]) );
    k=k+1;
    error_data.append([log_error,exp_error])
    mean_degree.append(np.mean(subject_sum_degree_data[ii]));
    plt.close();

data = np.load('../raw_data/subject12_cellular_data.npz', allow_pickle=True)
sel_neuron = data['sel_neuron']
sc_mat = data['sc_mat']
t_CellXYZ_data = data['t_CellXYZ_data']
t_spot_data = data['t_spot_data']

with open("../raw_data/subject12_cellular_data_sc_mat_indices.pkl", "rb") as f:
    indices_ii = pickle.load(f)






palette=generate_n_colors(28);
bar_positions = np.arange(len(sc_mat));

degree_data=np.sum(sc_mat,axis=1);
analyze_distributions_with_plots(degree_data,'degree_data',[10,10000],axes[7]);

ii=new_set_synapse_id[3]#hb
[log_error,exp_error]=analyze_distributions_with_plots(subject_sum_degree_data[ii],regionLabels[ii],[1,1000], axes[2]);
    
ii=new_set_synapse_id[4]#TH
[log_error,exp_error]=analyze_distributions_with_plots(subject_sum_degree_data[ii],regionLabels[ii],[1,1000], axes[3]);

ii=new_set_synapse_id[15]#MOS3
[log_error,exp_error]=analyze_distributions_with_plots(subject_sum_degree_data[ii],regionLabels[ii],[1,1000], axes[4]);

ii=new_set_synapse_id[18]#MON
[log_error,exp_error]=analyze_distributions_with_plots(subject_sum_degree_data[ii],regionLabels[ii],[1,1000], axes[5]);


axes[0].plot(error_data,'s-')    
axes[1].plot(mean_degree,'s-')    








#analyze_distributions_on_ccdf(degree_data,'degree_data');
#analyze_lognormal_distribution(degree_data,500,'degree_data');

calculated_skewness = skew(degree_data);
log_data=np.log10(degree_data[degree_data>0]);
mu, std = np.mean(log_data), np.std(log_data)
stat_ks, p_ks = stats.shapiro(log_data);

excess_kurtosis = kurtosis(degree_data, fisher=True);
print('degree_data excess_kurtosis', excess_kurtosis,calculated_skewness,p_ks)

selected_xyz = t_CellXYZ_data[sel_neuron]

N = selected_xyz.shape[0]
M = 5000
if M > N:
    raise ValueError("선택할 수 있는 뉴런 수보다 M이 큽니다.")

pick_indices = np.random.choice(N, M, replace=False)
sub_sc = sc_mat[np.ix_(pick_indices, pick_indices)]
sub_xyz = selected_xyz[pick_indices] 
dist_matrix = squareform(pdist(sub_xyz, metric='euclidean'))

triu_mask = np.triu(np.ones_like(sub_sc, dtype=bool), k=1)
valid_mask = (sub_sc > 0) & triu_mask

dist_data = dist_matrix[valid_mask] 
#analyze_distributions_with_plots(dist_data,'distance',[10,10000],axes[10]);
#analyze_lognormal_distribution(dist_data,500,'distance');
#plt.show();

excess_kurtosis = kurtosis(dist_data, fisher=True);
print('dist_data excess_kurtosis', excess_kurtosis)

regionLabels = ['MON', 'Cb', 'MOS1', 'MOS2', 'MOS3', 'MOS4', 'MOS5', 'IPN', 'IO', 'Hc', 'Ra', 'T', 'aRF', 'imRF', 'pRF', 'GG', 'Hb', 'Hi', 'Hr', 'OG', 'OB', 'OE', 'P', 'Pi', 'PT', 'PO', 'PrT', 'R', 'SP', 'TeO', 'Th', 'TL', 'TS', 'TG', 'VR', 'NX','rMON', 'rCb', 'rMOS1', 'rMOS2', 'rMOS3', 'rMOS4', 'rMOS5', 'rIPN', 'rIO', 'rHc', 'rRa', 'rT', 'raRF', 'rimRF', 'rpRF', 'rGG', 'rHb', 'rHi', 'rHr', 'rOG', 'rOB', 'rOE', 'rP', 'rPi', 'rPT', 'rPO', 'rPrT', 'rR', 'rSP', 'rTeO', 'rTh', 'rTL', 'rTS', 'rTG', 'rVR', 'rNX'];

sort_id=np.load('../raw_data/subject_12_sort_id.npy');
error_data=[];
sel_axis=[4,5,6,7,0,1,2];
kk=0;
for i in range(28):
    pick_num=indices_ii[i];
    
    if(len(pick_num)>500):
        
        sub_sc = sc_mat[np.ix_(pick_num,pick_num)];
        
        sub_degree=np.sum(sub_sc,axis=1);
        excess_kurtosis = kurtosis(sub_degree, fisher=True);
        calculated_skewness = skew(sub_degree);
        log_data=np.log(sub_degree[sub_degree>0]);
        mu, std = np.mean(log_data), np.std(log_data)
        stat_ks, p_ks = stats.kstest(log_data, 'norm', args=(mu, std))
        
        #print(regionLabels[sort_id[i]], excess_kurtosis,calculated_skewness,p_ks);
        #data.append([excess_kurtosis,calculated_skewness  ])
        #if(kk<7):
        #    [log_error,exp_error]=analyze_distributions_with_plots(sub_degree,regionLabels[sort_id[i]],[10,1000],axes[sel_axis[kk]])
        
        print(f'kk : {kk}')
        kk=kk+1;
        error_data.append([ log_error,exp_error ]) ;                  
        #analyze_lognormal_distribution(sub_degree,55,regionLabels[sort_id[i]]);



#plt.figure();
#axes[3].plot(error_data,'o-');
#axes[3].set_xticks([0,1,2,3,4,5,6,7])

y_data=[];
region_kurtosis=[];
for i in range(28):
 
    pick_num=indices_ii[i];
    sub_sc = sc_mat[np.ix_(pick_num,pick_num)];
    
    sub_degree_data=np.sum(sub_sc,axis=1);
    qq=np.sort(sub_degree_data/len(pick_num));
    y_data.extend(qq);

    excess_kurtosis = kurtosis(sub_degree_data/len(pick_num), fisher=False);
   
    #region_kurtosis.append(np.mean(sub_degree_data)/len(pick_num));
    
    region_kurtosis.append(excess_kurtosis);
    #region_kurtosis.append(np.std(sub_degree_data/len(pick_num)));
    
    
    print(i,regionLabels[sort_id[i]], np.mean(sub_degree_data), len(pick_num));

#plt.figure();
#plt.plot(region_kurtosis,'o-');

    
y_data=np.array(y_data);
for i in range(28):
    tt=indices_ii[i];
    axes[6].bar(bar_positions[tt],y_data[tt],color=palette[i], width=1.0,log=True);

axes[6].tick_params(axis='x', pad=1, length=1)
axes[6].tick_params(axis='y', pad=1, length=1)



for ax in axes:
    ax.tick_params(axis='x', pad=1, length=1)
    ax.tick_params(axis='y', pad=1, length=1)
    
    
        
fig.savefig('./fig2.svg',format='svg',dpi=300,transparent=True);

plt.show();




