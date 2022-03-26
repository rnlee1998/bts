# from curses import flash
import numpy as np
import matplotlib.pyplot as plt
def cal_var(npz_file):
    d1 = npz_file["d1"]
    d2 = npz_file["d2"]
    d3 = npz_file["d3"]
    silog = npz_file["silog"]
    log10 = npz_file["log10"]
    abs_rel = npz_file["abs_rel"]
    sq_rel = npz_file["sq_rel"]
    rms = npz_file["rms"]
    log_rms = npz_file["log_rms"]
    return [d1.std(),d2.std(),d3.std(),silog.std(),log10.std(),abs_rel.std(),sq_rel.std(),rms.std(),log_rms.std()]
def cal_mean(npz_file):
    d1 = npz_file["d1"]
    d2 = npz_file["d2"]
    d3 = npz_file["d3"]
    silog = npz_file["silog"]
    log10 = npz_file["log10"]
    abs_rel = npz_file["abs_rel"]
    sq_rel = npz_file["sq_rel"]
    rms = npz_file["rms"]
    log_rms = npz_file["log_rms"]
    return [d1.mean(),d2.mean(),d3.mean(),silog.mean(),log10.mean(),abs_rel.mean(),sq_rel.mean(),rms.mean(),log_rms.mean()]

nyu_top5 = np.load('npz_data/nyu_top5.npz')
nyu_top5_4 = np.load('npz_data/nyu_top5_4.npz')
nyu_top5_6 = np.load('npz_data/nyu_top5_6.npz')
nyu_top5_8 = np.load('npz_data/nyu_top5_8.npz')

nyu_bts = np.load('npz_data/nyu_bts_densenet161.npz')
nyu_bts_4 = np.load('npz_data/nyu_bts_densenet_4.npz')
nyu_bts_6 = np.load('npz_data/nyu_bts_densenet_6.npz')
nyu_bts_8 = np.load('npz_data/nyu_bts_densenet_8.npz')

nyu_trans = np.load('npz_data/nyu_transdepth.npz')
nyu_trans_4 = np.load('npz_data/nyu_transdepth_4.npz')
nyu_trans_6 = np.load('npz_data/nyu_transdepth_6.npz')
nyu_trans_8 = np.load('npz_data/nyu_transdepth_8.npz')

kitti_top3 = np.load('npz_data/kitti_top3.npz')
kitti_top3_60 = np.load('npz_data/kitti_top3_60.npz')
kitti_top3_40 = np.load('npz_data/kitti_top3_40.npz')
kitti_top3_20 = np.load('npz_data/kitti_top3_20.npz')

kitti_bts = np.load('npz_data/kitti_bts_densenet161.npz')
kitti_bts_60 = np.load('npz_data/kitti_bts_densenet161_60.npz')
kitti_bts_40 = np.load('npz_data/kitti_bts_densenet161_40.npz')
kitti_bts_20 = np.load('npz_data/kitti_bts_densenet161_20.npz')

kitti_trans = np.load('npz_data/kitti_transdepth.npz')
kitti_trans_60 = np.load('npz_data/kitti_transdepth_60.npz')
kitti_trans_40 = np.load('npz_data/kitti_transdepth_40.npz')
kitti_trans_20 = np.load('npz_data/kitti_transdepth_20.npz')


dataset = "KITTI"
mode = "mean"
plot = False
if dataset=="KITTI" and mode=="Standard deviation":
    x = [20,40,60,80]
    fusion_matrix = np.array([cal_var(kitti_top3_20),cal_var(kitti_top3_40),cal_var(kitti_top3_60),cal_var(kitti_top3)])
    bts_matrix = np.array([cal_var(kitti_bts_20),cal_var(kitti_bts_40),cal_var(kitti_bts_60),cal_var(kitti_bts)])
    trans_matrix =  np.array([cal_var(kitti_trans_20),cal_var(kitti_trans_40),cal_var(kitti_trans_60),cal_var(kitti_trans)])
elif dataset=="KITTI" and mode=="mean":
    x = [20,40,60,80]
    fusion_matrix = np.array([cal_mean(kitti_top3_20),cal_mean(kitti_top3_40),cal_mean(kitti_top3_60),cal_mean(kitti_top3)])
    bts_matrix = np.array([cal_mean(kitti_bts_20),cal_mean(kitti_bts_40),cal_mean(kitti_bts_60),cal_mean(kitti_bts)])
    trans_matrix =  np.array([cal_mean(kitti_trans_20),cal_mean(kitti_trans_40),cal_mean(kitti_trans_60),cal_mean(kitti_trans)])
elif dataset=='NYU'and mode=="Standard deviation":
    x=[4,6,8,10]
    fusion_matrix = np.array([cal_var(nyu_top5_4),cal_var(nyu_top5_6),cal_var(nyu_top5_8),cal_var(nyu_top5)])
    bts_matrix = np.array([cal_var(nyu_bts_4),cal_var(nyu_bts_6),cal_var(nyu_bts_8),cal_var(nyu_bts)])
    trans_matrix =  np.array([cal_var(nyu_trans_4),cal_var(nyu_trans_6),cal_var(nyu_trans_8),cal_var(nyu_trans)])
elif dataset=='NYU'and mode=="mean":
    x=[4,6,8,10]
    fusion_matrix = np.array([cal_mean(nyu_top5_4),cal_mean(nyu_top5_6),cal_mean(nyu_top5_8),cal_mean(nyu_top5)])
    bts_matrix = np.array([cal_mean(nyu_bts_4),cal_mean(nyu_bts_6),cal_mean(nyu_bts_8),cal_mean(nyu_bts)])
    trans_matrix =  np.array([cal_mean(nyu_trans_4),cal_mean(nyu_trans_6),cal_mean(nyu_trans_8),cal_mean(nyu_trans)])
if plot:
    plt.figure(figsize=(5,4.3))
    plt.plot(x,fusion_matrix[:,7],color="deeppink",linewidth=2,linestyle='-.',label='TEDepth(3)', marker='d')
    plt.plot(x,bts_matrix[:,7],color="k",linewidth=2,linestyle='--',label='BTS', marker='d')
    plt.plot(x,trans_matrix[:,7],color="m",linewidth=2,linestyle=':',label='TransDepth', marker='d')
    plt.xlabel("Max capped depth (m)")
    if mode=="Standard deviation":
        plt.ylabel("RMSE (%s)"%(mode))
    else:
        plt.ylabel("RMSE ")
    plt.legend(loc=2)
    plt.grid()
    plt.savefig('%s_%s.png'%(mode,dataset),dpi=600)
    plt.show()

nyu_matrix = np.array([cal_var(nyu_trans),cal_var(nyu_bts),cal_var(nyu_top5)])
kitti_matrix = np.array([cal_var(kitti_trans),cal_var(kitti_bts),cal_var(kitti_top3)])
f = open("kitti_std.txt",'w')
for i in range(3):
    for j in range(9):
        f.write('%.6f'%(kitti_matrix[i][j])+'&')
    f.write('\n')
f.close()