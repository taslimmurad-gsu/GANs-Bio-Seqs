from collections import Counter
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

attr = np.load("path of target labels .npy file")
seqs =  np.load("path of sequences .npy file")
seqs = list(seqs)
attr = list(attr)
print('len ohe', len(seqs))
print('len attr', len(attr))

x1_pssm2vec = np.array(seqs)
y1_pssm2vec = np.array(attr)
x1_pssm2vec = seqs
y1_pssm2vec = attr

df2_pssm2vec = pd.DataFrame()
df2_pssm2vec["y"] = y1_pssm2vec
cnt_pssm2vec = len (df2_pssm2vec['y'].unique())
tsne_pssm2vec = TSNE(n_components=2, verbose=1, random_state=123,perplexity=50)
z1_pssm2vec = tsne_pssm2vec.fit_transform(x1_pssm2vec)
df2_pssm2vec["comp-1"] = z1_pssm2vec[:,0]
df2_pssm2vec["comp-2"] = z1_pssm2vec[:,1]
az = df2_pssm2vec["y"]
X_embedded_orig_final2 = np.array(z1_pssm2vec)
colors = cm.rainbow(np.linspace(0, 1, len(np.unique(az))+10))
len(np.unique(az))
hst = np.unique(az)
order_top_hosts =  az
colors_top_hosts = {hst[0] : colors[0],
                    hst[1] : colors[1],
                    hst[2] : colors[2],
                    hst[3]: colors[3],
                    hst[4] : colors[4],
                    }

data_frame_top_hosts = pd.DataFrame({' ':
                                   np.array(X_embedded_orig_final2[:,0]),
                           '.': np.array(X_embedded_orig_final2[:,1]),
                           'Variants:': az})

sns.set(rc={'figure.figsize':(22,19), 'axes.facecolor':'white', 'figure.facecolor':'white'})

sns.scatterplot(x = ' ',
            y = '.',
            hue = 'Variants:',
            hue_order = order_top_hosts,
            palette = colors_top_hosts,
            data = data_frame_top_hosts,
            s = 240,
            linewidth=0.01,
            style = 'Variants:')

plt.tick_params(labelsize=55)
lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=20, mode = "expand", ncol = 6)
for handle in lgnd.legendHandles:
    handle.set_sizes([526.0])
plt.legend().remove()
write_path = "path to save .png tsne file"
plt.savefig(write_path)
