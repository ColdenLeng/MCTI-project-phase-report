import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

data_dir = '../results/attack_data'
save_dir = '../results/vanilla/HIST'
os.makedirs(save_dir, exist_ok=True)

target_train = np.load(os.path.join(data_dir, 'target_train.npy'))
target_test  = np.load(os.path.join(data_dir, 'target_test.npy'))
loss_idx = 10

member_loss = target_train[:, loss_idx]
nonmember_loss = target_test[:, loss_idx]

clip_max = 0.2
member_loss_clip = member_loss[member_loss < clip_max]
nonmember_loss_clip = nonmember_loss[nonmember_loss < clip_max]

plt.figure(figsize=(8, 6))
plt.scatter(np.zeros_like(member_loss_clip)+0.1, member_loss_clip, alpha=0.6, label='Member (Train)', s=20, c='C0')
plt.scatter(np.zeros_like(nonmember_loss_clip)+0.9, nonmember_loss_clip, alpha=0.6, label='Non-member (Test)', s=20, c='C1')
plt.xticks([0.1, 0.9], ['Member', 'Non-member'])
plt.ylim(0, clip_max)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss Distribution: Member vs Non-member', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'loss_strip_clipped.png'), dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(member_loss_clip, bins=80, alpha=0.7, label='Member (Train)', color='C0', density=True)
plt.hist(nonmember_loss_clip, bins=80, alpha=0.7, label='Non-member (Test)', color='C1', density=True)
plt.xlim(0, clip_max)
plt.xlabel('Loss', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Loss Histogram: Member vs Non-member', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'loss_hist_clipped.png'), dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.boxplot([member_loss_clip, nonmember_loss_clip], labels=['Member', 'Non-member'],
            showmeans=True, meanline=True,
            boxprops=dict(linewidth=2), medianprops=dict(linewidth=2, color='orange'))
plt.ylim(0, clip_max)
plt.ylabel('Loss')
plt.title('Boxplot: Member vs Non-member Loss')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'loss_boxplot_clipped.png'), dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.kdeplot(member_loss_clip, label='Member (Train)', color='C0', bw_adjust=0.4)
sns.kdeplot(nonmember_loss_clip, label='Non-member (Test)', color='C1', bw_adjust=0.4)
plt.xlim(0, clip_max)
plt.xlabel('Loss')
plt.ylabel('Density')
plt.title('KDE: Member vs Non-member (0~0.2)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'loss_kde_clipped.png'), dpi=300)
plt.close()

df = pd.DataFrame({
    'loss': np.concatenate([member_loss_clip, nonmember_loss_clip]),
    'group': ['Member']*len(member_loss_clip) + ['Non-member']*len(nonmember_loss_clip)
})
plt.figure(figsize=(7,4))
sns.violinplot(data=df, x='group', y='loss', scale='width', inner=None)
plt.ylim(0, clip_max)
plt.ylabel('Loss')
plt.title('Loss Violinplot (Main Range)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'loss_violin_clipped.png'), dpi=300)
plt.close()
