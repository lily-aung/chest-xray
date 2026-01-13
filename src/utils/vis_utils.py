from sklearn.metrics import  ConfusionMatrixDisplay
import os
import matplotlib.pyplot as plt 
def ensure_dir(p): 
    os.makedirs(p,exist_ok=True)
def plot_confusion(cm,class_names,out_dir,prefix):
    ensure_dir(out_dir)
    disp=ConfusionMatrixDisplay(cm,class_names)
    fig,ax=plt.subplots(figsize=(6,6)); disp.plot(ax=ax,values_format="d")
    fig.savefig(os.path.join(out_dir,f"{prefix}_confusion.png"),dpi=200); plt.close(fig)