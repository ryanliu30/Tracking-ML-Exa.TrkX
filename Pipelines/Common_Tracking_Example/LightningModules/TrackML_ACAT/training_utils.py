import sys
import yaml
sys.path.append("/global/homes/r/ryanliu/Tracking-ML-Exa.TrkX/Pipelines/Common_Tracking_Example/LightningModules/TrackML_ACAT/")
from EdgeClassifier.Models.IN import EC_InteractionGNN
from GNNEmbedding.Models.IN import Embedding_InteractionGNN
from GNNEmbedding.Models.HGNN_GMM import Embedding_HierarchicalGNN_GMM
from GNNEmbedding.Models.HGNN_HDBSCAN import Embedding_HierarchicalGNN_HDNSCAN
from BipartiteClassification.Models.HGNN_GMM import BC_HierarchicalGNN_GMM
from BipartiteClassification.Models.HGNN_HDBSCAN import BC_HierarchicalGNN_HDBSCAN

path = "/global/homes/r/ryanliu/Tracking-ML-Exa.TrkX/Pipelines/Common_Tracking_Example/LightningModules/TrackML_ACAT/"

def process_hparams(hparams):
    if hparams["use_toy"]:
        hparams["regime"] = []
        hparams["spatial_channels"] = 2
    
    if hparams["hidden"] == "ratio":
        hparams["hidden"] = hparams["hidden_ratio"]*hparams["latent"]
    
    if "cluster_granularity" not in hparams:
        hparams["cluster_granularity"] = 0
    
#     if "gpus" in hparams:
#         hparams["lr"] *= hparams["gpus"]
    
    return hparams

def model_selector(model_name, sweep_configs = {}):
    if model_name == "EC-IN" or model_name == "1":
        with open(path + "EdgeClassifier/Configs/IN.yaml") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)
        model = EC_InteractionGNN(process_hparams({**hparams, **sweep_configs}))
    elif model_name == "Embedding-IN" or model_name == "2":
        with open(path + "GNNEmbedding/Configs/IN.yaml") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader) 
        model = Embedding_InteractionGNN(process_hparams({**hparams, **sweep_configs}))
    elif model_name == "Embedding-HGNN-GMM" or model_name == "3":
        with open(path + "GNNEmbedding/Configs/HGNN_GMM.yaml") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader) 
        model = Embedding_HierarchicalGNN_GMM(process_hparams({**hparams, **sweep_configs}))
    elif model_name == "Embedding-HGNN-HDBSCAN" or model_name == "4":
        with open(path + "GNNEmbedding/Configs/HGNN_HDBSCAN.yaml") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader) 
        model = Embedding_HierarchicalGNN_HDNSCAN(process_hparams({**hparams, **sweep_configs}))
    elif model_name == "BC-HGNN-GMM" or model_name == "5":
        with open(path + "BipartiteClassification/Configs/HGNN_GMM.yaml") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader) 
        model = BC_HierarchicalGNN_GMM(process_hparams({**hparams, **sweep_configs}))
    elif model_name == "BC-HGNN-HDBSCAN" or model_name == "6":
        with open(path + "BipartiteClassification/Configs/HGNN_HDBSCAN.yaml") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader) 
        model = BC_HierarchicalGNN_HDBSCAN(process_hparams({**hparams, **sweep_configs}))        
    else:
        raise ValueError("Can't Find Model Name {}!".format(model_name))
        
    return model
