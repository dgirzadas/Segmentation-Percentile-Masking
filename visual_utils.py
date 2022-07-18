import numpy as np
import torch
from utils import probscale
from matplotlib import pyplot as plt


"""
Visualises the input image
"""
def imshow(img):
    npimg = np.array(img)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')


"""
Converts raw stage outputs of the model into visualisable set of variables
"""
def process_output(outputs1, outputs2, outputs3, mask1, mask2):
    mask1 = mask1.squeeze(1)
    mask2 = mask2.squeeze(1)

    # Compute the prediction confidences of each stage
    confs1 = torch.max(probscale(outputs1).detach().cpu(), dim=1)[0].squeeze(0)
    confs2 = torch.max(probscale(outputs2).detach().cpu(), dim=1)[0].squeeze(0)
    confs3 = torch.max(probscale(outputs3).detach().cpu(), dim=1)[0].squeeze(0)

    # Form the masked outputs of each stage
    out1 = ((torch.argmax(outputs1, dim=1)) * (1 - mask1)).detach().cpu().squeeze(0)
    out2 = ((torch.argmax(outputs2, dim=1)) * mask1 * (1 - mask2)).detach().cpu().squeeze(0)
    out3 = ((torch.argmax(outputs3, dim=1)) * mask1 * mask2).detach().cpu().squeeze(0)

    # Combine the stage outputs into a complete image
    preds = out1 + out2 + out3

    # Format the stage masks into a convenient shape
    mask1 = mask1.type(torch.LongTensor).cuda()[0]
    mask2 = mask2.type(torch.LongTensor).cuda()[0]

    return confs1, confs2, confs3, out1, out2, out3, mask1, mask2, preds


"""
Plots the prediction confidence, accepted regions and accepted predictions of each stage
"""
def plot_stages(inputs, confs1, confs2, confs3, mask1, mask2, out1, out2, out3):
    fig, ax = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)

    # Stage 1
    ax[0, 0].imshow(inputs.cpu().transpose(0, 1).transpose(1, 2))
    im1 = ax[0, 0].imshow(confs1, alpha=0.5)
    im1.set_clim(0, 1)
    ax[0, 0].set_title("Prediction Confidence")
    ax[0, 0].axis('off')
    ax[0, 1].imshow(((1 - mask1).repeat(3, 1, 1).detach().cpu() * inputs.cpu()).transpose(0, 1).transpose(1, 2))
    ax[0, 1].set_title("Accepted Regions")
    ax[0, 1].axis('off')
    ax[0, 2].imshow(inputs.cpu().transpose(0, 1).transpose(1, 2))
    alphas = (out1 != 0).type(torch.FloatTensor)
    pr1 = ax[0, 2].imshow(out1, cmap='tab10', alpha=alphas * 0.3)
    pr1.set_clim(0, 10)
    ax[0, 2].set_title("Accepted Predictions")
    ax[0, 2].axis('off')
    plt.colorbar(im1, ax=ax[0, 0], location='left', fraction=0.03)

    # Stage 2
    ax[1, 0].imshow(inputs.cpu().transpose(0, 1).transpose(1, 2))
    im2 = ax[1, 0].imshow(confs2, alpha=0.5)
    im2.set_clim(0, 1)
    ax[1, 0].axis('off')
    ax[1, 1].imshow(
        ((mask1 * (1 - mask2)).repeat(3, 1, 1).detach().cpu() * inputs.cpu()).transpose(0, 1).transpose(1, 2))
    ax[1, 1].axis('off')
    ax[1, 2].imshow(inputs.cpu().transpose(0, 1).transpose(1, 2))
    alphas = (out2 != 0).type(torch.FloatTensor)
    pr2 = ax[1, 2].imshow(out2, cmap='tab10', alpha=alphas * 0.3)
    pr2.set_clim(0, 10)
    ax[1, 2].axis('off')
    plt.colorbar(im2, ax=ax[1, 0], location='left', fraction=0.03)

    # Stage 3
    ax[2, 0].imshow(inputs.cpu().transpose(0, 1).transpose(1, 2))
    im3 = ax[2, 0].imshow(confs3, alpha=0.5)
    im3.set_clim(0, 1)
    ax[2, 0].axis('off')
    ax[2, 1].imshow(((mask1 * mask2).repeat(3, 1, 1).detach().cpu() * inputs.cpu()).transpose(0, 1).transpose(1, 2))
    ax[2, 1].axis('off')
    ax[2, 2].imshow(inputs.cpu().transpose(0, 1).transpose(1, 2))
    alphas = (out3 != 0).type(torch.FloatTensor)
    pr3 = ax[2, 2].imshow(out3, cmap='tab10', alpha=alphas * 0.3)
    pr3.set_clim(0, 10)
    ax[2, 2].axis('off')
    plt.colorbar(im3, ax=ax[2, 0], location='left', fraction=0.03)


"""
Plots the ground truth and prediction side-by-side
"""
def plot_results(inputs, preds, labels):
    # Input, Prediction, Ground Truth:
    fig, (gt, pr) = plt.subplots(1, 2, figsize=(12,12), constrained_layout = True)
    gt.imshow(inputs.cpu().transpose(0,1).transpose(1,2))
    gt.set_title("Ground Truth")
    gt.axis('off')
    gt.imshow(labels.cpu(), cmap='tab10', alpha = 0.3)

    pr.imshow(inputs.cpu().transpose(0,1).transpose(1,2))
    pr.set_title("Prediction")
    pr.axis('off')
    pr_end = pr.imshow(preds, cmap='tab10', alpha = 0.3)
    pr_end.set_clim(0,10)
