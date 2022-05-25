from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.models_init import WakeUpModel, EfficientNet
from utils.utils import *
from utils.data_loaders import LoaderCreator

# wake_up or detector
MODEL_TYPE = 'wake_up'
best_epoch = 20
if MODEL_TYPE == 'wake_up':
    time_folder = '08052022_00-19'
else:
    time_folder = '10052022_02-05'

model = WakeUpModel(n_channel=N_CHANNEL_WAKE_UP)
model.load_state_dict(torch.load(SAVE_MODEL_DIR + MODEL_TYPE + '/' + time_folder + '/' + MODEL_TYPE + '_' + 'epoch_' +
                                 str(best_epoch) + '.pt'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.eval()

visualisation = {}

def hook_fn(m, i, o):
    visualisation[m] = o

def get_all_layers(net):
    for name, layer in net._modules.items():
        layer.register_forward_hook(hook_fn)


get_all_layers(model)


LC = LoaderCreator(DATA_DIR,
                   model_type=MODEL_TYPE,
                   validation=False,
                   test_size=TEST_SIZE,
                   batch_size=BATCH_SIZE,
                   prob=PROB,
                   from_disc=True,
                   seed=SEED)

train_loader, _, test_loader = LC.get_loaders()

preds = []
labels = []
for data, lab in test_loader:
    data, lab = data.to(device), lab.to(device)
    preds.extend(model(data).squeeze().detach().cpu().numpy())
    labels.extend(lab.detach().cpu().numpy())
    break


# Just to check whether we got all layers
print(visualisation.keys())  # output includes sequential layers
import matplotlib.pyplot as plt
import seaborn as sns
for i,k in enumerate(visualisation):
    if i == 6:
        a = visualisation[k]
print(a.shape)
print(data.shape)
print(labels)
sns.heatmap(a[1,0,:,:].detach().cpu().numpy())
plt.show()
sns.heatmap(data[1,0,:,:].detach().cpu().numpy())
plt.show()
