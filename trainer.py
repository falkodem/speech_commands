from pathlib import Path
from sklearn.model_selection import train_test_split
from data_loaders import FirstModelDataset
from torch.utils.data import DataLoader

TRAIN_DIR = Path('SpeechCommands/')

files = sorted(list(TRAIN_DIR.rglob('*.wav')))
labels = [path.parent.name for path in files]
train_files, val_test_files, train_labels, val_test_labels = train_test_split(files,
                                                                              labels,
                                                                              test_size=0.3,
                                                                              stratify=labels,
                                                                              shuffle=True)
val_files, test_files, val_labels, test_labels = train_test_split(val_test_files,
                                                                  val_test_labels,
                                                                  test_size=0.5,
                                                                  stratify=val_test_labels,
                                                                  shuffle=True)

train_dataset = FirstModelDataset(train_files, mode='train', prob=0.5, model_type='wake_up')
val_dataset = FirstModelDataset(val_files, mode='val', prob=0.5, model_type='wake_up')
test_dataset = FirstModelDataset(test_files, mode='test', prob=1, model_type='wake_up')


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print('----------------')
for inputs, labels in test_loader:
    print(inputs.shape)
    print(labels)
