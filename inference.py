import torch
from torch.utils.data import DataLoader
import argparse
from Model import transCSVD
from Dataset import dataset_trans
from collections import OrderedDict

# Add function to remove 'module.' prefix

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# ---------------- Argument Parsing ----------------
parser = argparse.ArgumentParser(description='CSVD classification model inference script')
parser.add_argument('--data_dir', type=str, required=False, default='example_data/', help='Example data folder')
parser.add_argument('--model_path', type=str, required=False, default="example_model/bestv_ci.pth", help='Trained model path')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--out_txt', type=str, default='result.txt', help='Output result txt file name')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data location
PATIENTS_ARCHIVES_path = args.data_dir
# Model location
state_dict_load = torch.load(args.model_path, map_location=device)

# Data index location
testfile_path = "example_data/index.txt"


# ---------------- Dataset and DataLoader ----------------
# Use the specified testfile_path as the patient ID list
patients_txt = testfile_path

test_dataset = dataset_trans.CSVDDataset(
    PATIENTS_ARCHIVES_path=PATIENTS_ARCHIVES_path,
    patient_lst_txt_path=patients_txt,
    use_radiomics=True,
    use_label=True,  # Need label for comparison
    transform=None
)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

# ---------------- Model Loading ----------------
model = transCSVD.TransCSVD(
    element_embedding_dim=24, num_element=100, image_size=None,
    patch_size=4, num_classes=2, depth=8, heads=12, mlp_dim=768, dim_head=64, img_dim=32,
    dropout=0.1, emb_dropout=0.4, load_weight=False, verbose=False
)
model.load_state_dict(remove_module_prefix(state_dict_load['model_state_dict']) if 'model_state_dict' in state_dict_load else remove_module_prefix(state_dict_load), strict=False)
model = model.to(device)
model.eval()

# ---------------- Inference and Output ----------------
with open(args.out_txt, 'w') as fout:
    fout.write('Name\tPred_Label\tProb_0\tProb_1\tCI_Label\tBinary_CI_Label\n')
    with torch.no_grad():
        # Variables for accuracy calculation
        total_samples = 0
        correct_original = 0
        correct_binary = 0
        
        for idx, data in enumerate(test_dataloader):
            radiomics = data['Radiomics'] if 'Radiomics' in data else None
            if radiomics is not None:
                radiomics = [r.to(device) for r in radiomics]
            output, _ = model(radiomics=radiomics, rad_names=data.get('Radiomics_Names', None))
            logits = output[1]
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            names = data['Name'] if 'Name' in data else [f'sample_{idx}']
            ci_labels = data['Label'] if 'Label' in data else torch.full_like(pred, -1)
            
            # Binarize CI labels: set all labels greater than 1 to 1
            binary_ci_labels = ci_labels.clone()
            binary_ci_labels[binary_ci_labels > 1] = 1
            
            # Update accuracy counters
            total_samples += len(names)
            pred_cpu = pred.cpu()  # Move prediction results to CPU
            correct_original += (pred_cpu == ci_labels).sum().item()
            correct_binary += (pred_cpu == binary_ci_labels).sum().item()
            
            for n, p, prob, ci, binary_ci in zip(names, pred.cpu().numpy(), probs.cpu().numpy(), ci_labels.cpu().numpy(), binary_ci_labels.cpu().numpy()):
                fout.write(f'{n}\t{p}\t{prob[0]:.4f}\t{prob[1]:.4f}\t{ci}\t{binary_ci}\n')
                print(f'{n} -> Predicted label: {p}, Probability: [{prob[0]:.4f}, {prob[1]:.4f}], Original CI label: {ci}, Binary CI label: {binary_ci}')
        
        # Calculate and output accuracy
        original_accuracy = correct_original / total_samples if total_samples > 0 else 0
        binary_accuracy = correct_binary / total_samples if total_samples > 0 else 0
        
        print("\n===== Accuracy Statistics =====")
        print(f"Total samples: {total_samples}")
        print(f"Original label accuracy: {correct_original}/{total_samples} = {original_accuracy:.4f} ({original_accuracy*100:.2f}%)")
        print(f"Binary label accuracy: {correct_binary}/{total_samples} = {binary_accuracy:.4f} ({binary_accuracy*100:.2f}%)")
        
        # Write accuracy results to file as well
        fout.write(f"\n===== Accuracy Statistics =====\n")
        fout.write(f"Total samples: {total_samples}\n")
        fout.write(f"Original label accuracy: {correct_original}/{total_samples} = {original_accuracy:.4f} ({original_accuracy*100:.2f}%)\n")
        fout.write(f"Binary label accuracy: {correct_binary}/{total_samples} = {binary_accuracy:.4f} ({binary_accuracy*100:.2f}%)\n") 