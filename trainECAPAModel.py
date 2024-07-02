import argparse, glob, os, torch, warnings, time
from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel

parser = argparse.ArgumentParser(description="ECAPA_trainer")

# Training Settings
parser.add_argument('--num_frames', type=int, default=200, help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch', type=int, default=80, help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=400, help='Batch size')
parser.add_argument('--n_cpu', type=int, default=4, help='Number of loader threads')
parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] epochs')

# Training and evaluation path/lists, save path
# parser.add_argument('--train_list', type=str, default="/data08/VoxCeleb2/train_list.txt", help='The path of the training list')
# parser.add_argument('--train_path', type=str, default="/data08/VoxCeleb2/train/wav", help='The path of the training data')
parser.add_argument('--eval_list', type=str, default="test-set2.txt", help='The path of the evaluation list')
parser.add_argument('--eval_path', type=str, default="/test-clean", help='The path of the evaluation data')
# parser.add_argument('--musan_path', type=str, default="/data08/Others/musan_split", help='The path to the MUSAN set')
# parser.add_argument('--rir_path', type=str, default="/data08/Others/RIRS_NOISES/simulated_rirs", help='The path to the RIR set')
parser.add_argument('--save_path', type=str, default="exps/exp1", help='Path to save the score.txt and models')
parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')

# Model and Loss settings
parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax')
parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int, default=5994, help='Number of speakers')

# Command
parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')

# Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
s = ECAPAModel(**vars(args))
s = s.to(device)  # Move model to the device

# Only do evaluation, the initial_model is necessary
if args.eval:
    if args.initial_model == "":
        raise ValueError("An initial model must be provided for evaluation.")
    print(f"Model {args.initial_model} loaded from previous state!")
    s.load_parameters(args.initial_model)
    EER, minDCF = s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
    print(f"EER {EER:.2f}%, minDCF {minDCF:.4f}%")
else:
    # Define the data loader
    trainloader = train_loader(**vars(args))
    trainLoader = torch.utils.data.DataLoader(trainloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)

    # Search for the exist models
    modelfiles = glob.glob(f'{args.save_path}/model_0*.model')
    modelfiles.sort()

    # If initial_model is exist, system will train from the initial_model
    if args.initial_model != "":
        print(f"Model {args.initial_model} loaded from previous state!")
        s.load_parameters(args.initial_model)
        epoch = 1
    elif len(modelfiles) >= 1:
        print(f"Model {modelfiles[-1]} loaded from previous state!")
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s.load_parameters(modelfiles[-1])
    else:
        epoch = 1

    EERs = []
    score_file = open(args.save_path + "/score.txt", "a+")

    while epoch <= args.max_epoch:
        # Training for one epoch
        loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader, device=device)

        # Evaluation every [test_step] epochs
        if epoch % args.test_step == 0:
            s.save_parameters(f"{args.save_path}/model_{epoch:04d}.model")
            EERs.append(s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path, device=device)[0])
            print(time.strftime("%Y-%m-%d %H:%M:%S"), f"{epoch} epoch, ACC {acc:.2f}%, EER {EERs[-1]:.2f}%, bestEER {min(EERs):.2f}%")
            score_file.write(f"{epoch} epoch, LR {lr}, LOSS {loss}, ACC {acc:.2f}%, EER {EERs[-1]:.2f}%, bestEER {min(EERs):.2f}%\n")
            score_file.flush()

        epoch += 1

    score_file.close()
