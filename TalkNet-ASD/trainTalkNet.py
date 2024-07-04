import time, os, torch, argparse, warnings, glob

from dataLoader import TrainLoader, ValLoader
from utils.tools import *
from talkNet import TalkNet

def main():
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "TalkNet Training")
    # Training details
    parser.add_argument('--lr', type=float, default=0.0001,help='Learning rate')
    parser.add_argument('--lrDecay', type=float, default=0.95, help='Learning rate decay rate')
    parser.add_argument('--maxEpoch', type=int, default=25, help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int, default=1, help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize', type=int, default=1000, help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--nDataLoaderThread', type=int, default=4, help='Number of loader threads')
    # Data path
    parser.add_argument('--dataPathAVA', type=str, default="AVDIAR_ASD", help='Save path of AVA dataset')
    parser.add_argument('--savePath', type=str, default="TalkNet-ASD/finetuned")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation only
    parser.add_argument('--downloadAVA', dest='downloadAVA', action='store_true', help='Only download AVA dataset and do related preprocess')
    parser.add_argument('--evaluation', dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model [pretrain_AVA.model]')
    args = parser.parse_args()
    # Data loader
    args = init_args(args)

    if args.downloadAVA == True:
        preprocess_AVA(args)
        quit()

    loader = TrainLoader(trial_file_name=args.trainTrialAVA,
                          audio_path=os.path.join(args.audioPathAVA , 'train'),
                          visual_path=os.path.join(args.visualPathAVA, 'train'),
                          batch_size=args.batchSize
                    )
    trainLoader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=True, num_workers=args.nDataLoaderThread) # type: ignore

    loader = ValLoader(trial_file_name=args.evalTrialAVA,
                        audio_path=os.path.join(args.audioPathAVA , args.evalDataType),
                        visual_path=os.path.join(args.visualPathAVA, args.evalDataType),
                    )
    valLoader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=False, num_workers=16) # type: ignore

    if args.evaluation == True:
        download_pretrain_model_AVA()
        s = TalkNet(**vars(args))
        s.loadParameters('pretrain_AVA.model')
        print("Model %s loaded from previous state!"%('pretrain_AVA.model'))
        mAP = s.evaluate_network(loader=valLoader, **vars(args))
        print(f"mAP {mAP}")
        quit()

    modelFiles = glob.glob('%s/*.model'%args.modelSavePath)
    modelFiles.sort()  
    if len(modelFiles) >= 1:
        print("Model %s loaded from previous state!"%modelFiles[-1])
        epochStart = 1 # int(os.path.splitext(os.path.basename(modelFiles[-1]))[0][6:]) + 1
        s = TalkNet(epoch=epochStart, **vars(args))
        s.loadParameters(modelFiles[-1])
    else:
        epochStart = 1
        s = TalkNet(epoch=epochStart, **vars(args))

    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")

    for epoch in range(epochStart, args.maxEpoch + 1):        
        loss, lr = s.train_network(epoch=epoch, loader=trainLoader)
        
        if epoch % args.testInterval == 0:        
            s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
            mAPs.append(s.evaluate_network(epoch=epoch, loader=valLoader, **vars(args)))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%\n"%(epoch, lr, loss, mAPs[-1], max(mAPs)))
            scoreFile.flush()

if __name__ == '__main__':
    main()
