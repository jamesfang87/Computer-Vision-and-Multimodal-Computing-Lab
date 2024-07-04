import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas
from tqdm import tqdm
from loss import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel

class TalkNet(nn.Module):
    def __init__(self, **kwargs):
        super(TalkNet, self).__init__()     
        self.args = kwargs
        self.model = talkNetModel().cpu()
        self.lossAV = lossAV().cpu()
        self.lossA = lossA().cpu()
        self.lossV = lossV().cpu()
        self.optim = torch.optim.Adam(self.parameters(), lr=self.args['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=self.args['lrDecay'])
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']        
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):

            print(f"audio feature shape: {audioFeature.shape}")
            print(f"visual feature shape: {visualFeature.shape}")

            self.zero_grad()
            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cpu()) # feedForward
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cpu())
            
            print(f"audio embedding shape: {audioEmbed.shape}")
            print(f"visual embeeding shape: {visualEmbed.shape}")

            
            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)

            print(f"audio embedding shape after cross attention: {audioEmbed.shape}")
            print(f"visual embeeding shape after cross attention: {visualEmbed.shape}")

            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)

            print(f"after self attention: {outsAV.shape}")

            outsA = self.model.forward_audio_backend(audioEmbed)
            outsV = self.model.forward_visual_backend(visualEmbed)
            labels = labels[0].reshape((-1)).cpu() # Loss
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)

            print(prec.shape)

            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)
            nloss = nlossAV + 0.4 * nlossA.detach() + 0.4 * nlossV.detach()
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  
        sys.stdout.write("\n")      
        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        for audioFeature, visualFeature, labels in tqdm(loader):
            with torch.no_grad():
                print(f"audio feature shape: {audioFeature.shape}")
                print(f"visual feature shape: {visualFeature.shape}")
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cpu())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cpu())

                print(f"audio embedding shape: {audioEmbed.shape}")
                print(f"visual embeeding shape: {visualEmbed.shape}")

                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)

                print(f"audio embedding shape after cross attention: {audioEmbed.shape}")
                print(f"visual embeeding shape after cross attention: {visualEmbed.shape}")

                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)

                print(f"after self attention: {outsAV.shape}")

                labels = labels[0].reshape((-1)).cpu()             
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)

                # print(predScore)
                print(predScore.shape)

                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
        

        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = pandas.Series(['SPEAKING_AUDIBLE' for _ in evalLines])
        scores = pandas.Series(predScores)

        evalRes = pandas.read_csv(evalOrig)
        evalRes['label'] = labels
        evalRes['score'] = scores
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        
        # change path if working directory is elsewhere
        cmd = "python3 -O TalkNet-ASD/utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, capture_output=True).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location=torch.device('cpu'))
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)

        # transfer learning: only training the fc layer in loss
        for param in self.model.visualFrontend.parameters():
            param.requires_grad = False 

        for param in self.model.visualTCN.parameters():
            param.requires_grad = False

        for param in self.model.visualConv1D.parameters():
            param.requires_grad = False

        for param in self.model.audioEncoder.parameters():
            param.requires_grad = False
        
        for param in self.model.crossA2V.parameters():
            param.requires_grad = False

        for param in self.model.crossV2A.parameters():
            param.requires_grad = False

        for param in self.model.selfAV.parameters():
            param.requires_grad = False