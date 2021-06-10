import itertools
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from dataset import Datasets
from tensorboardX import SummaryWriter
from SGU_Net import SGU_Net
from metrics import *
from autoED import conv_encoder,conv_decoder
from utils import *
from losses import DC_and_HDBinary_loss

writer_train = SummaryWriter("GD_run/train")
writer_val = SummaryWriter("GD_run/val")
wirter_all = SummaryWriter("GD_run/all")

# 是否使用cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),   
    transforms.Normalize([0.5], [0.5])
])

def y_transforms(x):
    img = np.asarray(x)
    img = torch.from_numpy(img)
    img = img.type(torch.LongTensor)
    return img

def test_model(model,criterion,dataload):
    model = model.eval()
    model = model.to(device)

    total_loss = 0
    total_acc = 0
    dt_size = len(dataload.dataset)
    for x,y in dataload:
        inputs = x.to(device)
        labels = y.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        predictions_all = outputs.argmax(dim=1)
        predictions_all = predictions_all.squeeze().cpu().detach().numpy()
        lab = labels.squeeze().cpu().detach().numpy()
        
        iou = Dice(predictions_all,lab)

        total_loss = total_loss + loss.item()
        total_acc = total_acc + iou

    print("ave_test_Dice:{},ave_test_loss:{}".format(total_acc/dt_size,total_loss/dt_size))

    return total_acc/dt_size,total_loss/dt_size

def train_cir():

    model = SGU_Net(1,2).to(device)
    encoder = conv_encoder().to(device)
    decoder = conv_decoder().to(device)

    batch_size = 8
    num_epochs = 100
    criterion = DC_and_HDBinary_loss()
    shape_criterion = torch.nn.L2Loss()
    reconstruction_crition = DC_and_HDBinary_loss()

    optimizer_G = optim.Adam(model.parameters(),lr=0.0002,betas=(0.5, 0.999))
    optimizer_D =torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.0002,betas=(0.5, 0.999))#itertools.chain可以将将参数链接起来
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(num_epochs, 0, num_epochs//2).step  # lr_lambda为操作学习率的函数
    )
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D, lr_lambda=LambdaLR(num_epochs, 0, num_epochs//2).step  # lr_lambda为操作学习率的函数
    )

    train_iter = 0

    data_dataset = Datasets("","",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(data_dataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        train_iter += 1
        dt_size = len(dataloaders.dataset)
        epoch_G_loss = 0
        epoch_D_loss = 0
        tot_acc = 0
        step = 0

        model = model.train()

        for x, y in dataloaders:
            step += 1

            inputs = x.to(device)
            labels = y.to(device)

            optimizer_G.zero_grad()
            outputs = model(inputs)
            loss_G = criterion(outputs, labels)

            pred_ou = outputs.clone()
            pred_ou = pred_ou.argmax(dim=1).unsqueeze(1)
            pred_ou = pred_ou.float()

            lab = labels.clone()
            lab = lab.unsqueeze(1)
            lab = lab.float()

            pre_shape_vector = encoder(pred_ou)
            pre_res_outputs = decoder(pre_shape_vector)
            labels_shape_vector = encoder(lab)
            labels_res_outputs = decoder(labels_shape_vector)
            loss_shape = shape_criterion(pre_shape_vector,labels_shape_vector)
            loss_totalG = loss_G + 5*loss_shape
            loss_totalG.backward(retain_graph=True)
            optimizer_G.step()

            ##train D
            optimizer_D.zero_grad()
            outputs_copy = pred_ou.clone()
            outputs_copy = outputs_copy.float()

            blab = labels.clone()
            blab = blab.unsqueeze(1)
            blab = blab.float()

            loss_res1 = reconstruction_crition(pre_res_outputs,outputs_copy.detach())
            loss_res2 = reconstruction_crition(labels_res_outputs,blab)
            loss_totalD = -(-loss_res1-loss_res2+0.001*loss_shape)
            loss_totalD.backward()
            optimizer_D.step()

            epoch_G_loss += float(loss_totalG.item())
            epoch_D_loss += float((loss_totalD.item()))

            pred = outputs.argmax(dim=1)
            train_acc = comput_Dice(pred.to("cpu"),labels.to("cpu")).item()

            tot_acc += train_acc
            print("%d/%d,train_G_loss:%f,train_D_loss:%f,train_acc:%f" % (step, (dt_size - 1) // dataloaders.batch_size + 1, loss_totalG.item(),loss_totalD.item(),train_acc))

        lr_scheduler_G.step()
        lr_scheduler_D.step()
        print("epoch %d, ave_G_loss:%f,ave_D_loss:%f,ave_acc:%f" % (epoch+1, epoch_G_loss/((dt_size - 1) / dataloaders.batch_size + 1),epoch_G_loss/((dt_size - 1) / dataloaders.batch_size + 1),tot_acc/((dt_size - 1) / dataloaders.batch_size + 1)))

        writer_train.add_scalar("train_G_loss",epoch_G_loss/((dt_size - 1) / dataloaders.batch_size + 1),train_iter)
        writer_train.add_scalar("train_D_loss", epoch_D_loss / ((dt_size - 1) / dataloaders.batch_size + 1), train_iter)
        writer_train.add_scalar("train_acc",tot_acc/((dt_size - 1) / dataloaders.batch_size + 1),train_iter)

        with torch.no_grad():
            data_dataset_test = Datasets("","", transform=x_transforms,target_transform=y_transforms)
            dataloaders_test = DataLoader(data_dataset_test, batch_size=1)
            criterion = DC_and_HDBinary_loss()
            model = model.eval()
            iou,loss = test_model(model,criterion,dataloaders_test)

        writer_val.add_scalar("val_loss",loss,train_iter)
        writer_val.add_scalar("val_acc",iou,train_iter)

        wirter_all.add_scalars("loss",{'train_G_loss':epoch_G_loss/((dt_size - 1) / dataloaders.batch_size + 1),'train_D_loss':epoch_D_loss/((dt_size - 1) / dataloaders.batch_size + 1),'val_loss':loss},train_iter)
        wirter_all.add_scalars("acc",{'train_acc':tot_acc/((dt_size - 1) / dataloaders.batch_size + 1),'val_acc':iou},train_iter)

        torch.save(model.state_dict(),'ckp/SGU_Netepoch{}G.pth'.format(epoch+1))
        torch.save(encoder.state_dict(), 'ckp/SGU_Netepoch{}DE.pth'.format(epoch + 1))
        torch.save(decoder.state_dict(), 'ckp/SGU_Netepoch{}DD.pth'.format(epoch + 1))



